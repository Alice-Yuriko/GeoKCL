import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, get_linear_schedule_with_warmup
import os
from tqdm import tqdm

# =================配置参数=================
CONFIG = {
    "model_name": "pretrained_models/roberta-base",
    "train_file": "data/地质知识-对比学习-指定.jsonl",
    "output_dir": "gk/知识嵌入+规则对齐/geology_contrastive_bert",
    "max_length": 256,
    "batch_size": 16 * 3,  # 根据显存调整
    "epochs": 10,
    "learning_rate": 2e-5,
    "temperature": 0.05,  # 对比学习温度系数
    "lambda_know": 1.0,  # 知识对齐损失的权重
    "lambda_mlm": 0.5,  # MLM 损失权重 (通常设为 0.1~1.0)
    "seed": 42,
}


# =================环境设置=================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =================数据处理=================
class GeologyDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        print(f"载入数据: {len(self.data)} 条")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {"anchor": item["原始文本"], "pos": item["正样本"], "neg": item["负样本"], "rule": item["地质知识"], "rule_id": int(item["地质知识编号"])}


def collate_fn(batch):
    anchors = [item["anchor"] for item in batch]
    positives = [item["pos"] for item in batch]
    negatives = [item["neg"] for item in batch]
    rules = [item["rule"] for item in batch]
    rule_ids = torch.tensor([item["rule_id"] for item in batch], dtype=torch.long)
    return anchors, positives, negatives, rules, rule_ids


# =================MLM 掩码工具=================
def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """
    标准的 BERT Masking 策略
    """
    labels = inputs.clone()
    # 我们不计算 PAD 的 loss
    probability_matrix = torch.full(labels.shape, mlm_probability).to(inputs.device)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool).to(inputs.device)

    # 将特殊字符 ([CLS], [SEP], [PAD]) 的 mask 概率设为 0
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    # 只有被 mask 的地方才需要计算 loss，其他地方设为 -100 (CrossEntropyLoss 的 ignore_index)
    labels[~masked_indices] = -100

    # 80% 替换为 [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8).to(inputs.device)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% 替换为随机 Token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5).to(inputs.device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long).to(inputs.device)
    inputs[indices_random] = random_words[indices_random]

    # 剩余 10% 保持原样 (但仍需预测)
    return inputs, labels


# =================模型定义=================
class GeologyMultiTaskModel(nn.Module):
    def __init__(self, model_name):
        super(GeologyMultiTaskModel, self).__init__()
        # 使用 BertForMaskedLM，它包含 Encoder 和 MLM Head
        self.bert_mlm = BertForMaskedLM.from_pretrained(model_name)

        hidden_size = self.bert_mlm.config.hidden_size

        # 对比学习专用的 Projection Head
        self.cl_projection = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 128))

    def forward_mlm(self, input_ids, attention_mask, labels):
        # BertForMaskedLM 的 forward 会自动计算 loss (如果提供了 labels)
        outputs = self.bert_mlm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    def forward_cl(self, input_ids, attention_mask):
        # 获取 Encoder 的输出用于对比学习
        # base_model 即为内部的 BertModel
        outputs = self.bert_mlm.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Mean Pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # 训练时通过 Projection Head
        if self.training:
            return self.cl_projection(mean_pooled)
        else:
            return mean_pooled

    def save_pretrained(self, path):
        # 保存整个 BertForMaskedLM
        # 这样模型既包含了学到的 Embedding，也保留了适应过地质领域的 MLM Head (可选用于进一步微调)
        self.bert_mlm.save_pretrained(path)


# =================对比损失函数=================
class MultiGranularityLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(MultiGranularityLoss, self).__init__()
        self.temp = temperature

    def forward(self, h_anchor, h_pos, h_neg, h_rule, rule_ids):
        # 归一化
        h_anchor = F.normalize(h_anchor, p=2, dim=1)
        h_pos = F.normalize(h_pos, p=2, dim=1)
        h_neg = F.normalize(h_neg, p=2, dim=1)
        h_rule = F.normalize(h_rule, p=2, dim=1)

        batch_size = h_anchor.shape[0]

        # --- Instance Loss ---
        sim_pos = torch.sum(h_anchor * h_pos, dim=1) / self.temp
        sim_neg_hard = torch.sum(h_anchor * h_neg, dim=1) / self.temp

        # In-batch negatives
        sim_matrix = torch.mm(h_anchor, h_anchor.t()) / self.temp
        mask = torch.eye(batch_size, device=h_anchor.device).bool()
        sim_matrix.masked_fill_(mask, -1e9)

        # Pos + HardNeg + InBatchNegs
        neg_logits = torch.cat([sim_neg_hard.unsqueeze(1), sim_matrix], dim=1)
        logits_inst = torch.cat([sim_pos.unsqueeze(1), neg_logits], dim=1)
        labels_inst = torch.zeros(batch_size, dtype=torch.long, device=h_anchor.device)
        loss_inst = F.cross_entropy(logits_inst, labels_inst)

        # --- Knowledge Loss ---
        sim_rule_pos = torch.sum(h_anchor * h_rule, dim=1) / self.temp
        sim_matrix_rule = torch.mm(h_anchor, h_rule.t()) / self.temp

        # Mask same rule in batch
        rule_matches = rule_ids.unsqueeze(0) == rule_ids.unsqueeze(1)
        sim_matrix_rule.masked_fill_(rule_matches, -1e9)

        logits_know = torch.cat([sim_rule_pos.unsqueeze(1), sim_matrix_rule], dim=1)
        labels_know = torch.zeros(batch_size, dtype=torch.long, device=h_anchor.device)
        loss_know = F.cross_entropy(logits_know, labels_know)

        return loss_inst, loss_know


# =================训练流程=================
def train():
    tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])
    dataset = GeologyDataset(CONFIG["train_file"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)

    model = GeologyMultiTaskModel(CONFIG["model_name"])
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    criterion_cl = MultiGranularityLoss(temperature=CONFIG["temperature"])

    scaler = torch.amp.GradScaler()

    total_steps = len(dataloader) * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    print("开始多任务训练 (CL + MLM)...")

    for epoch in range(CONFIG["epochs"]):
        total_loss_avg = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}")

        for anchors, positives, negatives, rules, rule_ids in loop:
            # --- 步骤 1: 准备 MLM 输入 (仅使用 Anchor) ---
            # 我们希望模型学习原始地质文本的 Token 分布
            mlm_encodings = tokenizer(anchors, padding=True, truncation=True, max_length=CONFIG["max_length"], return_tensors="pt")
            input_ids_mlm = mlm_encodings["input_ids"].to(device)
            att_mask_mlm = mlm_encodings["attention_mask"].to(device)

            # 创建 Mask
            input_ids_masked, labels_mlm = mask_tokens(input_ids_mlm.clone(), tokenizer)

            # --- 步骤 2: 准备 CL 输入 (Anchor, Pos, Neg, Rule) ---
            all_texts_cl = anchors + positives + negatives + rules
            cl_encodings = tokenizer(all_texts_cl, padding=True, truncation=True, max_length=CONFIG["max_length"], return_tensors="pt")
            input_ids_cl = cl_encodings["input_ids"].to(device)
            att_mask_cl = cl_encodings["attention_mask"].to(device)

            rule_ids = rule_ids.to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 任务 A: Masked Language Modeling
                loss_mlm = model.forward_mlm(input_ids_masked, att_mask_mlm, labels_mlm)

                # 任务 B: Contrastive Learning
                all_embeddings = model.forward_cl(input_ids_cl, att_mask_cl)

                # 拆分向量
                bs = len(anchors)
                h_anchor = all_embeddings[:bs]
                h_pos = all_embeddings[bs : 2 * bs]
                h_neg = all_embeddings[2 * bs : 3 * bs]
                h_rule = all_embeddings[3 * bs :]

                loss_inst, loss_know = criterion_cl(h_anchor, h_pos, h_neg, h_rule, rule_ids)

                # 总损失
                loss_cl_total = loss_inst + CONFIG["lambda_know"] * loss_know
                loss_total = loss_cl_total + CONFIG["lambda_mlm"] * loss_mlm

            # 反向传播
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss_avg += loss_total.item()
            loop.set_postfix(total=loss_total.item(), cl=loss_cl_total.item(), mlm=loss_mlm.item())

        print(f"Epoch {epoch + 1} Average Loss: {total_loss_avg / len(dataloader):.4f}")

    # =================保存模型=================
    if not os.path.exists(CONFIG["output_dir"]):
        os.makedirs(CONFIG["output_dir"])

    print(f"保存模型到 {CONFIG['output_dir']} ...")
    model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    print("保存完成。")


if __name__ == "__main__":
    train()
