import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup


# =================配置参数=================
class Config:
    model_name = "pretrained_models/roberta-base"  # 基础模型，可换成 roberta-base-chinese
    data_path = r"data/地质知识-对比学习-指定.jsonl"  # 您的数据路径
    output_dir = "gk/output/geology_contrastive_bert"
    max_length = 256  # 文本最大长度
    batch_size = 16 * 4  # 根据显存调整
    epochs = 20
    learning_rate = 2e-5
    temperature = 0.05  # 对比学习温度系数
    seed = 42
    use_bf16 = True  # 开启 BF16


# =================1. 数据集定义=================
class GeologyDataset(Dataset):
    def __init__(self, data_path):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                # 提取三元组：原始文本(Anchor)，正样本(Pos)，负样本(Hard Neg)
                self.samples.append({"anchor": item["原始文本"], "positive": item["正样本"], "negative": item["负样本"]})
        print(f"已加载 {len(self.samples)} 条地质三元组数据。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class GeologyCollator:
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        anchors = [b["anchor"] for b in batch]
        positives = [b["positive"] for b in batch]
        negatives = [b["negative"] for b in batch]

        # 将三者拼接在一起进行一次性Tokenize，提高效率
        # 顺序：[Anchor_1, ..., Anchor_N, Pos_1, ..., Pos_N, Neg_1, ..., Neg_N]
        all_texts = anchors + positives + negatives

        encoded = self.tokenizer(all_texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        return encoded


# =================2. 模型定义=================
class GeologyContrastiveModel(nn.Module):
    def __init__(self, model_name, temperature=0.05):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.temperature = temperature

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        # outputs.last_hidden_state shape: [Batch, Seq_Len, Hidden]
        token_embeddings = outputs.last_hidden_state

        # 1. 扩展 mask 以匹配 embedding 维度 [Batch, Seq_Len, 1]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # 2. 计算加权和 (去除 padding 的影响)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        # 3. 计算有效 token 数量 (防止除以0)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # 4. 平均池化
        sentence_embeddings = sum_embeddings / sum_mask

        # 5. 归一化
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

    def compute_loss(self, embeddings, batch_size):
        """
        计算带有Hard Negative的InfoNCE Loss
        embeddings shape: [3 * Batch_Size, Hidden_Dim]
        结构顺序: [Anchors, Positives, Negatives]
        """
        # 拆分向量
        anchors, positives, negatives = torch.split(embeddings, batch_size, dim=0)

        # 1. 计算 Anchor 和 Positive 的相似度
        # [Batch_Size, Batch_Size]
        sim_a_p = torch.matmul(anchors, positives.T) / self.temperature

        # 2. 计算 Anchor 和 Negative (Hard Negative) 的相似度
        # [Batch_Size, Batch_Size]
        sim_a_n = torch.matmul(anchors, negatives.T) / self.temperature

        # 3. 构建Logits
        # 我们希望 Anchor[i] 与 Positive[i] 匹配 (对角线)
        # 负样本包括：Batch内的其他Positives，以及所有的Negatives (Hard Negatives)

        # 拼接 logits: [Batch_Size, 2 * Batch_Size]
        # 左半部分是 Anchor vs Positives，右半部分是 Anchor vs Negatives
        logits = torch.cat([sim_a_p, sim_a_n], dim=1)

        # 标签：对于第 i 个 anchor，正确的 positive 索引就是 i
        labels = torch.arange(batch_size).to(logits.device)

        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)

        return loss

    def save_pretrained(self, output_dir):
        # 只需要保存内部的 BERT，这样下游任务可以直接 AutoModel.from_pretrained 加载
        self.bert.save_pretrained(output_dir)


# =================3. 训练流程=================
def train():
    # 环境设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 随机种子
    torch.manual_seed(Config.seed)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(Config.model_name)

    # 数据加载
    dataset = GeologyDataset(Config.data_path)
    collator = GeologyCollator(tokenizer, Config.max_length)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, collate_fn=collator)

    # 模型
    model = GeologyContrastiveModel(Config.model_name, Config.temperature).to(device)
    model.train()

    # 优化器
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
    total_steps = len(dataloader) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # BF16 Scaler (虽然bf16通常不需要scaler，但为了代码健壮性可保留，或者直接用autocast)
    # 注意：BFloat16 只有在 Ampere (RTX3090/A100) 及以上架构支持。如果是旧卡请改为 float16
    dtype = torch.bfloat16 if Config.use_bf16 and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Training precision: {dtype}")

    # 训练循环
    for epoch in range(Config.epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{Config.epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # 当前Batch的实际大小 (最后一个batch可能不足设定值)
            # 输入的总行数是 3 * current_batch_size
            current_batch_size = input_ids.size(0) // 3

            # 混合精度前向传播
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                embeddings = model(input_ids, attention_mask)
                loss = model.compute_loss(embeddings, current_batch_size)

            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

    # =================4. 保存模型=================
    if not os.path.exists(Config.output_dir):
        os.makedirs(Config.output_dir)

    print("Saving model and tokenizer...")
    # 保存模型权重 (Standard Hugging Face Format)
    model.save_pretrained(Config.output_dir)
    # 保存tokenizer
    tokenizer.save_pretrained(Config.output_dir)
    print(f"Model saved to {Config.output_dir}")


if __name__ == "__main__":
    train()
