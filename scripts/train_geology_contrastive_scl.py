import argparse
import json
import os
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import BertConfig, BertForMaskedLM, BertTokenizer, get_linear_schedule_with_warmup

# ==========================================
# 0. 辅助工具：N*M 采样器与早停机制
# ==========================================


class BalancedBatchSampler(Sampler):
    """
    N*M 采样器：
    每次采样 n_classes 个类别，每个类别抽取 m_samples 个样本。
    Batch Size = n_classes * m_samples
    """

    def __init__(self, labels, n_classes, m_samples):
        self.labels = np.array(labels)
        self.n_classes = n_classes
        self.m_samples = m_samples
        self.label_set = list(set(self.labels))

        # 构建 {label: [indices]} 字典
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        # 移除样本数过少的类别（或者提示警告）
        # 这里采用简单的循环采样策略（如有不足则重复采样）

        # 计算一个epoch大约有多少个batch (向下取整)
        self.num_batches = len(self.labels) // (self.n_classes * self.m_samples)
        if self.num_batches == 0:
            self.num_batches = 1  # 至少跑一个

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices = []
            # 1. 随机选择 N 个类别
            if len(self.label_set) < self.n_classes:
                selected_classes = np.random.choice(self.label_set, self.n_classes, replace=True)
            else:
                selected_classes = np.random.choice(self.label_set, self.n_classes, replace=False)

            # 2. 每个类别选择 M 个样本
            for label in selected_classes:
                indices = self.label_to_indices[label]
                # 如果该类样本不够 M 个，则允许重复采样
                replace = len(indices) < self.m_samples
                selected_indices = np.random.choice(indices, self.m_samples, replace=replace)
                batch_indices.extend(selected_indices)

            # 3. 如果需要 shuffle batch 内部（通常不需要，DataLoader 会处理，但为了保险打乱一下顺序）
            # 注意：SupCon 实际上不需要 batch 内部 shuffle，但为了 BERT Batch Norm 更好，可以 shuffle
            # 这里返回的是 indices list
            yield batch_indices

    def __len__(self):
        return self.num_batches


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=3, delta=0.001, path="checkpoint.pt", verbose=False):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, tokenizer):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, tokenizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, tokenizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, tokenizer):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        # 保存最优模型
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        model.bert_lm.save_pretrained(self.path)
        tokenizer.save_pretrained(self.path)
        self.val_loss_min = val_loss


# ==========================================
# 1. 损失函数 (保持不变)
# ==========================================


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(similarity_matrix / self.temperature) * logits_mask
        log_prob = similarity_matrix / self.temperature - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        mask_sum = mask.sum(1)
        # 避免除以0：如果某个样本在 batch 里没有正样本对，mask_sum为0
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # 只计算存在正样本对的 loss
        loss = -mean_log_prob_pos
        # 如果整个 batch 都没有正样本对 (极少见情况)，loss设为0
        loss = loss.mean()
        return loss


class KGInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(KGInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, anchor, positive, negative):
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)
        pos_score = self.cos(anchor, positive) / self.temperature
        neg_score = self.cos(anchor, negative) / self.temperature
        logits = torch.cat([pos_score.unsqueeze(1), neg_score.unsqueeze(1)], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)


# ==========================================
# 2. 数据集与 Collator
# ==========================================


class GeologyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # 收集 Relation 用于 Sampler
        self.relations = sorted(list(set([d["relation"] for d in self.data])))
        self.rel2id = {r: i for i, r in enumerate(self.relations)}
        # 预先计算所有样本的 label id 供 sampler 使用
        self.labels = [self.rel2id[d["relation"]] for d in self.data]
        print(f"Loaded {len(self.data)} samples. Relations: {len(self.relations)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "anchor": item["原始文本"],
            "positive": item["正样本"],
            "negative": item["负样本"],
            "knowledge": item["地质知识"],
            "relation_id": self.rel2id.get(item["relation"], -1),
        }


class GeologyCollator:
    def __init__(self, tokenizer, max_len=128, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_probability = mlm_probability

    def __call__(self, batch):
        anchors = [b["anchor"] for b in batch]
        positives = [b["positive"] for b in batch]
        negatives = [b["negative"] for b in batch]
        knowledge = [b["knowledge"] for b in batch]
        relation_ids = torch.tensor([b["relation_id"] for b in batch], dtype=torch.long)

        enc_anchor = self.tokenizer(anchors, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt", return_special_tokens_mask=True)
        enc_pos = self.tokenizer(positives, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        enc_neg = self.tokenizer(negatives, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        enc_know = self.tokenizer(knowledge, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")

        mlm_inputs = enc_anchor.input_ids.clone()
        mlm_labels = mlm_inputs.clone()
        special_tokens_mask = enc_anchor.special_tokens_mask.bool()
        probability_matrix = torch.full(mlm_labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        mlm_labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(mlm_labels.shape, 0.8)).bool() & masked_indices
        mlm_inputs[indices_replaced] = self.tokenizer.mask_token_id

        return {"anchor_input": enc_anchor, "pos_input": enc_pos, "neg_input": enc_neg, "know_input": enc_know, "mlm_input_ids": mlm_inputs, "mlm_labels": mlm_labels, "relation_ids": relation_ids}


# ==========================================
# 3. 模型定义 (更新权重输入)
# ==========================================


class GeologyContrastiveModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.config = BertConfig.from_pretrained(model_name)
        self.bert_lm = BertForMaskedLM.from_pretrained(model_name)
        self.bert = self.bert_lm.bert

        hidden_size = self.config.hidden_size
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128),
        )
        self.kg_loss_fn = KGInfoNCELoss()
        self.supcon_loss_fn = SupConLoss()

    def get_embedding(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, batch, weights):
        """
        weights: dict {'mlm': float, 'kg': float, 'supcon': float}
        """
        loss_dict = {}
        total_loss = 0.0

        # 1. MLM
        if weights["mlm"] > 0:
            mlm_outputs = self.bert_lm(
                input_ids=batch["mlm_input_ids"],
                attention_mask=batch["anchor_input"]["attention_mask"],
                labels=batch["mlm_labels"],
            )
            loss_dict["mlm_loss"] = mlm_outputs.loss
            total_loss += weights["mlm"] * mlm_outputs.loss

        # Encoding
        if weights["kg"] > 0 or weights["supcon"] > 0:
            anchor_emb = self.get_embedding(batch["anchor_input"]["input_ids"], batch["anchor_input"]["attention_mask"])
            anchor_z = self.proj_head(anchor_emb)

            # 2. KG-CL
            if weights["kg"] > 0:
                pos_emb = self.get_embedding(batch["pos_input"]["input_ids"], batch["pos_input"]["attention_mask"])
                neg_emb = self.get_embedding(batch["neg_input"]["input_ids"], batch["neg_input"]["attention_mask"])
                know_emb = self.get_embedding(batch["know_input"]["input_ids"], batch["know_input"]["attention_mask"])

                pos_z = self.proj_head(pos_emb)
                neg_z = self.proj_head(neg_emb)
                know_z = self.proj_head(know_emb)

                loss_instance = self.kg_loss_fn(anchor_z, pos_z, neg_z)
                loss_knowledge = 0
                if know_z.shape[0] > 1:
                    sim_ak = torch.matmul(F.normalize(anchor_z, dim=1), F.normalize(know_z, dim=1).T) / 0.05
                    labels = torch.arange(know_z.shape[0], device=know_z.device)
                    loss_knowledge = F.cross_entropy(sim_ak, labels)

                kg_loss = loss_instance + 0.5 * loss_knowledge
                loss_dict["kg_loss"] = kg_loss
                total_loss += weights["kg"] * kg_loss

            # 3. SupCon
            if weights["supcon"] > 0:
                if anchor_z.shape[0] > 1:
                    supcon_loss = self.supcon_loss_fn(anchor_z, batch["relation_ids"])
                    loss_dict["supcon_loss"] = supcon_loss
                    total_loss += weights["supcon"] * supcon_loss

        return total_loss, loss_dict


# ==========================================
# 4. 训练主流程 (更新Loop逻辑)
# ==========================================


def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/地质知识-对比学习-指定-with-relations.jsonl")
    parser.add_argument("--model_name", type=str, default="pretrained_models/roberta-base")
    parser.add_argument("--output_dir", type=str, default="gk/gk+scl/geology_contrastive_bert")

    # Batch N*M 配置
    parser.add_argument("--n_classes", type=int, default=9, help="每个Batch包含N个类别")
    parser.add_argument("--m_samples", type=int, default=9, help="每个类别包含M个样本 (Batch Size = N*M)")

    # 权重配置
    parser.add_argument("--w_mlm", type=float, default=1.0, help="MLM损失权重")
    parser.add_argument("--w_kg", type=float, default=1.0, help="知识对比损失权重")
    parser.add_argument("--w_supcon", type=float, default=1.0, help="监督对比损失权重")

    # 训练控制
    parser.add_argument("--epochs", type=int, default=-1, help="正整数为固定轮次，-1为自动早停")
    parser.add_argument("--patience", type=int, default=3, help="早停耐心值")
    parser.add_argument("--lr", type=float, default=2e-5)

    args = parser.parse_args()

    # 权重字典
    loss_weights = {"mlm": args.w_mlm, "kg": args.w_kg, "supcon": args.w_supcon}
    print(f"Config: N={args.n_classes}, M={args.m_samples}, Weights={loss_weights}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    # 1. 数据准备
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    all_data = load_data(args.data_path)

    # 数据集分割逻辑
    if args.epochs == -1:
        random.shuffle(all_data)
        split_idx = int(len(all_data) * 0.9)
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        print(f"Auto-split for Early Stopping: Train={len(train_data)}, Val={len(val_data)}")
    else:
        train_data = all_data
        val_data = None

    train_dataset = GeologyDataset(train_data)
    collator = GeologyCollator(tokenizer)

    train_sampler = BalancedBatchSampler(train_dataset.labels, n_classes=args.n_classes, m_samples=args.m_samples)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collator, num_workers=4, pin_memory=True)

    val_loader = None
    if val_data:
        val_dataset = GeologyDataset(val_data)
        # val_loader = DataLoader(val_dataset, batch_size=args.n_classes * args.m_samples, shuffle=False, collate_fn=collator)

        # 修改后：让验证集也用 BalancedBatchSampler (仅为了观察 Loss，实际推理不需要)
        # 1. 先初始化验证集的采样器 (和训练集逻辑一样，保证 N*M 结构)
        val_sampler = BalancedBatchSampler(
            val_dataset.labels,  # 传入验证集的标签列表
            n_classes=args.n_classes,
            m_samples=args.m_samples,
        )

        # 2. 填入 DataLoader 的参数
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,  # 核心：由它控制 batch 大小和内容
            collate_fn=collator,  # 核心：处理 Tokenizer 和 Padding
            num_workers=4,  # 推荐：使用 4 个进程加载数据
            pin_memory=True,  # 推荐：加速 CPU 到 GPU 的传输
        )

    # 2. 模型与优化器
    model = GeologyContrastiveModel(model_name=args.model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    if args.epochs > 0:
        total_epochs = args.epochs
        total_steps = len(train_loader) * total_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    else:
        total_epochs = 100
        total_steps = len(train_loader) * total_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
        early_stopping = EarlyStopping(patience=args.patience, path=args.output_dir, verbose=True)

    # 3. 训练循环
    model.train()
    print("Start Training...")

    for epoch in range(total_epochs):
        # ---------------- Train ----------------
        model.train()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        # 累积器
        epoch_loss_sum = 0.0
        epoch_detail_loss_sum = defaultdict(float)  # 存储各个子loss的总和

        for batch in train_iterator:
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else {sk: sv.to(device) for sk, sv in v.items()} for k, v in batch.items()}

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_bf16):
                loss, loss_dict = model(batch_gpu, weights=loss_weights)

            loss.backward()
            optimizer.step()
            scheduler.step()

            # --- 统计逻辑 ---
            epoch_loss_sum += loss.item()
            for k, v in loss_dict.items():
                epoch_detail_loss_sum[k] += v.item()

            # --- 实时显示 (进度条) ---
            # 缩短key的显示名称以适应屏幕
            logs = {"Total": f"{loss.item():.4f}"}
            name_map = {"mlm_loss": "MLM", "kg_loss": "KG", "supcon_loss": "Sup"}
            for k, v in loss_dict.items():
                short_name = name_map.get(k, k)
                logs[short_name] = f"{v.item():.4f}"

            train_iterator.set_postfix(logs)

        # --- Epoch 结束打印平均值 ---
        avg_train_loss = epoch_loss_sum / len(train_loader)
        log_msg = f"[Train Epoch {epoch + 1}] Avg Total: {avg_train_loss:.4f}"
        for k, v in epoch_detail_loss_sum.items():
            short_name = {"mlm_loss": "MLM", "kg_loss": "KG", "supcon_loss": "Sup"}.get(k, k)
            log_msg += f" | Avg {short_name}: {v / len(train_loader):.4f}"
        print(log_msg)

        # ---------------- Control Logic ----------------
        if args.epochs > 0:
            if epoch == args.epochs - 1:
                print(f"Saving final model to {args.output_dir}")
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                model.bert_lm.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                break
        else:
            # ---------------- Validation ----------------
            model.eval()
            val_loss_sum = 0.0
            val_detail_loss_sum = defaultdict(float)

            with torch.no_grad():
                for batch in val_loader:
                    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else {sk: sv.to(device) for sk, sv in v.items()} for k, v in batch.items()}
                    with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_bf16):
                        loss, loss_dict = model(batch_gpu, weights=loss_weights)

                    val_loss_sum += loss.item()
                    for k, v in loss_dict.items():
                        val_detail_loss_sum[k] += v.item()

            avg_val_loss = val_loss_sum / len(val_loader)

            # 打印验证集详情
            val_log_msg = f"[Val Epoch {epoch + 1}] Avg Total: {avg_val_loss:.4f}"
            for k, v in val_detail_loss_sum.items():
                short_name = {"mlm_loss": "MLM", "kg_loss": "KG", "supcon_loss": "Sup"}.get(k, k)
                val_log_msg += f" | Avg {short_name}: {v / len(val_loader):.4f}"
            print(val_log_msg)

            # 早停检查
            early_stopping(avg_val_loss, model, tokenizer)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

    print("Done!")


if __name__ == "__main__":
    main()
