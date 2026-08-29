"""
地质多元空间关系抽取模型训练脚本

该脚本实现了基于PQMT方法的超关系抽取模型，用于地质多元空间关系抽取任务。

主要功能模块：
1. 数据处理模块：负责数据加载、预处理、增强和特征提取
2. 模型训练模块：实现模型训练逻辑、损失计算和参数更新
3. 模型评估模块：实现模型评估逻辑和指标计算
4. 模型保存模块：负责模型检查点保存和管理

核心组件：
- ACEDataset：超关系数据集处理类
- SupervisedContrastiveSampler：监督对比学习采样器
- BertForACEBothOneDropoutSub：主模型类
- train：模型训练函数
- evaluate：模型评估函数

使用说明：
- 从仓库根目录运行：python main.py --dataset hyperred_hyperrelation --data_dir data/hyperred --output_dir outputs
- 查看完整参数：python main.py --help

PQMT方法的主要实现：
- 数据预处理层：ACEDataset.initialize方法
- 位置感知网络层 (PosiNet)：src/model.py 中的 PosiNet 类
- 限定符交互注意力网络层 (QIA)：src/model.py 中的 qiaAttention 类
- 多任务学习层 (MTL)：src/model.py 中的 BertForACEBothOneDropoutSub.forward 方法
- 解码层：evaluate 函数中的结果处理部分
"""

import argparse
import glob
import itertools
import json
import logging
import os
import random
import re
import shutil
import time
import timeit
from collections import defaultdict

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
)
from tqdm import tqdm, trange
from transformers import (
    BertConfig,
    BertTokenizer,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from src.model import BertForACEBothOneDropoutSub

# 日志配置
logger = logging.getLogger(__name__)

# 任务标签映射
# 这些字典将在运行时根据数据集初始化
task_ner_labels = {}  # 实体标签列表
task_rel_labels = {}  # 关系标签列表
task_q_labels = {}  # 限定词标签列表


# 工具函数：有序去重
# 保持列表中元素的顺序，同时去除重复元素
def sset(arr):
    """
    有序去重函数

    Args:
        arr: 输入列表

    Returns:
        m: 去重后的有序列表
    """
    m = []
    for i in arr:
        if i not in m:
            m.append(i)
    return m


# 工具函数：设置随机种子
# 确保实验的可重复性
def set_seed(args):
    """
    设置随机种子，确保实验可重复性

    Args:
        args: 包含seed参数的配置对象
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class SupervisedContrastiveSampler(Sampler):
    """
    监督对比学习采样器，确保每个batch包含N种label，每种label有K个样本。
    通过阅读ACEDataset源码，确定样本的标签为关系标签。
    """

    def __init__(self, data_source, n_labels_per_batch, k_samples_per_label):
        """
        参数:
            data_source (Dataset): 用于采样的数据集。
            n_labels_per_batch (int): 每个批次中不同标签的数量。
            k_samples_per_label (int): 每个标签在一个批次中的样本数量。
        """
        self.data_source = data_source
        self.n_labels_per_batch = n_labels_per_batch
        self.k_samples_per_label = k_samples_per_label
        self.batch_size = self.n_labels_per_batch * self.k_samples_per_label
        self.label_to_indices = defaultdict(list)

        logger.info("正在初始化 SupervisedContrastiveSampler：正在处理数据集以将标签映射到索引...")
        # 根据ACEDataset.initialize方法，每个item在self.data中是一个以subject实体为中心的字典。
        # 'examples'键包含一个潜在关系的列表。每个example元组的第二个元素(index 1)是关系标签ID。
        # 我们使用找到的第一个正向关系标签作为该数据项的采样标签。
        for i, item in enumerate(tqdm(self.data_source.data, desc="正在为采样器映射标签", dynamic_ncols=True)):
            primary_label = 0  # 默认为 'NIL' 标签
            if "examples" in item and item["examples"]:
                for example in item["examples"]:
                    label = example[1]
                    # 标签0是'NIL'。我们寻找第一个有意义的关系标签。
                    if label > 0:
                        primary_label = label
                        break
            self.label_to_indices[primary_label].append(i)

        # 过滤掉样本数不足的标签
        self.labels_with_enough_samples = [label for label, indices in self.label_to_indices.items() if len(indices) >= self.k_samples_per_label]

        if not self.labels_with_enough_samples:
            raise ValueError(f"没有任何标签拥有至少 k_samples_per_label 个样本。k_samples_per_label={self.k_samples_per_label}")

        self.num_samples = len(self.data_source)

    def __iter__(self):
        # 为当前epoch创建一个索引副本并打乱
        epoch_indices = {label: random.sample(indices, len(indices)) for label, indices in self.label_to_indices.items()}

        # 跟踪仍有足够样本的标签
        available_labels = list(self.labels_with_enough_samples)
        random.shuffle(available_labels)

        final_indices = []

        while len(available_labels) >= self.n_labels_per_batch:
            # 为当前batch采样N个标签
            sampled_labels = random.sample(available_labels, self.n_labels_per_batch)

            # 为每个采样到的标签获取K个样本
            for label in sampled_labels:
                # 从该标签的列表末尾弹出k个样本
                for _ in range(self.k_samples_per_label):
                    final_indices.append(epoch_indices[label].pop())

            # 创建一个batch后，更新可用标签列表
            available_labels = [label for label in available_labels if len(epoch_indices[label]) >= self.k_samples_per_label]

        return iter(final_indices)

    def __len__(self):
        # 返回数据集中样本的总数
        # DataLoader将处理最后一个不完整的batch
        return self.num_samples


class ACEDataset(Dataset):
    """
    用于处理超关系抽取数据集的自定义Dataset类

    Args:
        tokenizer: 用于文本编码的预训练分词器
        args: 命令行参数，包含数据路径、模型配置等
        evaluate: 是否为评估模式
        do_test: 是否为测试模式
        max_pair_length: 最大实体对长度
    """

    def __init__(self, tokenizer, args=None, evaluate=False, do_test=False, max_pair_length=None):
        # 1. 确定数据集文件路径
        if not evaluate:
            file_path = os.path.join(args.data_dir, args.train_file)
        else:
            if do_test:
                # 如果测试文件路径包含"models"，则直接使用该路径，否则从数据目录加载
                file_path = args.test_file if args.test_file.find("models") != -1 else os.path.join(args.data_dir, args.test_file)
            else:
                # 验证集同理
                file_path = args.dev_file if args.dev_file.find("models") != -1 else os.path.join(args.data_dir, args.dev_file)

        assert os.path.isfile(file_path), f"数据集文件不存在: {file_path}"

        # 2. 初始化基本属性
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.max_pair_length = max_pair_length
        self.max_entity_length = self.max_pair_length * 2

        # 3. 初始化模式和配置
        self.evaluate = evaluate
        self.use_typemarker = args.use_typemarker
        self.cuda_device = args.cuda_device
        self.args = args
        self.no_sym = args.no_sym

        # 4. 初始化标签列表
        # 实体标签列表，添加"NIL"作为默认标签
        self.ner_label_list = ["NIL"] + task_ner_labels[self.args.dataset]
        # 对称标签列表，目前仅包含"NIL"
        self.sym_labels = ["NIL"]
        # 关系标签列表，包含普通标签和逆关系标签
        relation_labels = list(sset(task_rel_labels[self.args.dataset] + task_q_labels[self.args.dataset]))
        self.label_list = ["NIL"] + relation_labels + [x + "-1" for x in relation_labels]
        # 限定符标签列表，结构与关系标签列表相同
        self.q_label_list = ["NIL"] + relation_labels + [x + "-1" for x in relation_labels]

        # 5. 初始化预测结果存储
        self.global_predicted_ners = {}
        # 6. 加载并处理数据集
        self.initialize()

    def process_to_hyperrelation(self, data):
        """
        过滤超关系数据集，保留包含限定词或普通三元组的样本

        Args:
            data: 原始数据集，包含sentences、ner和relations字段

        Returns:
            hr_data: 处理后的超关系数据集
        """
        hr_data = {"sentences": [], "ner": [], "relations": []}

        # 遍历每个句子的关系列表
        for i, sen_rels in enumerate(data["relations"]):
            hr_sen_rels = []
            for rel in sen_rels:
                # 检查关系元组的第6个元素（索引5），即限定成分列表（Qualifiers）
                # len(rel[5]) >= 0 意味着：即使限定列表为空（普通三元组），也会被保留
                # 如果只想训练包含限定词的"真"超关系，可以改为 len(rel[5]) >= 1
                if len(rel[5]) >= 0:
                    hr_sen_rels.append(rel)

            # 只有当句子包含至少一个超关系时，才保留该句子
            if len(hr_sen_rels) != 0:
                hr_data["sentences"].append(data["sentences"][i])
                hr_data["ner"].append(data["ner"][i])
                hr_data["relations"].append(hr_sen_rels)

        return hr_data

    def initialize(self):
        """
        初始化数据集，加载并处理数据

        主要步骤：
        1. 加载原始数据
        2. 处理为超关系数据
        3. 进行文本编码和特征提取
        4. 构建训练样本
        """
        tokenizer = self.tokenizer
        max_num_subwords = self.max_seq_length - 4  # 预留4个位置给特殊标记

        # 创建标签映射字典
        label_map = {label: i for i, label in enumerate(self.label_list)}  # 关系标签映射
        ner_label_map = {label: i for i, label in enumerate(self.ner_label_list)}  # 实体标签映射
        q_label_map = {label: i for i, label in enumerate(self.q_label_list)}  # 限定词标签映射

        def tokenize_word(text):
            """对单个单词进行分词"""
            if isinstance(tokenizer, RobertaTokenizer) and (text[0] != "'") and (len(text) != 1 or not self.is_punctuation(text)):
                return tokenizer.tokenize(text, add_prefix_space=True)
            return tokenizer.tokenize(text)

        # 初始化统计变量
        self.ner_tot_recall = 0  # 实体总数
        self.tot_recall = 0  # 关系总数
        self.q_tot_recall = 0  # 限定词关系总数
        self.data = []  # 存储处理后的训练样本

        # 存储真实标签，用于评估
        self.ner_golden_labels = set()  # 实体真实标签
        self.golden_labels = set()  # 关系真实标签
        self.golden_labels_withner = set()  # 带实体类型的关系真实标签
        self.q_golden_labels = set()  # 限定词关系真实标签
        self.q_golden_labels_withner = set()  # 带实体类型的限定词关系真实标签

        maxR = 0  # 记录最大关系数
        q_maxR = 0  # 记录最大限定词关系数
        maxL = 0  # 记录最大序列长度

        # 读取数据文件
        with open(self.file_path, "r", encoding="utf-8") as f:
            for l_idx, line in tqdm(enumerate(f), desc="处理数据", dynamic_ncols=True):
                # 小数据集模式下，只处理前100条数据
                if self.args.smallerdataset and l_idx > 100:
                    break

                # 解析JSON数据
                data = json.loads(line)
                # 处理为超关系数据
                data = self.process_to_hyperrelation(data)
                # 跳过没有关系的样本
                if len(data["relations"]) == 0:
                    continue

                # 测试模式下，限制数据量
                if self.args.output_dir.find("test") != -1 and len(self.data) > 100:
                    break

                # 提取句子、实体和关系数据
                sentences = data["sentences"]
                ners = data["ner"]
                std_ners = data["ner"]  # 标准实体标签
                relations = data["relations"]

                # 统计关系数量
                for sentence_relation in relations:
                    for x in sentence_relation:
                        self.tot_recall += 1
                        for q in x[5]:
                            self.q_tot_recall += 1

                # 计算句子边界
                sentence_boundaries = [0]
                words = []
                L = 0
                for i in range(len(sentences)):
                    L += len(sentences[i])
                    sentence_boundaries.append(L)
                    words += sentences[i]

                # 分词和子词映射
                tokens = [tokenize_word(w) for w in words]
                subwords = [w for li in tokens for w in li]
                maxL = max(maxL, len(subwords))
                # 构建单词到子词的映射，用于后续实体位置转换
                token2subword = [0] + list(itertools.accumulate(len(li) for li in tokens))
                # 计算子词级别的句子边界
                subword_sentence_boundaries = [sum(len(li) for li in tokens[:p]) for p in sentence_boundaries]

                # 处理每个句子
                for n in range(len(subword_sentence_boundaries) - 1):
                    sentence_ners = ners[n]
                    sentence_relations = relations[n]
                    std_ner = std_ners[n]

                    # 构建标准实体标签字典
                    std_entity_labels = {}
                    self.ner_tot_recall += len(std_ner)

                    # 存储实体真实标签
                    for start, end, label in std_ner:
                        std_entity_labels[(start, end)] = label
                        self.ner_golden_labels.add(((l_idx, n), (start, end), label))

                    # 存储全局预测实体
                    self.global_predicted_ners[(l_idx, n)] = list(sentence_ners)

                    # 计算句子在文档中的位置
                    doc_sent_start, doc_sent_end = subword_sentence_boundaries[n : n + 2]

                    # 计算上下文长度
                    left_length = doc_sent_start  # 句子左侧上下文长度
                    right_length = len(subwords) - doc_sent_end  # 句子右侧上下文长度
                    sentence_length = doc_sent_end - doc_sent_start  # 句子本身长度
                    half_context_length = int((max_num_subwords - sentence_length) / 2)

                    # 分配左右上下文长度
                    if sentence_length < max_num_subwords:
                        if left_length < right_length:
                            # 左侧上下文较短，优先分配左侧
                            left_context_length = min(left_length, half_context_length)
                            right_context_length = min(
                                right_length,
                                max_num_subwords - left_context_length - sentence_length,
                            )
                        else:
                            # 右侧上下文较短，优先分配右侧
                            right_context_length = min(right_length, half_context_length)
                            left_context_length = min(
                                left_length,
                                max_num_subwords - right_context_length - sentence_length,
                            )
                    else:
                        # 句子本身长度超过限制，不添加上下文
                        left_context_length = 0
                        right_context_length = 0

                    # 计算文档偏移量
                    doc_offset = doc_sent_start - left_context_length
                    # 提取目标子词序列
                    target_tokens = subwords[doc_offset : doc_sent_end + right_context_length]
                    # 添加特殊标记 [CLS] 和 [SEP]
                    target_tokens = [tokenizer.cls_token] + target_tokens[: self.max_seq_length - 4] + [tokenizer.sep_token]
                    # 验证序列长度
                    assert len(target_tokens) <= self.max_seq_length - 2

                pos2label = {}
                q_pos2label = {}

                for x in sentence_relations:
                    pos2label[(x[0], x[1], x[2], x[3])] = label_map[x[4]]
                    self.golden_labels.add(((l_idx, n), (x[0], x[1]), (x[2], x[3]), x[4]))
                    self.golden_labels_withner.add(
                        (
                            (l_idx, n),
                            (x[0], x[1], std_entity_labels[(x[0], x[1])]),
                            (x[2], x[3], std_entity_labels[(x[2], x[3])]),
                            x[4],
                        )
                    )
                    pos2label[(x[2], x[3], x[1], x[0])] = label_map[x[4] + "-1"]
                    self.golden_labels.add(((l_idx, n), (x[2], x[3]), (x[0], x[1]), x[4] + "-1"))
                    self.golden_labels_withner.add(
                        (
                            (l_idx, n),
                            (x[2], x[3], std_entity_labels[(x[2], x[3])]),
                            (x[0], x[1], std_entity_labels[(x[0], x[1])]),
                            x[4] + "-1",
                        )
                    )
                    for q in x[5]:
                        q_pos2label[(x[0], x[1], x[2], x[3], q[0], q[1])] = (
                            label_map[x[4]],
                            q_label_map[q[2]],
                        )
                        self.q_golden_labels.add(
                            (
                                (l_idx, n),
                                (x[0], x[1]),
                                (x[2], x[3]),
                                x[4],
                                (q[0], q[1]),
                                q[2],
                            )
                        )
                        self.q_golden_labels_withner.add(
                            (
                                (l_idx, n),
                                (x[0], x[1], std_entity_labels[(x[0], x[1])]),
                                (x[2], x[3], std_entity_labels[(x[2], x[3])]),
                                x[4],
                                (q[0], q[1], std_entity_labels[(q[0], q[1])]),
                                q[2],
                            )
                        )

                        q_pos2label[(x[2], x[3], x[0], x[1], q[0], q[1])] = (
                            label_map[x[4] + "-1"],
                            q_label_map[q[2]],
                        )
                        self.q_golden_labels.add(
                            (
                                (l_idx, n),
                                (x[2], x[3]),
                                (x[0], x[1]),
                                x[4] + "-1",
                                (q[0], q[1]),
                                q[2],
                            )
                        )
                        self.q_golden_labels_withner.add(
                            (
                                (l_idx, n),
                                (x[2], x[3], std_entity_labels[(x[2], x[3])]),
                                (x[0], x[1], std_entity_labels[(x[0], x[1])]),
                                x[4] + "-1",
                                (q[0], q[1], std_entity_labels[(q[0], q[1])]),
                                q[2],
                            )
                        )

                        q_pos2label[(x[0], x[1], q[0], q[1], x[2], x[3])] = (
                            q_label_map[q[2]],
                            label_map[x[4]],
                        )
                        self.q_golden_labels.add(
                            (
                                (l_idx, n),
                                (x[0], x[1]),
                                (q[0], q[1]),
                                q[2],
                                (x[2], x[3]),
                                x[4],
                            )
                        )
                        self.q_golden_labels_withner.add(
                            (
                                (l_idx, n),
                                (x[0], x[1], std_entity_labels[(x[0], x[1])]),
                                (q[0], q[1], std_entity_labels[(q[0], q[1])]),
                                q[2],
                                (x[2], x[3], std_entity_labels[(x[2], x[3])]),
                                x[4],
                            )
                        )

                        q_pos2label[(x[2], x[3], q[0], q[1], x[0], x[1])] = (
                            q_label_map[q[2]],
                            label_map[x[4] + "-1"],
                        )
                        self.q_golden_labels.add(
                            (
                                (l_idx, n),
                                (x[2], x[3]),
                                (q[0], q[1]),
                                q[2],
                                (x[0], x[1]),
                                x[4] + "-1",
                            )
                        )
                        self.q_golden_labels_withner.add(
                            (
                                (l_idx, n),
                                (x[2], x[3], std_entity_labels[(x[2], x[3])]),
                                (q[0], q[1], std_entity_labels[(q[0], q[1])]),
                                q[2],
                                (x[0], x[1], std_entity_labels[(x[0], x[1])]),
                                x[4] + "-1",
                            )
                        )

                        q_pos2label[(q[0], q[1], x[0], x[1], x[2], x[3])] = (
                            q_label_map[q[2] + "-1"],
                            label_map[x[4]],
                        )
                        self.q_golden_labels.add(
                            (
                                (l_idx, n),
                                (q[0], q[1]),
                                (x[0], x[1]),
                                q[2] + "-1",
                                (x[2], x[3]),
                                x[4],
                            )
                        )
                        self.q_golden_labels_withner.add(
                            (
                                (l_idx, n),
                                (q[0], q[1], std_entity_labels[(q[0], q[1])]),
                                (x[0], x[1], std_entity_labels[(x[0], x[1])]),
                                q[2] + "-1",
                                (x[2], x[3], std_entity_labels[(x[2], x[3])]),
                                x[4],
                            )
                        )

                        q_pos2label[(q[0], q[1], x[2], x[3], x[0], x[1])] = (
                            label_map[x[4]],
                            q_label_map[q[2] + "-1"],
                        )
                        self.q_golden_labels.add(
                            (
                                (l_idx, n),
                                (q[0], q[1]),
                                (x[2], x[3]),
                                x[4],
                                (x[0], x[1]),
                                q[2] + "-1",
                            )
                        )
                        self.q_golden_labels_withner.add(
                            (
                                (l_idx, n),
                                (q[0], q[1], std_entity_labels[(q[0], q[1])]),
                                (x[2], x[3], std_entity_labels[(x[2], x[3])]),
                                x[4],
                                (x[0], x[1], std_entity_labels[(x[0], x[1])]),
                                q[2] + "-1",
                            )
                        )

                entities = list(sentence_ners)

                for sub in entities:
                    cur_ins = []
                    q_cur_ins = []

                    if sub[0] < 10000:
                        sub_s = token2subword[sub[0]] - doc_offset + 1
                        sub_e = token2subword[sub[1] + 1] - doc_offset
                        sub_label = ner_label_map[sub[2]]

                        if self.use_typemarker:
                            l_m = "[unused%d]" % (2 + sub_label)
                            r_m = "[unused%d]" % (2 + sub_label + len(self.ner_label_list))
                        else:
                            l_m = "[unused0]"
                            r_m = "[unused1]"

                        sub_tokens = target_tokens[:sub_s] + [l_m] + target_tokens[sub_s : sub_e + 1] + [r_m] + target_tokens[sub_e + 1 :]
                        sub_e += 2
                    else:
                        sub_s = len(target_tokens)
                        sub_e = len(target_tokens) + 1
                        sub_tokens = target_tokens + ["[unused0]", "[unused1]"]
                        sub_label = -1

                    if sub_e >= self.max_seq_length - 1:
                        continue

                    for start, end, obj_label in sentence_ners:
                        doc_entity_start = token2subword[start]
                        doc_entity_end = token2subword[end + 1]
                        left = doc_entity_start - doc_offset + 1
                        right = doc_entity_end - doc_offset

                        obj = (start, end)
                        if obj[0] >= sub[0]:
                            left += 1
                            if obj[0] > sub[1]:
                                left += 1

                        if obj[1] >= sub[0]:
                            right += 1
                            if obj[1] > sub[1]:
                                right += 1

                        label = pos2label.get((sub[0], sub[1], obj[0], obj[1]), 0)

                        if right >= self.max_seq_length - 1:
                            continue

                        cur_ins.append(((left, right, ner_label_map[obj_label]), label, obj))

                        for q_start, q_end, qul_label in sentence_ners:
                            q_doc_entity_start = token2subword[q_start]
                            q_doc_entity_end = token2subword[q_end + 1]
                            q_left = q_doc_entity_start - doc_offset + 1
                            q_right = q_doc_entity_end - doc_offset

                            q = (q_start, q_end)
                            if q[0] >= sub[0]:
                                q_left += 1
                                if q[0] > sub[1]:
                                    q_left += 1

                            if q[1] >= sub[0]:
                                q_right += 1
                                if q[1] > sub[1]:
                                    q_right += 1

                            if q_right >= self.max_seq_length - 1:
                                continue

                            label = q_pos2label.get((sub[0], sub[1], obj[0], obj[1], q[0], q[1]), (0, 0))
                            q_cur_ins.append(
                                (
                                    (left, right, ner_label_map[obj_label]),
                                    label[0],
                                    obj,
                                    (q_left, q_right, ner_label_map[qul_label]),
                                    label[1],
                                    q,
                                )
                            )

                    maxR = max(maxR, len(cur_ins))
                    q_maxR = max(q_maxR, len(q_cur_ins))
                    q_dL = self.max_pair_length * self.max_pair_length
                    if self.args.shuffle:
                        np.random.shuffle(cur_ins)
                        np.random.shuffle(q_cur_ins)

                    for i in range(0, len(q_cur_ins), q_dL):
                        q_examples = q_cur_ins[i : i + q_dL]
                        item = {
                            "index": (l_idx, n),
                            "sentence": sub_tokens,
                            "examples": q_examples,
                            "sub": (sub, (sub_s, sub_e), sub_label),  # 头实体（Subject）: (原始数据中的头实体信息, 处理后的头实体信息（加了[unused0]和[unused1]）, 实体类型)
                        }
                        # example:
                        # (
                        # (Obj_Start, Obj_End, Obj_Type),  # 0. 候选 Object 在 sentence 中的位置和类型
                        # Main_Rel_Label,                  # 1. Subject 和 Object 的主关系标签 (GT)
                        # (Obj_Orig_S, Obj_Orig_E),        # 2. Object 在原始文本中的单词级索引
                        # (Qual_Start, Qual_End, Qual_Type), # 3. 候选 Qualifier 在 sentence 中的位置和类型
                        # Qual_Rel_Label,                  # 4. Qualifier 的限定关系标签 (GT)
                        # (Qual_Orig_S, Qual_Orig_E)       # 5. Qualifier 在原始文本中的单词级索引
                        # )

                        self.data.append(item)

        logger.info("数据集中最长的一句话的长度: %s", maxR)
        logger.info("对于同一个头实体（Subject），数据集中最多有多少个候选二元关系: %s", q_maxR)
        logger.info("对于同一个头实体，数据集中最多有多少个候选超关系组合: %s", maxL)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        sub, sub_position, sub_label = entry["sub"]
        input_ids = self.tokenizer.convert_tokens_to_ids(entry["sentence"])

        L = len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))

        attention_mask = torch.zeros(
            (
                self.max_entity_length + self.max_seq_length,
                self.max_entity_length + self.max_seq_length,
            ),
            dtype=torch.int64,
        )
        attention_mask[:L, :L] = 1

        input_ids = input_ids + [3] * (int(np.sqrt(len(entry["examples"])))) + [self.tokenizer.pad_token_id] * (self.max_pair_length - int(np.sqrt(len(entry["examples"]))))
        input_ids = input_ids + [4] * (int(np.sqrt(len(entry["examples"])))) + [self.tokenizer.pad_token_id] * (self.max_pair_length - int(np.sqrt(len(entry["examples"]))))

        labels = []
        ner_labels = []
        q_labels = []
        q_ner_labels = []

        mention_pos = []
        mention_2 = []
        q_mention_pos = []
        q_mention_2 = []

        position_ids = list(range(self.max_seq_length)) + [0] * self.max_entity_length
        num_pair = self.max_pair_length

        sub_index = -1
        for x_idx, obj in enumerate(entry["examples"]):
            q_m2 = obj[3]
            if sub_position[0] + 1 == q_m2[0] and sub_position[1] - 1 == q_m2[1]:
                sub_index = x_idx % int(np.sqrt(len(entry["examples"])))
                break

        for x_idx, obj in enumerate(entry["examples"]):
            m2 = obj[0]
            label = obj[1]

            if x_idx % np.sqrt(len(entry["examples"])) == 0:
                w1 = int(x_idx / np.sqrt(len(entry["examples"])))
                w2 = w1 + num_pair

                w1 += self.max_seq_length
                w2 += self.max_seq_length

                position_ids[w1] = m2[0]
                position_ids[w2] = m2[1]

                for xx in [w1, w2]:
                    for yy in [w1, w2]:
                        attention_mask[xx, yy] = 1
                    attention_mask[xx, :L] = 1

            q_m2 = obj[3]
            q_label = obj[4]

            if x_idx % np.sqrt(len(entry["examples"])) == 0:
                temp_mention_pos = []
                temp_mention_2 = []
                temp_labels = []
                temp_ner_labels = []

                temp_q_mention_pos = []
                temp_q_mention_2 = []
                temp_q_labels = []
                temp_q_ner_labels = []

            temp_mention_pos.append((m2[0], m2[1]))
            temp_mention_2.append(obj[2])

            temp_q_mention_pos.append((q_m2[0], q_m2[1]))
            temp_q_mention_2.append(obj[5])

            if (
                x_idx % np.sqrt(len(entry["examples"])) == int(x_idx / np.sqrt(len(entry["examples"])))
                or x_idx % np.sqrt(len(entry["examples"])) == sub_index
                or int(x_idx / np.sqrt(len(entry["examples"]))) == sub_index
                or sub_index == -1
            ):
                temp_labels.append(-1)
            else:
                temp_labels.append(label)
            temp_ner_labels.append(m2[2])

            if (
                x_idx % np.sqrt(len(entry["examples"])) == int(x_idx / np.sqrt(len(entry["examples"])))
                or x_idx % np.sqrt(len(entry["examples"])) == sub_index
                or int(x_idx / np.sqrt(len(entry["examples"]))) == sub_index
                or sub_index == -1
            ):
                temp_q_labels.append(-1)
            else:
                temp_q_labels.append(q_label)
            temp_q_ner_labels.append(q_m2[2])

            if (x_idx + 1) % np.sqrt(len(entry["examples"])) == 0:
                temp_mention_pos += [(0, 0)] * (num_pair - len(temp_mention_pos))
                temp_labels += [-1] * (num_pair - len(temp_labels))
                temp_ner_labels += [-1] * (num_pair - len(temp_ner_labels))

                temp_q_mention_pos += [(0, 0)] * (num_pair - len(temp_q_mention_pos))
                temp_q_labels += [-1] * (num_pair - len(temp_q_labels))
                temp_q_ner_labels += [-1] * (num_pair - len(temp_q_ner_labels))

                mention_pos.append(temp_mention_pos)
                mention_2.append(temp_mention_2)
                labels.append(temp_labels)
                ner_labels.append(temp_ner_labels)

                q_mention_pos.append(temp_q_mention_pos)
                q_mention_2.append(temp_q_mention_2)
                q_labels.append(temp_q_labels)
                q_ner_labels.append(temp_q_ner_labels)

        mention_pos += [[(0, 0)] * num_pair] * (num_pair - len(mention_pos))
        q_mention_pos += [[(0, 0)] * num_pair] * (num_pair - len(q_mention_pos))
        labels += [[-1] * num_pair] * (num_pair - len(labels))
        q_labels += [[-1] * num_pair] * (num_pair - len(q_labels))
        ner_labels = ner_labels[0]
        q_ner_labels += [[-1] * num_pair] * (num_pair - len(q_ner_labels))

        item = [
            torch.tensor(input_ids),
            attention_mask,
            torch.tensor(position_ids),
            torch.tensor(sub_position),
            torch.tensor(mention_pos),
            torch.tensor(labels, dtype=torch.int64),
            torch.tensor(ner_labels, dtype=torch.int64),
            torch.tensor(sub_label, dtype=torch.int64),
            torch.tensor(q_mention_pos),
            torch.tensor(q_labels, dtype=torch.int64),
            torch.tensor(q_ner_labels, dtype=torch.int64),
        ]

        if self.evaluate:
            item.append(entry["index"])
            item.append(sub)
            item.append(mention_2)
            item.append(q_mention_2)

        return item

    @staticmethod
    def collate_fn(batch):
        fields = [x for x in zip(*batch)]
        num_metadata_fields = 4
        stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]
        stacked_fields.extend(fields[-num_metadata_fields:])
        return stacked_fields


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def train(args, model, tokenizer):
    """
    模型训练函数

    Args:
        args: 训练参数
        model: 要训练的模型
        tokenizer: 用于文本编码的分词器
    """
    # 1. 模型参数统计
    logger.info("\n===== 模型参数统计 =====")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 模型总可训练参数数量
    dtype_size = 2 if args.fp16 else 4  # 根据训练精度调整数据类型大小
    param_size_mb = total_params * dtype_size / (1024**2)  # 模型参数占用的存储空间（MB）

    logger.info(f"总可训练参数: {total_params:,}")
    logger.info(f"存储空间: {param_size_mb:.2f} MB (数据类型: {'fp16' if args.fp16 else 'fp32'})")
    logger.info("==========================\n")

    # 3. 初始化训练数据集
    train_dataset = ACEDataset(tokenizer=tokenizer, args=args, max_pair_length=args.max_pair_length)

    # 4. 调整训练批次大小并选择采样器
    if args.use_supervised_contrastive:
        # 监督对比学习模式：批次大小 = 标签数量 × 每个标签的样本数
        args.train_batch_size = args.n_labels_per_batch * args.k_samples_per_label
        # 使用监督对比学习采样器，确保每个批次包含指定数量的标签和样本
        train_sampler = SupervisedContrastiveSampler(train_dataset, args.n_labels_per_batch, args.k_samples_per_label)
        logger.info(f"使用监督对比学习采样器，每个batch包含{args.n_labels_per_batch}种label，每种label{args.k_samples_per_label}个样本")
    else:
        # 普通训练模式：批次大小 = 命令行参数中设置的批次大小
        args.train_batch_size = args.train_batch_size
        # 使用随机采样器
        train_sampler = RandomSampler(train_dataset)

    # 5. 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=4 * int(args.output_dir.find("test") == -1),  # 测试模式下减少worker数量，加快启动速度
    )

    # 6. 计算总训练步数
    if args.max_steps > 0:
        # 如果指定了最大步数，优先使用最大步数
        t_total = args.max_steps
        # 根据最大步数和梯度累积步数计算需要的训练轮数
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        # 否则根据训练轮数、数据加载器长度和梯度累积步数计算总步数
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # 7. 准备优化器和学习率调度器
    # 参数分组：区分需要权重衰减和不需要权重衰减的参数
    # - 不需要权重衰减的参数：bias和LayerNorm.weight
    # - 需要权重衰减的参数：其他所有参数
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # 初始化AdamW优化器，设置学习率和epsilon
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # 初始化学习率调度器（线性warmup和decay）
    if args.warmup_steps == -1:
        # 如果未指定warmup步数，使用总步数的10%作为warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * t_total), num_training_steps=t_total)
    else:
        # 否则使用指定的warmup步数
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # 9. 开始训练
    logger.info("***** 开始训练 *****")
    logger.info("  训练样本数 = %d", len(train_dataset))
    logger.info("  训练轮数 = %d", args.num_train_epochs)
    logger.info("  训练批次大小 = %d", args.train_batch_size)
    logger.info(
        "  总训练批次大小 (含并行、分布式和梯度累积) = %d",
        args.train_batch_size * args.gradient_accumulation_steps,
    )
    logger.info("  梯度累积步数 = %d", args.gradient_accumulation_steps)
    logger.info("  总优化步数 = %d", t_total)

    # 10. 初始化训练参数和损失记录
    global_step = 0  # 全局训练步数
    # 总损失和各部分损失
    tr_loss, logging_loss = 0.0, 0.0  # 总损失
    tr_ner_loss, logging_ner_loss = 0.0, 0.0  # 实体识别损失
    tr_re_loss, logging_re_loss = 0.0, 0.0  # 关系抽取损失
    tr_q_re_loss, logging_q_re_loss = 0.0, 0.0  # 限定词关系损失
    tr_scl_loss, logging_scl_loss = 0.0, 0.0  # 监督对比学习损失
    tr_gkc_loss, logging_gkc_loss = 0.0, 0.0  # 地质知识约束损失

    # 初始化最佳F1分数
    best_f1 = -1
    # 移除旧的实验数据文件
    if os.path.exists(os.path.join(args.output_dir, "experimental_data.json")):
        os.remove(os.path.join(args.output_dir, "experimental_data.json"))

    # 11. 开始训练轮次迭代
    model.zero_grad()  # 清空梯度
    set_seed(args)  # 设置随机种子
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=not len(args.cuda_device) > 0)
    epoch = 0

    # 12. 训练轮次迭代
    for _ in train_iterator:
        epoch += 1

        # 13. 如果启用了shuffle且不是第一个epoch，重新初始化数据集以打乱数据
        if args.shuffle and _ > 0:
            train_dataset.initialize()

        # 14. 创建epoch迭代器
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not len(args.cuda_device) > 0, dynamic_ncols=True)

        # 15. 批次迭代
        for step, batch in enumerate(epoch_iterator):
            model.train()  # 设置模型为训练模式

            # 16. 将batch数据移至指定设备
            batch = tuple(t.to(args.device) for t in batch)

            # 17. 自动混合精度训练
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 18. 构建模型输入
                inputs = {
                    "input_ids": batch[0],  # 输入token IDs
                    "attention_mask": batch[1],  # 注意力掩码
                    "position_ids": batch[2],  # 位置 IDs
                    "sub_positions": batch[3],  # 主语位置
                    "mention_pos": batch[4],  # 提及位置
                    "labels": batch[5],  # 关系标签
                    "ner_labels": batch[6],  # 实体标签
                    "q_labels": batch[9],  # 限定词标签
                    "q_ner_labels": batch[10],  # 限定词实体标签
                }

                # 19. 模型前向传播
                outputs = model(**inputs)
                # 20. 提取损失
                loss = outputs[0]  # 总损失
                re_loss = outputs[1]  # 关系抽取损失
                ner_loss = outputs[2]  # 实体识别损失
                q_re_loss = outputs[3]  # 限定词关系损失
                scl_loss = outputs[4] if len(outputs) > 4 else None  # 监督对比学习损失
                gkc_loss = outputs[5] if len(outputs) > 5 else None  # 地质知识约束损失

            # 22. 梯度累积时，调整损失
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                re_loss = re_loss / args.gradient_accumulation_steps
                ner_loss = ner_loss / args.gradient_accumulation_steps
                q_re_loss = q_re_loss / args.gradient_accumulation_steps
                if scl_loss is not None:
                    scl_loss = scl_loss / args.gradient_accumulation_steps
                if gkc_loss is not None:
                    gkc_loss = gkc_loss / args.gradient_accumulation_steps

            # 23. 损失反向传播，计算梯度
            loss.backward()

            # 24. 累加损失值
            tr_loss += loss.item()  # 累加总损失
            if re_loss > 0:
                tr_re_loss += re_loss.item()  # 累加关系抽取损失
            if ner_loss > 0:
                tr_ner_loss += ner_loss.item()  # 累加实体识别损失
            if q_re_loss > 0:
                tr_q_re_loss += q_re_loss.item()  # 累加限定词关系损失
            if scl_loss is not None and scl_loss > 0:
                # 累加监督对比学习损失，处理可能的非张量类型
                tr_scl_loss = getattr(tr_scl_loss, "item", lambda: tr_scl_loss)() if hasattr(tr_scl_loss, "item") else tr_scl_loss
                tr_scl_loss += scl_loss.item() if hasattr(scl_loss, "item") else scl_loss
            # if gkc_loss is not None and gkc_loss > 0:
            #     tr_gkc_loss = getattr(tr_gkc_loss, "item", lambda: tr_gkc_loss)() if hasattr(tr_gkc_loss, "item") else tr_gkc_loss
            #     tr_gkc_loss += gkc_loss.item() if hasattr(gkc_loss, "item") else gkc_loss
            # 25. 更新进度条显示，实时展示训练状态
            postfix_dict = {
                "loss": "{:.4f}".format(loss.item()),  # 总损失
                "re": "{:.4f}".format(re_loss.item() if hasattr(re_loss, "item") else re_loss),  # 关系抽取损失
                "q_re": "{:.4f}".format(q_re_loss.item() if hasattr(q_re_loss, "item") else q_re_loss),  # 限定词关系损失
                "scl": "{:.4f}".format(scl_loss.item() if hasattr(scl_loss, "item") else scl_loss),  # 监督对比学习损失
                "lr": "{:.1e}".format(scheduler.get_lr()[0]),  # 当前学习率
            }
            epoch_iterator.set_postfix(postfix_dict)

            # 26. 梯度累积结束，更新模型参数
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # 27. 梯度裁剪，防止梯度爆炸
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # 28. 优化器更新参数
                optimizer.step()
                # 29. 学习率调度器更新学习率
                scheduler.step()
                # 30. 清空梯度，准备下一次迭代
                model.zero_grad()
                # 31. 更新全局训练步数
                global_step += 1

                # 32. 定期记录训练日志
                if len(args.cuda_device) > 0 and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # 计算平均损失
                    logs = {
                        "lr": scheduler.get_lr()[0],  # 当前学习率
                        "loss": (tr_loss - logging_loss) / args.logging_steps,  # 平均总损失
                        "RE_loss": (tr_re_loss - logging_re_loss) / args.logging_steps,  # 平均关系抽取损失
                        "NER_loss": (tr_ner_loss - logging_ner_loss) / args.logging_steps,  # 平均实体识别损失
                        "q_RE_loss": (tr_q_re_loss - logging_q_re_loss) / args.logging_steps,  # 平均限定词关系损失
                        "SCL_loss": (tr_scl_loss - logging_scl_loss) / args.logging_steps,  # 平均监督对比学习损失
                    }

                    # 更新日志记录变量，用于下一次计算平均损失
                    logging_loss = tr_loss
                    logging_re_loss = tr_re_loss
                    logging_ner_loss = tr_ner_loss
                    logging_q_re_loss = tr_q_re_loss
                    logging_scl_loss = tr_scl_loss if scl_loss is not None else 0

                    # 如果有日志记录回调，则调用它
                    if hasattr(args, "logging_callback") and args.logging_callback:
                        args.logging_callback.on_log(args, None, None, logs=logs)

                # 33. 定期保存模型检查点
                if len(args.cuda_device) > 0 and args.save_steps > 0 and global_step % args.save_steps == 0:
                    update = True  # 默认更新模型

                    # 34. 如果启用了训练期间评估，则进行评估
                    if args.evaluate_during_training:
                        # 调用评估函数
                        results = evaluate(args, model, tokenizer)
                        f1 = results["q_f1"]  # 限定词关系F1分数

                        # 记录评估日志
                        eval_logs = {"q_f1": f1, "num_hrf_pred": results["num_q_pred"]}
                        if hasattr(args, "logging_callback") and args.logging_callback:
                            args.logging_callback.on_log(args, None, None, logs=eval_logs)

                        # 保存实验数据
                        step_information = {"epoch": epoch, "global_step": global_step}
                        results.update(step_information)
                        with open(os.path.join(args.output_dir, "experimental_data.json"), "a") as f:
                            f.write(json.dumps(results, ensure_ascii=False) + "\n")

                        # 根据F1分数决定是否更新模型
                        if f1 > best_f1:
                            best_f1 = f1
                            logger.info("Best F1 %s", best_f1)
                        else:
                            update = False

                    # 35. 如果需要更新模型，则保存检查点
                    if update:
                        checkpoint_prefix = "checkpoint"
                        # 创建输出目录
                        output_dir = os.path.join(
                            args.output_dir,
                            f"{checkpoint_prefix}-{global_step}",
                        )
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        model.save_pretrained(output_dir)

                        # 保存训练参数
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("正在将模型检查点保存到 %s", output_dir)

                        _rotate_checkpoints(args, checkpoint_prefix)

                        ###############################################################
                        if args.test_when_update:
                            # 评估
                            results = {"dev_best_f1": best_f1}
                            if args.do_eval and len(args.cuda_device) > 0:
                                checkpoints = [args.output_dir]

                                WEIGHTS_NAME = "model.safetensors"

                                if args.eval_all_checkpoints:
                                    checkpoints = list(
                                        os.path.dirname(c)
                                        for c in sorted(
                                            glob.glob(
                                                args.output_dir + "/**/" + WEIGHTS_NAME,
                                                recursive=True,
                                            )
                                        )
                                    )

                                logger.info(
                                    "评估以下检查点：%s",
                                    checkpoints,
                                )
                                for checkpoint in checkpoints:
                                    global_step_str = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                                    result = evaluate(
                                        args,
                                        model,
                                        tokenizer,
                                        prefix=global_step_str,
                                        do_test=not args.no_test,
                                    )
                                    result = dict((k + "_{}".format(global_step_str), v) for k, v in result.items())
                                    results.update(result)
                                logger.info("%s", results)

                                if args.no_test:  # 选择开发集上的最佳结果
                                    bestv = 0
                                    for k, v in results.items():
                                        if v > bestv:
                                            bestk = k
                                    logger.info("%s", bestk)

                                output_eval_file = os.path.join(args.output_dir, "results.json")
                                json.dump(results, open(output_eval_file, "w"), ensure_ascii=False)

            # 记录每个epoch的平均损失等信息
            if step == len(train_dataloader) - 1:
                epoch_loss = tr_loss / (step + 1)
                epoch_re_loss = tr_re_loss / (step + 1) if tr_re_loss > 0 else 0
                epoch_q_re_loss = tr_q_re_loss / (step + 1) if tr_q_re_loss > 0 else 0
                epoch_scl_loss = tr_scl_loss / (step + 1) if tr_scl_loss > 0 else 0

                logger.info("Epoch: %d | Average loss: %.4f | RE loss: %.4f | Q-RE loss: %.4f | SCL loss: %.4f", epoch, epoch_loss, epoch_re_loss, epoch_q_re_loss, epoch_scl_loss)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step, best_f1


def evaluate(args, model, tokenizer, prefix="", do_test=False):
    """
    模型评估函数

    Args:
        args: 评估参数
        model: 要评估的模型
        tokenizer: 用于文本编码的分词器
        prefix: 评估前缀，用于日志记录
        do_test: 是否为测试模式

    Returns:
        results: 评估结果字典，包含各种评估指标
    """
    # 1. 初始化评估输出目录
    eval_output_dir = args.output_dir

    # 2. 初始化评估数据集
    eval_dataset = ACEDataset(
        tokenizer=tokenizer,
        args=args,
        evaluate=True,  # 评估模式
        do_test=do_test,  # 是否为测试模式
        max_pair_length=args.max_pair_length,  # 最大实体对长度
    )

    # 3. 获取真实标签和标签列表
    # 关系抽取相关
    golden_labels = set(eval_dataset.golden_labels)  # 关系真实标签集合
    golden_labels_withner = set(eval_dataset.golden_labels_withner)  # 带实体类型的关系真实标签
    label_list = list(eval_dataset.label_list)  # 关系标签列表
    sym_labels = list(eval_dataset.sym_labels)  # 对称标签列表
    tot_recall = eval_dataset.tot_recall  # 关系总数

    # 限定词关系抽取相关
    q_golden_labels = set(eval_dataset.q_golden_labels)  # 限定词关系真实标签集合
    q_golden_labels_withner = set(eval_dataset.q_golden_labels_withner)  # 带实体类型的限定词关系真实标签
    q_label_list = list(eval_dataset.q_label_list)  # 限定词关系标签列表
    q_tot_recall = eval_dataset.q_tot_recall  # 限定词关系总数

    # 4. 创建评估输出目录
    if not os.path.exists(eval_output_dir) and len(args.cuda_device) > 0:
        os.makedirs(eval_output_dir)

    # 5. 设置评估批次大小
    if args.use_supervised_contrastive:
        # 监督对比学习模式：批次大小 = 标签数量 × 每个标签的样本数
        args.eval_batch_size = args.n_labels_per_batch * args.k_samples_per_label
    else:
        # 普通评估模式：批次大小 = 命令行参数中设置的批次大小
        args.eval_batch_size = args.eval_batch_size

    # 6. 初始化评估相关变量
    scores = defaultdict(dict)  # 存储评估分数
    example_subs = set()  # 存储示例主语
    # 计算有效标签数量
    num_label = int((len(label_list) + len(sym_labels)) / 2)
    num_q_label = int((len(q_label_list) + len(sym_labels)) / 2)

    # 7. 记录评估开始日志
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # 8. 设置模型为评估模式
    model.eval()

    # 9. 创建评估数据加载器
    eval_sampler = SequentialSampler(eval_dataset)  # 顺序采样器，确保评估结果可重复
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=ACEDataset.collate_fn,  # 自定义批处理函数
        num_workers=4 * int(args.output_dir.find("test") == -1),  # 测试模式下减少worker数量
    )

    # 10. 记录评估示例数量
    logger.info("  Num examples = %d", len(eval_dataset))
    start_time = timeit.default_timer()  # 记录评估开始时间

    # 11. 批次评估循环
    for batch in tqdm(eval_dataloader, desc="Evaluating", dynamic_ncols=True):
        # 12. 提取元数据
        indexs = batch[-4]  # 样本索引
        subs = batch[-3]  # 主语实体
        batch_m2s = batch[-2]  # 宾语实体列表
        q_batch_m2s = batch[-1]  # 限定词实体列表

        # 13. 将模型输入数据移至指定设备
        batch = tuple(t.to(args.device) for t in batch[:-4])

        # 14. 模型预测（不计算梯度）
        with torch.no_grad():
            # 构建模型输入
            inputs = {
                "input_ids": batch[0],  # 输入token IDs
                "attention_mask": batch[1],  # 注意力掩码
                "position_ids": batch[2],  # 位置 IDs
                "sub_positions": batch[3],  # 主语位置
                "mention_pos": batch[4],  # 提及位置
            }

            # 模型前向传播
            outputs = model(**inputs)
            logits = outputs[0]  # 关系预测logits
            q_logits = outputs[2]  # 限定词关系预测logits

            # 应用softmax或log_softmax
            if args.eval_logsoftmax:
                logits = torch.nn.functional.log_softmax(logits, dim=-1)
                q_logits = torch.nn.functional.log_softmax(q_logits, dim=-1)
            elif args.eval_softmax:
                logits = torch.nn.functional.softmax(logits, dim=-1)
                q_logits = torch.nn.functional.softmax(q_logits, dim=-1)

            # 实体类型预测
            ner_preds = torch.argmax(outputs[1], dim=-1)  # 实体类型预测
            q_ner_preds = torch.argmax(outputs[3], dim=-1)  # 限定词实体类型预测

            # 将预测结果移至CPU并转换为numpy数组
            logits = logits.cpu().numpy()
            q_logits = q_logits.cpu().numpy()
            ner_preds = ner_preds.cpu().numpy()
            q_ner_preds = q_ner_preds.cpu().numpy()

            for i in range(len(indexs)):
                index = indexs[i]
                sub = subs[i]
                m2s = batch_m2s[i]
                q_m2s = q_batch_m2s[i]
                example_subs.add(((index[0], index[1]), (sub[0], sub[1])))

                for j in range(len(m2s)):
                    ner_label = eval_dataset.ner_label_list[ner_preds[i, j]]
                    for k in range(len(q_m2s[j])):
                        obj = m2s[j][k]
                        q = q_m2s[j][k]
                        q_ner_label = eval_dataset.ner_label_list[q_ner_preds[i, j, k]]
                        scores[(index[0], index[1])][((sub[0], sub[1]), (obj[0], obj[1]), (q[0], q[1]))] = (
                            logits[i, j, k].tolist(),
                            ner_label,
                            q_logits[i, j, k].tolist(),
                            q_ner_label,
                        )

    cor, q_cor, tot_pred, tot_pred_r, cor_with_ner, q_cor_with_ner = 0, 0, 0, 0, 0, 0
    global_predicted_ners = eval_dataset.global_predicted_ners
    ner_golden_labels = eval_dataset.ner_golden_labels
    ner_cor, ner_tot_pred, ner_ori_cor = 0, 0, 0
    tot_output_results = defaultdict(list)

    for example_index, pair_dict in sorted(scores.items(), key=lambda x: x[0]):
        visited = set([])
        sentence_results = []
        for k123, (v123, v1_ner_label, q123, _) in pair_dict.items():
            if k123 in visited:
                continue
            visited.add(k123)
            v = list(v123)
            q = list(q123)
            m1 = k123[0]
            m2 = k123[1]
            m3 = k123[2]
            if not args.sameentity and (m1 == m2 or m2 == m3 or m3 == m1):
                continue
            k213 = (m2, m1, m3)
            v213s = pair_dict.get(k213, None)
            if v213s:
                visited.add(k213)
                v213, v2_ner_label, q213, _ = v213s
                v213 = v213[: len(sym_labels)] + v213[num_label:] + v213[len(sym_labels) : num_label]
                for j in range(len(v213)):
                    v[j] += v213[j]
                for j in range(len(q213)):
                    q[j] += q213[j]
            else:
                assert False

            k132 = (m1, m3, m2)
            v132s = pair_dict.get(k132, None)
            if v132s:
                visited.add(k132)
                v132, _, q132, _ = v132s
                temp = v132
                v132 = q132
                q132 = temp
                for j in range(len(v132)):
                    v[j] += v132[j]
                for j in range(len(q132)):
                    q[j] += q132[j]
            else:
                assert False

            k231 = (m2, m3, m1)
            v231s = pair_dict.get(k231, None)
            if v231s:
                visited.add(k231)
                v231, _, q231, _ = v231s
                temp = v231
                v231 = q231
                q231 = temp[: len(sym_labels)] + temp[num_label:] + temp[len(sym_labels) : num_label]
                for j in range(len(v231)):
                    v[j] += v231[j]
                for j in range(len(q231)):
                    q[j] += q231[j]
            else:
                assert False

            k312 = (m3, m1, m2)
            v312s = pair_dict.get(k312, None)
            if v312s:
                visited.add(k312)
                v312, v3_ner_label, q312, _ = v312s
                temp = v312
                v312 = q312[: len(sym_labels)] + q312[num_q_label:] + q312[len(sym_labels) : num_q_label]
                q312 = temp
                for j in range(len(v312)):
                    v[j] += v312[j]
                for j in range(len(q312)):
                    q[j] += q312[j]
            else:
                assert False

            k321 = (m3, m2, m1)
            v321s = pair_dict.get(k321, None)
            if v321s:
                visited.add(k321)
                v321, _, q321, _ = v321s
                q321 = q321[: len(sym_labels)] + q321[num_q_label:] + q321[len(sym_labels) : num_q_label]
                for j in range(len(v321)):
                    v[j] += v321[j]
                for j in range(len(q321)):
                    q[j] += q321[j]
            else:
                assert False

            pred_label = np.argmax(v)
            q_pred_label = np.argmax(q)
            if pred_label > 0 and q_pred_label > 0:
                if pred_label >= num_label:
                    pred_label = pred_label - num_label + len(sym_labels)
                    m1, m2, m3 = m2, m1, m3
                    v1_ner_label, v2_ner_label = v2_ner_label, v1_ner_label

                if q_pred_label >= num_q_label:
                    m1, m2, m3 = m3, m1, m2
                    temp = pred_label
                    pred_label = q_pred_label - num_q_label + len(sym_labels)
                    q_pred_label = temp
                    v1_ner_label, v2_ner_label, v3_ner_label = (
                        v3_ner_label,
                        v1_ner_label,
                        v2_ner_label,
                    )

                if label_list[pred_label].startswith("[k]"):
                    if q_label_list[q_pred_label].startswith("[k]"):
                        continue
                    m1, m2, m3 = m1, m3, m2
                    pred_label, q_pred_label = q_pred_label, pred_label
                    v1_ner_label, v2_ner_label, v3_ner_label = (
                        v1_ner_label,
                        v3_ner_label,
                        v2_ner_label,
                    )

                if label_list[pred_label].startswith("[r]"):
                    if q_label_list[q_pred_label].startswith("[r]"):
                        continue

                pred_score = v[pred_label]
                q_pred_score = q[q_pred_label]

                sentence_results.append(
                    (
                        pred_score,
                        m1,
                        m2,
                        pred_label,
                        v1_ner_label,
                        v2_ner_label,
                        q_pred_score,
                        m3,
                        q_pred_label,
                        q_ner_label,
                    )
                )

        sentence_results.sort(key=lambda x: -x[0])
        no_overlap = []

        def is_overlap(m1, m2):
            if m2[0] <= m1[0] and m1[0] <= m2[1]:
                return True
            if m1[0] <= m2[0] and m2[0] <= m1[1]:
                return True
            return False

        output_preds = []

        for item in sentence_results:
            m1 = item[1]
            m2 = item[2]
            m3 = item[-3]
            overlap = False
            for x in no_overlap:
                _m1 = x[1]
                _m2 = x[2]
                _m3 = x[-3]
                if item[3] == x[3] and (is_overlap(m1, _m1) and is_overlap(m2, _m2)) and item[-2] == x[-2] and is_overlap(m3, _m3):
                    overlap = True
                    break

            if not overlap:
                no_overlap.append(item)

        pos2ner = {}
        q_pos2ner = {}
        relation_visited = []
        rq_visited = []

        for item in no_overlap:
            m1 = item[1]
            m2 = item[2]
            m3 = item[-3]
            pred_label = label_list[item[3]]
            q_pred_label = q_label_list[item[-2]]
            is_visited_r = (example_index, m1, m2, pred_label) not in relation_visited
            is_visited_rq = (
                example_index,
                m1,
                m2,
                pred_label,
                m3,
                q_pred_label,
            ) not in rq_visited

            if is_visited_r:
                tot_pred_r += 1
                relation_visited.append((example_index, m1, m2, pred_label))

            if is_visited_rq:
                tot_pred += 1
                rq_visited.append((example_index, m1, m2, pred_label, m3, q_pred_label))

            ner_results = list(global_predicted_ners[example_index])
            for m in ner_results:
                pos2ner[(m[0], m[1])] = m[2]
                q_pos2ner[(m[0], m[1])] = m[2]

            output_preds.append((m1, m2, pred_label, m3, q_pred_label))

            if is_visited_r and (example_index, m1, m2, pred_label) in golden_labels:
                cor += 1
            if (
                is_visited_r
                and (
                    example_index,
                    (m1[0], m1[1], pos2ner[m1]),
                    (m2[0], m2[1], pos2ner[m2]),
                    pred_label,
                )
                in golden_labels_withner
            ):
                cor_with_ner += 1
            if is_visited_rq and (example_index, m1, m2, pred_label, m3, q_pred_label) in q_golden_labels:
                q_cor += 1
            if (
                is_visited_rq
                and (
                    example_index,
                    (m1[0], m1[1], pos2ner[m1]),
                    (m2[0], m2[1], pos2ner[m2]),
                    pred_label,
                    (m3[0], m3[1], q_pos2ner[m3]),
                    q_pred_label,
                )
                in q_golden_labels_withner
            ):
                q_cor_with_ner += 1

        tot_output_results[example_index[0]].append((example_index[1], output_preds))

        ner_results = list(global_predicted_ners[example_index])
        for start, end, label in ner_results:
            if (example_index, (start, end), label) in ner_golden_labels:
                ner_ori_cor += 1
            if (start, end) in pos2ner:
                label = pos2ner[(start, end)]
            if (example_index, (start, end), label) in ner_golden_labels:
                ner_cor += 1
            ner_tot_pred += 1

    evalTime = timeit.default_timer() - start_time
    logger.info(
        "  Evaluation done in total %f secs (%f example per second)",
        evalTime,
        len(global_predicted_ners) / evalTime,
    )

    if do_test:
        output_w = open(os.path.join(args.output_dir, "test_pred_results.json"), "w")
        json.dump(tot_output_results, output_w, ensure_ascii=False)
        output_w.close()

        result_set = to_gran_format(
            result_file=os.path.join(args.output_dir, "test_pred_results.json"),
            label_file=os.path.join(args.data_dir, args.test_file),
            output_file=os.path.join(args.output_dir, "test_hkg_results.json"),
        )
        res_comp_table = compaction(
            result_set,
            result_comp_file=os.path.join(args.output_dir, "test_hkg_results_comp.json"),
        )
        results_comp = statistic(res_comp_table, test_file=os.path.join(args.data_dir, args.test_file))
        output_w = open(os.path.join(args.output_dir, "test_pred_results_comp.json"), "w")
        json.dump(results_comp, output_w, ensure_ascii=False)
        output_w.close()
    else:
        output_w = open(os.path.join(args.output_dir, "valid_pred_results.json"), "w")
        json.dump(tot_output_results, output_w, ensure_ascii=False)
        output_w.close()

        result_set = to_gran_format(
            result_file=os.path.join(args.output_dir, "valid_pred_results.json"),
            label_file=os.path.join(args.data_dir, args.dev_file),
            output_file=os.path.join(args.output_dir, "valid_hkg_results.json"),
        )
        res_comp_table = compaction(
            result_set,
            result_comp_file=os.path.join(args.output_dir, "valid_hkg_results_comp.json"),
        )
        results_comp = statistic(res_comp_table, test_file=os.path.join(args.data_dir, args.dev_file))
        output_w = open(os.path.join(args.output_dir, "valid_pred_results_comp.json"), "w")
        json.dump(results_comp, output_w, ensure_ascii=False)
        output_w.close()

    ner_p = ner_cor / ner_tot_pred if ner_tot_pred > 0 else 0
    ner_r = ner_cor / len(ner_golden_labels)
    ner_f1 = 2 * (ner_p * ner_r) / (ner_p + ner_r) if ner_cor > 0 else 0.0

    p = cor / tot_pred_r if tot_pred_r > 0 else 0
    r = cor / tot_recall
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0

    q_p = q_cor / tot_pred if tot_pred > 0 else 0
    q_r = q_cor / q_tot_recall
    q_f1 = 2 * (q_p * q_r) / (q_p + q_r) if q_cor > 0 else 0.0

    p_with_ner = cor_with_ner / tot_pred_r if tot_pred_r > 0 else 0
    r_with_ner = cor_with_ner / tot_recall
    f1_with_ner = 2 * (p_with_ner * r_with_ner) / (p_with_ner + r_with_ner) if cor_with_ner > 0 else 0.0

    q_p_with_ner = q_cor_with_ner / tot_pred if tot_pred > 0 else 0
    q_r_with_ner = q_cor_with_ner / q_tot_recall
    q_f1_with_ner = 2 * (q_p_with_ner * q_r_with_ner) / (q_p_with_ner + q_r_with_ner) if q_cor_with_ner > 0 else 0.0

    results = {
        "f1": f1,
        "f1_with_ner": f1_with_ner,
        "q_f1": q_f1,
        "q_f1_with_ner": q_f1_with_ner,
        "ner_f1": ner_f1,
    }
    logger.info("Result: %s", json.dumps(results, ensure_ascii=False))

    results_p = {
        "p": p,
        "p_with_ner": p_with_ner,
        "q_p": q_p,
        "q_p_with_ner": q_p_with_ner,
        "ner_p": ner_p,
    }
    results.update(results_p)
    logger.info("Result: %s", json.dumps(results_p, ensure_ascii=False))

    results_r = {
        "r": r,
        "r_with_ner": r_with_ner,
        "q_r": q_r,
        "q_r_with_ner": q_r_with_ner,
        "ner_r": ner_r,
    }
    results.update(results_r)
    logger.info("Result: %s", json.dumps(results_r, ensure_ascii=False))

    results_num = {
        "correct_r": cor,
        "num_r_ans": tot_recall,
        "num_r_pred": tot_pred_r,
        "correct_q": q_cor,
        "num_q_ans": q_tot_recall,
        "num_q_pred": tot_pred,
    }
    results.update(results_num)
    logger.info("Result: %s", json.dumps(results_num, ensure_ascii=False))

    results.update(results_comp)

    return results


def to_gran_format(result_file, label_file, output_file):
    resf = open(result_file, "r")
    res_dict = json.load(resf)
    testf = open(label_file, "r")
    test_lines = testf.readlines()
    if os.path.exists(output_file):
        os.remove(output_file)
    rawf = open(output_file, "a")
    res_set = []
    for i in range(0, test_lines.__len__()):
        if str(i) not in res_dict.keys():
            test_dict = json.loads(test_lines[i])
            num_sens = len(test_dict["relations"])
            res_dict[str(i)] = []
            for k in range(num_sens):
                res_dict[str(i)].append([k, []])
    for i in range(0, test_lines.__len__()):
        hypers = []
        # res_set.append([])
        test_dict = json.loads(test_lines[i])
        sentence = test_dict["sentences"][0]
        for hyper_relation in res_dict[str(i)]:
            for hr in hyper_relation[1]:
                sub = ""
                obj = ""
                att = ""
                for index in range(hr[0][0], hr[0][1]):
                    sub = sub + sentence[index] + " "
                sub = sub + sentence[hr[0][1]]
                for index in range(hr[1][0], hr[1][1]):
                    obj = obj + sentence[index] + " "
                obj = obj + sentence[hr[1][1]]
                for index in range(hr[3][0], hr[3][1]):
                    att = att + sentence[index] + " "
                att = att + sentence[hr[3][1]]
                hyper = {
                    "N": 3,
                    "relation": hr[2],
                    "subject": sub,
                    "object": obj,
                    hr[4]: [att],
                }
                hyper = json.dumps(hyper, ensure_ascii=False) + "\n"
                hypers.append(hyper)
        rawf.writelines(hypers)
        res_set.append(hypers)
    return res_set


# 将超关系五元组合并为完整超关系
def compaction(res_set, result_comp_file):
    if os.path.exists(result_comp_file):
        os.remove(result_comp_file)
    resf_comp = open(result_comp_file, "a")
    res_table = []

    for res_line in res_set:
        res_comp_line = []
        # 用 map 将主三元组相同的超关系归到一类
        hy_map = {}
        for index in range(res_line.__len__()):
            res_dict = json.loads(res_line[index])
            rso = res_dict["relation"] + res_dict["subject"] + res_dict["object"]
            if rso in hy_map.keys():
                hy_map[rso].append(res_dict)
            else:
                hy_map[rso] = [res_dict]
        # 构建合并后的超关系
        for rso, ds in hy_map.items():
            t_d = {"N": 0}
            ext = 0
            for d in ds:
                for k, v in d.items():
                    if k in t_d.keys() and k != "relation" and k != "subject" and k != "object" and k != "N":
                        t_d[k] += v
                        ext += 1
                    else:
                        t_d[k] = v
            t_d["N"] = t_d.__len__() - 2 + ext
            res_comp_line.append(json.dumps(t_d, ensure_ascii=False))
        res_table.append(res_comp_line)
        formal_res_comp_line = []
        for hyper_relation in res_comp_line:
            formal_res_comp_line.append(hyper_relation + "\n")
        resf_comp.writelines(formal_res_comp_line)
    return res_table


def statistic(res_table, test_file):
    testf = open(test_file, "r")
    test_lines = testf.readlines()
    num_result = 0
    match = 0
    num_label = 0
    N_of_result = {}
    N_of_test = {}
    for i in range(0, test_lines.__len__()):
        res_list = res_table[i]
        test_dict = json.loads(test_lines[i])
        label_relations = test_dict["relations"][0]
        sentence = test_dict["sentences"][0]

        text_label_relations = []
        for label_relation in label_relations:
            sub = ""
            obj = ""
            att = ""
            text_label_relation = {"N": 0}
            for index in range(label_relation[0], label_relation[1]):
                sub = sub + sentence[index] + " "
            sub = sub + sentence[label_relation[1]]
            text_label_relation["relation"] = label_relation[4]
            text_label_relation["subject"] = sub
            for index in range(label_relation[2], label_relation[3]):
                obj = obj + sentence[index] + " "
            obj = obj + sentence[label_relation[3]]
            text_label_relation["object"] = obj
            ext = 0
            for att_pair in label_relation[5]:
                for index in range(att_pair[0], att_pair[1]):
                    att = att + sentence[index] + " "
                att = att + sentence[att_pair[1]]
                if att_pair[2] in text_label_relation.keys():
                    text_label_relation[att_pair[2]] += [att]
                    ext += 1
                else:
                    text_label_relation[att_pair[2]] = [att]
            text_label_relation["N"] = text_label_relation.__len__() - 2 + ext
            num_label += 1
            text_label_relations.append(json.dumps(text_label_relation, ensure_ascii=False))

        # 在同一个段落里作比较
        for res_hr in res_list:
            num_result += 1
            for label_hr in text_label_relations:
                if res_hr == label_hr:
                    match += 1

        for res_hr in res_list:
            res_hr = json.loads(res_hr)
            if res_hr["N"] in N_of_result.keys():
                N_of_result[res_hr["N"]] += 1
            else:
                N_of_result[res_hr["N"]] = 1

        for label_hr in text_label_relations:
            label_hr = json.loads(label_hr)
            if label_hr["N"] in N_of_test.keys():
                N_of_test[label_hr["N"]] += 1
            else:
                N_of_test[label_hr["N"]] = 1
    logger.info("match_comp = %s", match)
    logger.info("num_pred_comp = %s", num_result)
    logger.info("num_ans_comp = %s", num_label)
    p = match / num_result if num_result > 0 else 0.0
    r = match / num_label
    f1 = 2 * (p * r) / (p + r) if match > 0 else 0.0
    logger.info("p_comp = %s", p)
    logger.info("r_comp = %s", r)
    logger.info("f1_comp = %s", f1)
    logger.info("N_of_pred_comp = %s", N_of_result)
    logger.info("N_of_ans_comp = %s", N_of_test)
    return {
        "p_comp": p,
        "r_comp": r,
        "f1_comp": f1,
        "N_of_pred_comp": N_of_result,
        "N_of_ans_comp": N_of_test,
        "num_ans_comp": num_label,
        "num_pred_comp": num_result,
        "correct_comp": match,
    }


def main():
    parser = argparse.ArgumentParser()
    ##################################################################################################
    ## Required parameters
    # selec-dataset/naryschema !/.
    parser.add_argument("--dataset", default="hyperred_hyperrelation", type=str)
    parser.add_argument("--data_dir", default="data/geoshr", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--num_train_epochs", default=10.0, type=float)
    ##################################################################################################
    # select-cuda
    parser.add_argument("--cuda_device", default="0", type=str)
    ##################################################################################################
    # select-train/test
    parser.add_argument("--test_when_update", type=bool, default=True)  # True, don't change
    parser.add_argument("--do_train", action="store_true", default=True, help="Whether to run training.")  # True/False, don't change
    parser.add_argument(
        "--do_eval",
        action="store_true",
        default=True,
        help="Whether to run eval on the dev set.",
    )  # True, don't change
    ##################################################################################################
    parser.add_argument("--model_name_or_path", default="pretrained_models/bert-base-uncased", type=str)
    # "pretrained_models/roberta-base", "gk/知识嵌入+规则对齐/geology_contrastive_bert"
    ##################################################################################################
    # select-seed s
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")  # 42,43,44,45,46
    ##################################################################################################
    # select-(alpha,q_alpha) a
    parser.add_argument("--alpha", default=0.01, type=float)  # 1.0, 0.1, 0.01(best), 0.001, 0.0001
    parser.add_argument("--q_alpha", default=0.01, type=float)  # 1.0, 0.1, 0.01(best), 0.001, 0.0001
    ###################################################################################################
    # select-bs/lr p
    parser.add_argument(
        "--train_batch_size",
        default=48,
        type=int,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )  # 2e-5
    ###################################################################################################

    ## Other parameters
    parser.add_argument("--save_steps", type=int, default=1000)  # 1000
    parser.add_argument("--smallerdataset", default=False, type=bool)  # False
    parser.add_argument("--sameentity", default=False, type=bool)
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        default=True,
        help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        default=True,
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--eval_batch_size",
        default=48,
        type=int,
        help="Batch size for evaluation.",
    )  # 1
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=-1, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=5, help="Log every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        default=True,
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        default=True,
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=1,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )

    parser.add_argument("--train_file", default="train.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--label_file", default="label.json", type=str)

    parser.add_argument("--max_pair_length", type=int, default=32, help="")

    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--no_test", action="store_true")
    parser.add_argument("--eval_logsoftmax", action="store_true", default=True)
    parser.add_argument("--eval_softmax", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--lminit", action="store_true")
    parser.add_argument("--no_sym", action="store_true")
    parser.add_argument("--att_left", action="store_true")
    parser.add_argument("--att_right", action="store_true")
    parser.add_argument("--use_typemarker", action="store_true")
    parser.add_argument("--eval_unidirect", action="store_true")
    parser.add_argument("--scl_alpha", default=1.0, type=float, help="监督对比学习损失权重")
    parser.add_argument("--gkc_alpha", default=100.0, type=float, help="地质知识约束损失权重")
    parser.add_argument("--projection_dim", default=128, type=int, help="监督对比学习的投影维度")
    parser.add_argument("--temperature", default=0.07, type=float, help="监督对比学习的温度参数")
    parser.add_argument(
        "--use_supervised_contrastive",
        default=True,
        type=bool,
        help="是否使用监督对比学习",
    )
    parser.add_argument("--n_labels_per_batch", default=6, type=int, help="每个批次的标签数量")
    parser.add_argument("--k_samples_per_label", default=6, type=int, help="每个标签的样本数量")

    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3"
    # add new dataset labels for entity, relation and qualifier
    label_file = os.path.join(args.data_dir, args.label_file)

    with open(label_file, "r") as f:
        labels = json.load(f)
        task_ner_labels[args.dataset] = [list(labels["id"].keys())[i] for i in labels["entity"]]
        task_rel_labels[args.dataset] = [list(labels["id"].keys())[i] for i in labels["relation"]]
        task_q_labels[args.dataset] = [list(labels["id"].keys())[i] for i in labels["qualifier"]]

    # Setup CUDA, GPU training
    if len(args.cuda_device) > 0 and not args.no_cuda:
        device = ",".join(args.cuda_device)
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        device = torch.device("cuda:" + args.cuda_device[0] if torch.cuda.is_available() else "cpu")
        args.n_gpu = 1
    else:
        device = torch.device("cpu")
        args.n_gpu = 0
    args.device = device

    # Setup logging
    # 创建日志目录
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")

    # 创建基础配置
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if len(args.cuda_device) > 0 else logging.WARN,
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
            logging.FileHandler(log_file, mode="a"),  # 追加模式输出到文件
        ],
    )
    logger.warning(
        "device: %s, n_gpu: %s, 16-bits training: %s",
        device,
        args.n_gpu,
        args.fp16,
    )

    # Set seed
    set_seed(args)

    num_labels = len(set(task_rel_labels[args.dataset] + task_q_labels[args.dataset])) * 2 + 1
    num_ner_labels = len(task_ner_labels[args.dataset]) + 1
    num_q_labels = len(set(task_rel_labels[args.dataset] + task_q_labels[args.dataset])) * 2 + 1

    # 加载预训练模型与分词器
    config_class = BertConfig
    model_class = BertForACEBothOneDropoutSub
    tokenizer_class = BertTokenizer

    # BertConfig, BertTokenizer, BertModel Setting
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
    )
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    config.max_seq_length = args.max_seq_length
    config.alpha = args.alpha
    config.q_alpha = args.q_alpha
    config.num_ner_labels = num_ner_labels
    config.num_q_labels = num_q_labels

    # 监督对比学习损失参数
    config.scl_alpha = args.scl_alpha  # 监督对比学习损失的权重
    config.projection_dim = args.projection_dim  # 监督对比学习的投影维度
    config.temperature = args.temperature  # 监督对比学习的温度参数
    # 地质知识约束损失参数
    config.gkc_alpha = args.gkc_alpha  # 地质知识约束损失的权重

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    model.to(args.device)

    def re_init_weights(model):
        """
        只初始化非 BERT 的层（即下游任务的 Head），保留 BERT 参数不变。
        """
        # 遍历所有子模块
        for name, module in model.named_modules():
            # 1. 跳过根节点本身
            if name == "":
                continue

            # 2. 核心过滤：如果名字以 "bert" 开头，说明它是 BERT 主干的一部分，绝对不能动
            # (因为你的代码里写了 self.bert = BertModel(config)，所以名字一定是 bert.xxx)
            if name.startswith("bert"):
                continue

            # 3. 对剩下的层（dropout, classifier, ner_head 等）进行初始化
            # 这里判断是否是常见的有参数的层 (Linear, Conv, Embedding, LayerNorm)
            # if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d, nn.LayerNorm)):

            #     # 【关键】使用 HuggingFace 官方的初始化逻辑
            #     # BertPreTrainedModel 里面自带了 _init_weights 方法
            #     # 它会根据 config (std=0.02) 来初始化，比 module.reset_parameters() 更适合 BERT
            #     model._init_weights(module)

            #     logger.info(f"已重置层: {name} | 类型: {type(module).__name__}")
            if hasattr(module, "reset_parameters"):
                # 注意：某些自定义层可能没有这个方法，或者 LayerNorm 等层不需要随机初始化
                try:
                    module.reset_parameters()
                    logger.info("Initialized %s", module)
                except Exception as e:
                    logger.info("Skipped %s: %s", module, e)

    logger.info("Training/evaluation parameters %s", args)
    best_f1 = 0
    # Training
    if args.do_train:
        # 在训练前，务必执行这一步！
        re_init_weights(model)
        # train_dataset = load_and_cache_examples(args,  tokenizer, evaluate=False)
        global_step, tr_loss, best_f1 = train(args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # 保存最佳模型：如果你使用模型的默认名称，可以通过 from_pretrained() 重新加载它
    if args.do_train and len(args.cuda_device) > 0:
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and len(args.cuda_device) > 0:
            os.makedirs(args.output_dir)
        update = True
        if args.evaluate_during_training:
            results = evaluate(args, model, tokenizer)
            f1 = results["q_f1"]  # f1 = results['f1_with_ner']
            if f1 > best_f1:
                best_f1 = f1
                logger.info("Best F1 %s", best_f1)
            else:
                update = False

        if update:
            checkpoint_prefix = "checkpoint"
            output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            model.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)
            _rotate_checkpoints(args, checkpoint_prefix)

        # 创建分词器目录
        tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_dir)

    # Evaluation
    results = {"dev_best_f1": best_f1}
    if args.do_eval and len(args.cuda_device) > 0:
        checkpoints = [args.output_dir]

        WEIGHTS_NAME = "model.safetensors"

        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""

            model = model_class.from_pretrained(checkpoint, config=config)

            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step, do_test=not args.no_test)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
        logger.info("%s", results)

        if args.no_test:  # choose best resutls on dev set
            bestv = 0
            k = 0
            for k, v in results.items():
                if v > bestv:
                    bestk = k
            logger.info("%s", bestk)

        output_eval_file = os.path.join(args.output_dir, "results.json")
        json.dump(results, open(output_eval_file, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
