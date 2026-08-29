import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, init
from transformers import BertModel, BertPreTrainedModel


class SupConLoss(nn.Module):
    """监督对比损失
    以用三元组的关系标签为基础，构造正负样本对
    """

    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        输入:
            features: 模型的表示向量, shape [bsz, projection_dim].
            labels: 样本的真实标签, shape [bsz].
        输出:
            一个标量的损失值.
        """
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)

        # 创建一个mask来标识哪些样本具有相同的标签
        # mask[i, j] = 1 if labels[i] == labels[j] else 0
        mask = torch.eq(labels, labels.T).float().to(device)

        # L2-normalize 特征
        features = nn.functional.normalize(features, dim=1)

        # 计算特征之间的点积相似度
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 创建一个mask来屏蔽对角线元素（即样本与自身的比较）
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # 计算log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # 计算每个样本的平均log-likelihood
        # mask.sum(1) 是每个样本的正样本数量
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        # 最终损失是所有样本损失的负均值
        loss = -mean_log_prob_pos.mean()

        return loss


class PosiNet(nn.Module):
    def __init__(self, d_model, n=49, simple=False):
        super(PosiNet, self).__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        if simple:
            self.position_biases = torch.zeros((n, n))
        else:
            self.position_biases = nn.Parameter(torch.ones((n, n)))
        self.d_model = d_model
        self.n = n
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def reset_parameters(self):
        # 重新执行初始化逻辑
        nn.init.ones_(self.position_biases)

    def forward(self, input):
        bs, n, dim = input.shape

        q = self.fc_q(input)  # bs,n,dim
        k = self.fc_k(input).view(1, bs, n, dim)  # 1,bs,n,dim
        v = self.fc_v(input).view(1, bs, n, dim)  # 1,bs,n,dim

        numerator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)) * v, dim=2)  # n,bs,dim
        denominator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)), dim=2)  # n,bs,dim

        out = numerator / denominator  # n,bs,dim
        out = self.sigmoid(q) * (out.permute(1, 0, 2))  # bs,n,dim

        return out


class Depth_Pointwise_Conv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        if k == 1:
            self.depth_conv = nn.Identity()
        else:
            self.depth_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, groups=1)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.pointwise_conv.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.depth_conv.weight, nonlinearity="relu")
        # 如果有偏置，也初始化
        if self.pointwise_conv.bias is not None:
            nn.init.zeros_(self.pointwise_conv.bias)
        if self.depth_conv.bias is not None:
            nn.init.zeros_(self.depth_conv.bias)

    def forward(self, x):
        out = self.pointwise_conv(self.depth_conv(x))
        return out


class qiaAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=0.1):
        super(qiaAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.conv1 = Depth_Pointwise_Conv1d(h * d_v, d_model, 1)
        self.conv3 = Depth_Pointwise_Conv1d(h * d_v, d_model, 3)
        self.conv5 = Depth_Pointwise_Conv1d(h * d_v, d_model, 5)
        self.dy_paras = nn.Parameter(torch.ones(3))
        self.softmax = nn.Softmax(-1)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def reset_parameters(self):
        # 重新执行初始化逻辑
        self.conv3.reset_parameters()
        self.conv5.reset_parameters()
        nn.init.ones_(self.dy_paras)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        # Self Attention
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        v2 = v.permute(0, 1, 3, 2).contiguous().view(b_s, -1, nk)  # bs,dim,n
        self.dy_paras = nn.Parameter(self.softmax(self.dy_paras))
        out2 = self.dy_paras[0] * self.conv1(v2) + self.dy_paras[1] * self.conv3(v2) + self.dy_paras[2] * self.conv5(v2)
        out2 = out2.permute(0, 2, 1)  # bs.n.dim

        out = out + out2
        return out


class BertForACEBothOneDropoutSub(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels
        self.num_q_labels = config.num_q_labels
        # 监督对比学习损失参数
        self.scl_alpha = config.scl_alpha  # 监督对比学习损失的权重
        self.projection_dim = config.projection_dim  # 监督对比学习的投影维度
        self.temperature = config.temperature  # 监督对比学习的温度参数

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m3 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.q_re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_q_labels)
        self.q_re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_q_labels)
        self.q_re_classifier_m3 = nn.Linear(config.hidden_size * 2, self.num_q_labels)

        self.alpha = torch.tensor([config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32)
        self.ner_alpha = torch.tensor([config.alpha] + [1.0] * (self.num_ner_labels - 1), dtype=torch.float32)
        self.q_alpha = torch.tensor([config.q_alpha] + [1.0] * (self.num_q_labels - 1), dtype=torch.float32)

        # 投影头：用于将BERT的输出映射到监督对比学习的表示空间
        self.projection_head = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(), nn.Linear(config.hidden_size, self.projection_dim))

        self.init_weights()

        self.posi = PosiNet(config.hidden_size * 2, 32)
        self.qia = qiaAttention(d_model=config.hidden_size * 2, d_k=config.hidden_size * 2, d_v=config.hidden_size * 2, h=8)
        self.fc = nn.Linear(config.hidden_size * 2 * 2, config.hidden_size * 2)

    def reset_parameters(self):
        # 重新执行初始化逻辑
        self.posi.reset_parameters()
        self.qia.reset_parameters()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mentions=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sub_positions=None,
        labels=None,
        ner_labels=None,
        q_labels=None,
        q_ner_labels=None,
        q2_labels=None,
        q3_labels=None,
        mention_pos=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs.last_hidden_state
        hidden_states = self.dropout(hidden_states)
        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        # 非主语实体特征
        e2_hidden_states = hidden_states[:, seq_len : seq_len + ent_len]
        e3_hidden_states = hidden_states[:, seq_len + ent_len :]

        feature_vector1 = torch.cat([e2_hidden_states, e3_hidden_states], dim=2)
        feature_vector2 = self.posi(feature_vector1)
        feature_vector3 = self.qia(feature_vector1, feature_vector1, feature_vector1)

        # Combine feature vectors with weights
        feature_vector = torch.cat([feature_vector2, feature_vector3], dim=-1)
        feature_vector = self.fc(feature_vector)

        ner_prediction_scores = self.ner_classifier(feature_vector)

        # 关系特征
        r_feature_vector = torch.stack(([feature_vector] * ent_len), dim=2)

        # 限定符特征
        q_feature_vector = torch.stack(([feature_vector] * ent_len), dim=1)
        q_ner_prediction_scores = self.ner_classifier(q_feature_vector)

        # 主语实体特征
        e1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        e1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        e1_feature_vector = torch.cat([e1_start_states, e1_end_states], dim=-1)

        # Calculate prediction scores
        m1_scores = self.re_classifier_m1(e1_feature_vector)  # bsz, num_label
        m2_scores = self.re_classifier_m2(r_feature_vector)  # bsz, ent_len, num_label
        m3_scores = self.re_classifier_m3(q_feature_vector)
        re_prediction_scores = m1_scores.unsqueeze(1).unsqueeze(2) + m2_scores + m3_scores

        q_m1_scores = self.q_re_classifier_m1(e1_feature_vector)  # bsz, num_label
        q_m2_scores = self.q_re_classifier_m2(r_feature_vector)  # bsz, ent_len, num_label
        q_m3_scores = self.q_re_classifier_m3(q_feature_vector)
        q_re_prediction_scores = q_m1_scores.unsqueeze(1).unsqueeze(2) + q_m2_scores + q_m3_scores

        outputs = (re_prediction_scores, ner_prediction_scores, q_re_prediction_scores, q_ner_prediction_scores)

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1, weight=self.alpha.to(re_prediction_scores))
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1, weight=self.ner_alpha.to(ner_prediction_scores))
            loss_fct_q_re = CrossEntropyLoss(ignore_index=-1, weight=self.q_alpha.to(q_re_prediction_scores))
            re_loss = loss_fct_re(re_prediction_scores.view(-1, self.num_labels), labels.view(-1))
            ner_loss = loss_fct_ner(ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1))
            q_re_loss = loss_fct_q_re(q_re_prediction_scores.view(-1, self.num_q_labels), q_labels.view(-1))

            pooled_hidden_states = torch.mean(hidden_states, dim=1)  # (bsz, hidden_size)  使用平均池化将序列表示转换为单个向量表示
            contrastive_labels = labels.view(bsz, -1).max(dim=1).values
            scl_loss = SupConLoss(temperature=self.temperature)(self.projection_head(pooled_hidden_states), contrastive_labels.view(-1))  # 监督对比学习损失
            loss = re_loss + ner_loss + q_re_loss + self.scl_alpha * scl_loss
            outputs = (loss, re_loss, ner_loss, q_re_loss, scl_loss) + outputs

            # loss = re_loss + ner_loss + q_re_loss + self.scl_alpha * 0
            # outputs = (loss, re_loss, ner_loss, q_re_loss, 0) + outputs

        return outputs
