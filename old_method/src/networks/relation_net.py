import torch
from torch import nn
import math
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}
class Mlp(nn.Module):
    def __init__(self, hidden_size):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 3072)
        self.fc2 = nn.Linear(3072, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class RelationNet(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim * 2),
                                    nn.BatchNorm1d(feature_dim * 2),
                                    nn.LeakyReLU(),
                                    nn.Linear(feature_dim * 2, feature_dim))

    def forward(self, features_1, features_2):
        pair = torch.cat([features_1, features_2], dim=1)
        return self.layers(pair)


class DisjointRelationNet(nn.Module):
    def __init__(self, feature_dim, out_dim, num_classes,dropout_prob=0.5):
        super().__init__()
        self.attention_norm = nn.LayerNorm(feature_dim, eps=1e-6)
        self.attn = Attention(feature_dim)
        self.ffn_norm = nn.LayerNorm(feature_dim, eps=1e-6)
        self.ffn = Mlp(feature_dim)
        self.layers = nn.Sequential(
                                    nn.Linear(feature_dim, num_classes)
                                    )

    def forward(self,x):
        h =x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        x_last = x[:, -1, :]  # 提取最后一个 num 维度的元素，得到形状 [b, 512]

        #pair = torch.cat([features_1, features_2,features_3], dim=1)
        return self.layers(x_last)


class Mapper(nn.Module):
    def __init__(self, feature_dim, out_dim, num_classes,dropout_prob=0.5):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(feature_dim, feature_dim * 2),
                                    nn.BatchNorm1d(feature_dim * 2),
                                    nn.LeakyReLU(),
                                    nn.Dropout(dropout_prob),
                                    nn.Linear(feature_dim * 2, out_dim),
                                    nn.LeakyReLU(),
                                    nn.Dropout(dropout_prob),
                                    nn.Linear(out_dim, num_classes),
                                    )

    def forward(self, features):
        return self.layers(features)
    

class Attention(nn.Module):
    def __init__(self,feature_dim):
        super(Attention, self).__init__()
        self.num_attention_heads = 8
        self.attention_head_size = int(feature_dim/ self.num_attention_heads)
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)
        self.query = nn.Linear(feature_dim, self.all_head_size)
        self.key = nn.Linear(feature_dim, self.all_head_size)
        self.value =nn.Linear(feature_dim, self.all_head_size)

        self.out = nn.Linear(feature_dim, feature_dim)
        self.attn_dropout = nn.Dropout(0.0)
        self.proj_dropout = nn.Dropout(0.0)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
  
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights
