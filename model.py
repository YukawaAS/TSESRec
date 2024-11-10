import torch
from torch import nn
from torch.nn import functional as F
import copy


class Model(nn.Module):
    def __init__(self, blocks, model_dim, embed_num, padding_idx):  #blocks: TransformerBlocks的实例，表示要使用的Transformer块。model_dim: 模型的维度。embed_num: 嵌入层的词汇表大小。padding_idx: 嵌入层的填充索引。 
        super().__init__()
        self.embed = nn.Embedding(embed_num, model_dim, padding_idx)
        self.blocks = blocks
        self.pred_head = nn.Linear(model_dim, embed_num, bias=False)
        # embedding层和预测头共享权值
        self.pred_head.weight = self.embed.weight

        # 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):    #参数初始化
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def run(self, inputs): #embeding → blocks → pred_head
        embedding = self.embed(inputs)
        outputs = self.blocks(embedding)
        outputs = self.pred_head(outputs)
        return outputs

    def forward(self, inputs):
        return self.run(inputs)

    def predict(self, inputs):
        return self.run(inputs)[:, -1,:]


class TransformerBlocks(nn.Module):
    def __init__(self, num_layers, d_model: int, nhead: int, dim_feedforward, dropout=0.0,  #层数、模型维度、头数、前馈网络的维度
                 activation = F.relu,
                 layer_norm_eps: float = 1e-5):
        super().__init__()
        self.layer_norm_eps = layer_norm_eps
        block = TransformerBlock(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps)
        self.layers = _get_clones(block, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, tgt_key_padding_mask = None):
        output = tgt
        for mod in self.layers:
            output = mod(output, tgt_key_padding_mask=tgt_key_padding_mask)
        return output


class TransformerBlock(nn.Module):
    r"""
    Transformer block
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward, dropout=0.0,
                 activation = F.relu,
                 layer_norm_eps: float = 1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, tgt, tgt_key_padding_mask = None):
        att_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1], tgt.device)
        x = tgt
        x = self.norm1(x + self._sa_block(x, att_mask, tgt_key_padding_mask))   #self-attention block
        x = self.norm3(x + self._ff_block(x))   #feed forward block
        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_clones(module, N): #用于创建一个包含多个相同模块的 nn.ModuleList。
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


if __name__ == '__main__':
    torch.manual_seed(123)
    model_dim = 6
    embed_num = 10
    padd_idx = 0

    blocks = TransformerBlocks(2, model_dim, 2, 32) #层数、模型维度、头数、前馈网络的维度
    model = Model(blocks, model_dim, embed_num, padd_idx)

    inputs = torch.tensor([[1, 2, 1, 1, 2, 3, 4, 5, 7], #torch.Size([2, 9])
                           [2, 3, 2, 4, 3, 2, 3, 0, 0]])

    out = model(inputs)
    print(out.shape)
    print(out)
