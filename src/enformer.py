import math
from pathlib import Path
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint_sequential
from einops import rearrange
from einops.layers.torch import Rearrange
from transformers import PreTrainedModel


class Enformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 1536
        half_dim = self.dim // 2
        self.depth = 11
        self.heads = 8
        self.device = 'cuda'
        self.output_heads = dict(human = 5313, mouse = 1643)
        self.target_length = 896
        self.use_checkpointing = True
        self.attn_dim_key = 64
        self.dropout_rate = 0.4
        self.attn_dropout = 0.05
        self.pos_dropout = 0.01
        self.use_convnext = False
        self.num_downsamples = 7    # genetic sequence is downsampled 2 ** 7 == 128x in default Enformer - can be changed for higher resolution
        self.dim_divisible_by = 128
        self.use_tf_gamma = False

        # create stem

        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, 15, padding = 7),
            Residual(ConvBlock(half_dim)),
            AttentionPool(half_dim, pool_size = 2)
        )
        
        # self.stem = Stem(half_dim, pool_size = 2)
        
        # create conv tower

        filter_list = exponential_linspace_int(half_dim, self.dim, num = (self.num_downsamples - 1), divisible_by = self.dim_divisible_by)
        filter_list = [half_dim, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size = 5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size = 2)
            ))

        self.conv_tower = nn.Sequential(*conv_layers)

        transformer = []
        for _ in range(self.depth):
            transformer.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(self.dim),
                    Attention(
                        self.dim,
                        heads = self.heads,
                        dim_key = self.attn_dim_key,
                        dim_value = self.dim // self.heads,
                        dropout = self.attn_dropout,
                        pos_dropout = self.pos_dropout,
                        num_rel_pos_features = self.dim // self.heads,
                        use_tf_gamma = False
                    ),
                    nn.Dropout(self.dropout_rate)
                )),
                Residual(nn.Sequential(
                    nn.LayerNorm(self.dim),
                    nn.Linear(self.dim, self.dim * 2),
                    nn.Dropout(self.dropout_rate),
                    nn.ReLU(),
                    nn.Linear(self.dim * 2, self.dim),
                    nn.Dropout(self.dropout_rate)
                ))
            ))

        self.transformer = nn.Sequential(*transformer)

    
    def trunk_checkpointed(self, x):
        x = rearrange(x, 'b n d -> b d n')
        x = self.stem(x)
        x = self.conv_tower(x)
        # print(f"Memory useage of x after conv_tower: {x.element_size() * x.nelement() / 1024 / 1024} MB")
        # print(f"Max memory consumption: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
        x = rearrange(x, 'b d n -> b n d')
        # x = checkpoint_sequential(self.transformer, len(self.transformer), x)
        # x = self.crop_final(x)
        # x = self.final_pointwise(x)
        return x
   
    def forward(
        self,
        x,
        return_only_embeddings = False,
        target_length = None
    ):
        if isinstance(x, list):
            x = str_to_one_hot(x)

        elif type(x) == torch.Tensor and x.dtype == torch.long:
            x = seq_indices_to_one_hot(x)
        # x.to(self.device)

        no_batch = x.ndim == 2

        if no_batch:
            x = rearrange(x, '... -> () ...')

        trunk_fn = self.trunk_checkpointed if self.use_checkpointing else self._trunk
        x = trunk_fn(x)
        
        return x


# constants

SEQUENCE_LENGTH = 196_608
TARGET_LENGTH = 896

def map_values(fn, d):
    return {key: fn(values) for key, values in d.items()}

def exponential_linspace_int(start, end, num, divisible_by = 1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]

one_hot_embed = torch.zeros(256, 4)
one_hot_embed[ord('a')] = torch.Tensor([1., 0., 0., 0.])
one_hot_embed[ord('c')] = torch.Tensor([0., 1., 0., 0.])
one_hot_embed[ord('g')] = torch.Tensor([0., 0., 1., 0.])
one_hot_embed[ord('t')] = torch.Tensor([0., 0., 0., 1.])
one_hot_embed[ord('n')] = torch.Tensor([0., 0., 0., 0.])
one_hot_embed[ord('A')] = torch.Tensor([1., 0., 0., 0.])
one_hot_embed[ord('C')] = torch.Tensor([0., 1., 0., 0.])
one_hot_embed[ord('G')] = torch.Tensor([0., 0., 1., 0.])
one_hot_embed[ord('T')] = torch.Tensor([0., 0., 0., 1.])
one_hot_embed[ord('N')] = torch.Tensor([0., 0., 0., 0.])
one_hot_embed[ord('.')] = torch.Tensor([0.25, 0.25, 0.25, 0.25])

def cast_list(t):
    return t if isinstance(t, list) else [t]

def torch_fromstring(seq_strs):
    batched = not isinstance(seq_strs, str)
    seq_strs = cast_list(seq_strs)
    np_seq_chrs = list(map(lambda t: np.fromstring(t, dtype = np.uint8), seq_strs))
    seq_chrs = list(map(torch.from_numpy, np_seq_chrs))
    return torch.stack(seq_chrs) if batched else seq_chrs[0]

def str_to_one_hot(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return one_hot_embed[seq_chrs.long()]

def seq_indices_to_one_hot(t, padding = -1):
    is_padding = t == padding
    t = t.clamp(min = 0)
    one_hot = F.one_hot(t, num_classes = 5)
    out = one_hot[..., :4].float()
    out = out.masked_fill(is_padding[..., None], 0.25)
    return out

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def MaybeSyncBatchnorm(is_distributed = None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x

class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)

        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False)

        nn.init.dirac_(self.to_attn_logits.weight)

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):  
        # print(f"Memory consumption inside attention pool: {torch.cuda.memory_allocated() / 1024 / 1024} MB")      
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)
        
            # print(f"Memory consumption after padding: {x.element_size() * x.nelement() / 1024 / 1024} MB")
            # print(f"Memory consumption: {torch.cuda.memory_allocated() / 1024 / 1024} MB")
            # print(f"Max memory consumption: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")

        x = self.pool_fn(x)
        
        # print(f"Memory consumption after pool: {x.element_size() * x.nelement() / 1024 / 1024} MB")
        # print(f"Memory consumption: {torch.cuda.memory_allocated() / 1024 / 1024} MB")
        # print(f"Max memory consumption: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
        
        logits = self.to_attn_logits(x)
        
        # print(f"Memory consumption after logits: {logits.element_size() * logits.nelement() / 1024 / 1024} MB")
        # print(f"Memory consumption: {torch.cuda.memory_allocated() / 1024 / 1024} MB")
        # print(f"Max memory consumption: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
        
        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)
        
        # print(f"Memory consumption after masked fill: {logits.element_size() * logits.nelement() / 1024 / 1024} MB")
        # print(f"Memory consumption: {torch.cuda.memory_allocated() / 1024 / 1024} MB")
        # print(f"Max memory consumption: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")

        attn = logits.softmax(dim = -1)

        # print(f"Memory consumption after softmax: {attn.element_size() * attn.nelement() / 1024 / 1024} MB")
        # print(f"Memory consumption: {torch.cuda.memory_allocated() / 1024 / 1024} MB")
        # print(f"Max memory consumption: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
        
        out = (x * attn).sum(dim = -1)

        # print(f"Memory consumption after sum: {out.element_size() * out.nelement() / 1024 / 1024} MB")
        # print(f"Memory consumption: {torch.cuda.memory_allocated() / 1024 / 1024} MB")
        # print(f"Max memory consumption: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")

        return out

class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return x

        return x[:, -trim:trim]

def ConvBlock(dim, dim_out = None, kernel_size = 1, is_distributed = None):
    batchnorm_klass = MaybeSyncBatchnorm(is_distributed = is_distributed)

    return nn.Sequential(
        batchnorm_klass(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )

def get_positional_features_exponential(positions, features, seq_len, min_half_life = 3., dtype = torch.float):
    max_range = math.log(seq_len) / math.log(2.)
    half_life = 2 ** torch.linspace(min_half_life, max_range, features, device = positions.device)
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.) / half_life * positions)

def get_positional_features_central_mask(positions, features, seq_len, dtype = torch.float):
    center_widths = 2 ** torch.arange(1, features + 1, device = positions.device).to(dtype)
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).to(dtype)

def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1., x) - rate * x
    log_normalization = (torch.lgamma(concentration) - concentration * torch.log(rate))
    return torch.exp(log_unnormalized_prob - log_normalization)

def get_positional_features_gamma(positions, features, seq_len, stddev = None, start_mean = None, eps = 1e-8, dtype = torch.float):
    if not exists(stddev):
        stddev = seq_len / (2 * features)

    if not exists(start_mean):
        start_mean = seq_len / features

    mean = torch.linspace(start_mean, seq_len, features, device = positions.device)

    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev ** 2

    probabilities = gamma_pdf(positions.to(dtype).abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities, dim = -1, keepdim = True)
    return outputs

def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

def get_positional_embed(seq_len, feature_size, device, use_tf_gamma, dtype = torch.float):
    distances = torch.arange(-seq_len + 1, seq_len, device = device)

    assert not use_tf_gamma or seq_len == 1536, 'if using tf gamma, only sequence length of 1536 allowed for now'

    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma if not use_tf_gamma else always(TF_GAMMAS.to(device))
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(f'feature size is not divisible by number of components ({num_components})')

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len, dtype = dtype))

    embeddings = torch.cat(embeddings, dim = -1)
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None] * embeddings), dim = -1)
    return embeddings.to(dtype)

def relative_shift(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim = -1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., :((t2 + 1) // 2)]

# attention classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_rel_pos_features,
        heads = 8,
        dim_key = 64,
        dim_value = 64,
        dropout = 0.,
        pos_dropout = 0.,
        use_tf_gamma = False
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias = False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features

        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias = False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # whether to use tf gamma

        self.use_tf_gamma = use_tf_gamma

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        content_logits = einsum('b h i d, b h j d -> b h i j', q + self.rel_content_bias, k)

        positions = get_positional_embed(n, self.num_rel_pos_features, device, use_tf_gamma = self.use_tf_gamma, dtype = self.to_rel_k.weight.dtype)
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)

        rel_k = rearrange(rel_k, 'n (h d) -> h n d', h = h)
        rel_logits = einsum('b h i d, h j d -> b h i j', q + self.rel_pos_bias, rel_k)
        rel_logits = relative_shift(rel_logits)

        logits = content_logits + rel_logits
        attn = logits.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# main class
class Stem(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.conv = nn.Conv1d(4, dim, 15, padding = 7)
        self.conv_block_batchnorm_klass = nn.BatchNorm1d(dim)
        self.conv_block_gelu = GELU()
        self.conv_block_conv = nn.Conv1d(dim, dim, 1, padding = 0)
        self.attention_pool = AttentionPool(dim, pool_size = pool_size)
    
    def forward(self, x):
        x = self.conv(x)
        x1 = self.conv_block_batchnorm_klass(x)
        x1 = self.conv_block_gelu(x1)
        x1 = self.conv_block_gelu(x1)
        x1 = self.conv_block_conv(x1)
        x1 = self.conv_block_conv(x1)
        # x = x + x1
        x += x1
        x = self.attention_pool(x)
        return x


def get_model():
    return Enformer()


def get_inputs(batch_size):
    # create batched example data
    inputs = (torch.randint(0, 5, (batch_size, 196_608)),)
    batch_index = [0]
    is_batched = True
    return inputs, batch_index, is_batched


if __name__ == "__main__":
    model = Enformer()
    seq = (torch.randint(0, 5, (8, 196_608)).to('cuda'),) # for ACGTN, in that order (-1 for padding)

    step = 0
    model.to('cuda')
    model.eval()

    with torch.no_grad():
        output = model(seq)
