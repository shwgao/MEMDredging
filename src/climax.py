from functools import lru_cache
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_

EMBEDDING_ONLY = False

class ClimaX(nn.Module):
    """Implements the ClimaX model as described in the paper,
    https://arxiv.org/abs/2301.10343

    Args:
        default_vars (list): list of default variables to be used for training
        img_size (list): image size of the input data
        patch_size (int): patch size of the input data
        embed_dim (int): embedding dimension
        depth (int): number of transformer layers
        decoder_depth (int): number of decoder layers
        num_heads (int): number of attention heads
        mlp_ratio (float): ratio of mlp hidden dimension to embedding dimension
        drop_path (float): stochastic depth rate
        drop_rate (float): dropout rate
        parallel_patch_embed (bool): whether to use parallel patch embedding
    """

    def __init__(
        self,
        default_vars,
        img_size=[32, 64],
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        parallel_patch_embed=False,
    ):
        super().__init__()

        # TODO: remove time_history parameter
        self.img_size = img_size
        self.patch_size = patch_size
        self.default_vars = default_vars
        self.parallel_patch_embed = parallel_patch_embed
        # variable tokenization: separate embedding layer for each input variable
        self.token_embeds = nn.ModuleList(
            [PatchEmbed(img_size, patch_size, 1, embed_dim) for i in range(len(default_vars))]
        )
        self.num_patches = self.token_embeds[0].num_patches

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # positional embedding and lead time embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
        self.lead_time_embed = nn.Linear(1, embed_dim)

        # --------------------------------------------------------------------------

        # ViT backbone
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # --------------------------------------------------------------------------

        # prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, len(self.default_vars) * patch_size**2))
        self.head = nn.Sequential(*self.head)

        # --------------------------------------------------------------------------

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.default_vars), dim), requires_grad=True)
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = len(self.default_vars)
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def aggregate_variables_copy(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.var_agg(var_query, x, x)  # BxL, D # pass need_weights=False to save computation x = self.var_agg(var_query, x, x, need_weights=False)  # BxL, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x
    
    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D
        
        # print(f"x.device: {x.device}")
        # print(f"self.var_query.device: {self.var_query.device}")
        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        
        x, _ = self.var_agg(var_query, x, x)  # BxL, D # pass need_weights=False to save computation x = self.var_agg(var_query, x, x, need_weights=False)  # BxL, D
        
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def forward_encoder(self, x: torch.Tensor, lead_times: torch.Tensor, variables):
        # x: `[B, V, H, W]` shape.

        if isinstance(variables, list):
            variables = tuple(variables)

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)
        
        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables).contiguous() # 1, V, D

        for i in range(len(var_ids)):
            id = var_ids[i]
            embeds.append(self.token_embeds[id](x[:, i : i + 1].contiguous()))  # B, 1, L, D
        x = torch.stack(embeds, dim=1)  # B, V, L, D
        # x = fused_stack_add.forward(embeds, var_embed, 1)  # B, V, L, D
        
        x = x + var_embed.unsqueeze(2)  # B, V, L, D
        # x += var_embed.unsqueeze(2)  # B, V, L, D
        
        # print(f"Memory usage of x: {x.element_size() * x.nelement() / 1024**2} MB")
        # print(f"Allocated memory of x: {torch.cuda.memory_allocated(x.device) / 1024**2} MB")

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D
        
        if not EMBEDDING_ONLY:
            # add pos embedding
            x = x + self.pos_embed
            # x += self.pos_embed

            # add lead time embedding
            lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
            lead_time_emb = lead_time_emb.unsqueeze(1)
            x = x + lead_time_emb  # B, L, D
            # x += lead_time_emb  # B, L, D

            x = self.pos_drop(x)
            
            # x = x.to('cuda:2')

            # apply Transformer blocks
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)

        return x

    def forward(self, x, y, lead_times, variables, out_variables, metric, lat):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        out_transformers = self.forward_encoder(x, lead_times, variables)  # B, L, D

        return out_transformers


class ModelConfigGlobal:
    def __init__(self, 
                 default_vars=None, 
                 img_size=None, 
                 patch_size=2, 
                 embed_dim=1024, 
                 depth=8, 
                 decoder_depth=2, 
                 num_heads=16, 
                 mlp_ratio=4, 
                 drop_path=0.1, 
                 drop_rate=0.1):
        if default_vars is None:
            default_vars = [
                "land_sea_mask", "orography", "lattitude", "2m_temperature",
                "10m_u_component_of_wind", "10m_v_component_of_wind", "geopotential_50",
                "geopotential_250", "geopotential_500", "geopotential_600", "geopotential_700",
                "geopotential_850", "geopotential_925", "u_component_of_wind_50",
                "u_component_of_wind_250", "u_component_of_wind_500", "u_component_of_wind_600",
                "u_component_of_wind_700", "u_component_of_wind_850", "u_component_of_wind_925",
                "v_component_of_wind_50", "v_component_of_wind_250", "v_component_of_wind_500",
                "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850",
                "v_component_of_wind_925", "temperature_50", "temperature_250", "temperature_500",
                "temperature_600", "temperature_700", "temperature_850", "temperature_925",
                "relative_humidity_50", "relative_humidity_250", "relative_humidity_500",
                "relative_humidity_600", "relative_humidity_700", "relative_humidity_850",
                "relative_humidity_925", "specific_humidity_50", "specific_humidity_250",
                "specific_humidity_500", "specific_humidity_600", "specific_humidity_700",
                "specific_humidity_850", "specific_humidity_925"
            ]
        if img_size is None:
            img_size = [32, 64]
        
        self.out_variables = ["geopotential_500", "temperature_850", "2m_temperature", 
                              "10m_u_component_of_wind", "10m_v_component_of_wind"]
        self.predict_range: 72
        
        self.default_vars = default_vars
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path
        self.drop_rate = drop_rate

