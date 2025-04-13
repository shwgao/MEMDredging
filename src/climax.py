from functools import lru_cache
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_
from torch.utils.checkpoint import checkpoint_sequential, checkpoint

EMBEDDING_ONLY = False


def lat_weighted_mse(pred, y, vars, lat, mask=None):
    """Latitude weighted mean squared error

    Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = (pred - y) ** 2  # [N, C, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (error[:, i] * w_lat * mask).sum() / mask.sum()
            else:
                loss_dict[var] = (error[:, i] * w_lat).mean()

    if mask is not None:
        loss_dict["loss"] = ((error * w_lat.unsqueeze(1)).mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = (error * w_lat.unsqueeze(1)).mean(dim=1).mean()

    return loss_dict

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
        drop_path=0.,  # TODO: I removed drop_path
        drop_rate=0.,  # TODO: I removed drop_rate
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
        
        self.out_variables = ["geopotential_500", "temperature_850", "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
        self.lat = np.random.rand(32)

        # --------------------------------------------------------------------------

        # ViT backbone
        self.pos_drop = nn.Dropout(p=0.)
        dpr = [x.item() for x in torch.linspace(0, 0., depth)]  # stochastic depth decay rule
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
        
        self.batch_cat_aggregate = False
        self.batch_aggregate = False
        self.mini_batch = 8
        self.checkpointing = False
        
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
        
        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        
        x, _ = self.var_agg(var_query, x, x, need_weights=False)  # BxL, D # pass need_weights=False to save computation x = self.var_agg(var_query, x, x, need_weights=False)  # BxL, D
        
        x = x.squeeze()
        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x
    
    def batch_aggregate_variables(self, x: torch.Tensor, batch_size: int = 4):
        """
        Batch processing version of aggregate_variables to reduce memory usage
        x: B, V, L, D
        """
        total_batch = x.shape[0]
        num_batches = (total_batch + batch_size - 1) // batch_size
        output_list = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_batch)
            batch_x = x[start_idx:end_idx]
            
            # Process small batch
            batch_output = self.aggregate_variables(batch_x)
            output_list.append(batch_output)
            
        # Concatenate all batches
        return torch.cat(output_list, dim=0)  # B, L, D
    
    def batched_cat_aggregate(self, x, var_embed, batch_size=8):
        total_batch = len(x)
        num_batches = (total_batch + batch_size - 1) // batch_size
        output_list = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_batch)
            batch_x = [x[i][start_idx:end_idx] for i in range(len(x))]
            
            # batch_embed = var_embed[:, start_idx:end_idx, :]
            
            batch_x = torch.stack(batch_x, dim=1)
            batch_x += var_embed.unsqueeze(2)
            
            # Process small batch
            batch_output = self.aggregate_variables(batch_x)
            output_list.append(batch_output)
            
        # Concatenate all batches
        return torch.cat(output_list, dim=0)  # B, L, D

    def forward_encoder2(self, x: torch.Tensor, lead_times: torch.Tensor, variables):
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
        
        x = x + var_embed.unsqueeze(2)  # B, V, L, D
        # x += var_embed.unsqueeze(2)  # B, V, L, D

        # Replace original aggregation with batched version
        x = self.batch_aggregate_variables(x, batch_size=8)  # B, L, D
        
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

    def forward_encoder(self, x: torch.Tensor, lead_times: torch.Tensor, variables):
        # x: `[B, V, H, W]` shape.
        # default L, D = 512, 1024

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
        
        if not self.batch_aggregate:
            x = torch.stack(embeds, dim=1)  # B, V, L, D
            # x = fused_stack_add.forward(embeds, var_embed, 1)  # B, V, L, D
            
            x = x + var_embed.unsqueeze(2)  # B, V, L, D
            # x += var_embed.unsqueeze(2)  # B, V, L, D
            
            # # variable aggregation
            # if self.batch_aggregate:
            #     x = self.batch_aggregate_variables(x, batch_size=self.mini_batch)  # B, L, D
            # else:
            # x = self.aggregate_variables(x)  # B, L, D
            if self.checkpointing:
                x = checkpoint(self.aggregate_variables, x, use_reentrant=True)
            else:
                x = self.aggregate_variables(x)  # B, L, D
            # x = checkpoint(self.batched_cat_aggregate, embeds, var_embed, 32, use_reentrant=True)
        else:
            x = self.batched_cat_aggregate(embeds, var_embed, batch_size=self.mini_batch)  # B, L, D
        
        if not EMBEDDING_ONLY:
            # x = torch.randn(32, 512, 1024, device='cuda:0')
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
        if self.checkpointing:
            x = checkpoint_sequential(nn.Sequential(*self.blocks), len(self.blocks), x, use_reentrant=True)
        else:
            for blk in self.blocks:
                x = blk(x)
        x = self.norm(x)

        return x

    # def forward(self, x, y, lead_times, variables, out_variables, metric, lat):
    # for simplicity, we remove the irrelevant arguments
    def forward(self, inputs, metrics=[lat_weighted_mse]):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        x, lead_times, variables, y = inputs
        
        out_transformers = self.forward_encoder(x, lead_times, variables)  # B, L, D
        
        preds = self.head(out_transformers)  # B, L, V*p*p

        preds = self.unpatchify(preds)
        out_var_ids = self.get_var_ids(tuple(self.out_variables), preds.device)
        preds = preds[:, out_var_ids]
        
        loss = 0.0  

        if self.training:
            if metrics is None:
                loss = None
            else:
                loss = [m(preds, y, self.out_variables, self.lat) for m in metrics]

            return loss[0]['loss']
        else:
            return preds


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


model_config = ModelConfigGlobal()

def get_model():
    # return model
    model = ClimaX(
        default_vars=model_config.default_vars,
        img_size=model_config.img_size,
        patch_size=model_config.patch_size,
        embed_dim=model_config.embed_dim,
        depth=model_config.depth,
        decoder_depth=model_config.decoder_depth,
        num_heads=model_config.num_heads,
        mlp_ratio=model_config.mlp_ratio,
        drop_path=model_config.drop_path,
        drop_rate=model_config.drop_rate,
    )
    return model

def get_inputs(batch_size):
    # create example data
    x = torch.randn(batch_size, 48, 32, 64, dtype=torch.float32)
    y = torch.randn(batch_size, 5, 32, 64, dtype=torch.float32)
    # x = torch.randn(batch_size, 48, 512, 1024, dtype=torch.float32)
    lead_times = torch.tensor([72]*batch_size, dtype=torch.float32)
    variables = model_config.default_vars
    # out_variables = model_config.out_variables
    # inputs = (x, None, lead_times, variables, out_variables, None, None)
    inputs = (x, lead_times, variables, y)
    batch_index = [0, 1]
    is_batched = True
    return inputs, batch_index, is_batched


if __name__ == "__main__":
    # test
    model = get_model()
    inputs, batch_index, is_batched = get_inputs(3)
    data_loader = [i.to('cuda:0') if hasattr(i, "to") else i for i in inputs]
    model = model.to('cuda:0')
    out_transformers = model(*data_loader)
    out_transformers2 = model(*data_loader)
    print(out_transformers)
    print(out_transformers2)
    print(torch.allclose(out_transformers, out_transformers2, atol=1e-8, rtol=1e-8))
    print(torch.max(torch.abs(out_transformers - out_transformers2)))
