import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from src.vit import ViT


class SimMIM(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_pixels = nn.Linear(encoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # for indexing purposes

        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions

        pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb

        # prepare mask tokens

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens)

        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_pixel_values = self.to_pixels(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # calculate reconstruction loss

        recon_loss = F.l1_loss(pred_pixel_values, masked_patches) / num_masked
        return recon_loss

def get_model():
    # return model
    
    v = ViT(
        image_size = 2048,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048
    )

    model = SimMIM(
        encoder = v,
        masking_ratio = 0.5  # they found 50% to yield the best results
    )
    return model

def get_inputs(batch_size):
    # create example data
    inputs = torch.randn(batch_size, 3, 2048, 2048, dtype=torch.float32)
    batch_index = [0]
    is_batched = True
    return (inputs,), batch_index, is_batched

if __name__ == "__main__":
    import torch
    

    v = ViT(
        image_size = 2048,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048
    )

    mim = SimMIM(
        encoder = v,
        masking_ratio = 0.5  # they found 50% to yield the best results
    )

    mim = mim.to('cuda')

    images = torch.randn(2, 3, 2048, 2048).to('cuda')

    loss = mim(images)
    loss.backward()
    print(torch.cuda.max_memory_allocated()/1024**3)
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    print(loss)

    # that's all!
    # do the above in a for loop many times with a lot of images and your vision transformer will learn

