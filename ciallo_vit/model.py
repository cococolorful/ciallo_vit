import torch
from torch import optim, nn, Tensor
import lightning as L
from jaxtyping import Float32

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels:int = 3, out_channels:int = 768, kernel_size:int=16, flatten:bool=True):
        super().__init__()

        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=kernel_size)
        self.flatten = flatten

    def forward(self, x:Float32[Tensor, "B C H W"]) -> Float32[Tensor, "B H*W//kernel_size**2 out_channels"]:
        assert x.shape[2] % self.proj.kernel_size[0] == 0 and x.shape[3] % self.proj.kernel_size[1] == 0, "Input image size must be divisible by the kernel size"
        assert x.shape[1] == self.proj.in_channels, "Input image must have the same number of channels as the number of input channels of the convolutional layer"

        x = self.proj(x) # (B, out_channels, H//kernel_size, W//kernel_size)
        if self.flatten:
            x = x.flatten(2) # (B, out_channels, H*W//kernel_size**2)
            x = x.transpose(1, 2) # (B, H*W//kernel_size**2, out_channels)
        return x.contiguous() # (B, H*W//kernel_size**2, out_channels)

class PositionEmbedding(nn.Module):
    def __init__(self, num_patches:int, embedding_dim:int):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, embedding_dim)) # TODO: Use cosine/sine position embedding
    
    def forward(self, x:Float32[Tensor, "B H*W//kernel_size**2 out_channels"]) -> Float32[Tensor, "B H*W//kernel_size**2 out_channels"]:
        return x + self.position_embedding

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim:int, mlp_hidding_dim:int, num_heads:int = 8, drop_rate:float=0.0):
        super().__init__()

        self.to_qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(nn.Linear(embedding_dim, mlp_hidding_dim), nn.ReLU(), nn.Linear(mlp_hidding_dim, embedding_dim))
        self.dropout = nn.Dropout(p=drop_rate) if drop_rate > 0.0 else nn.Identity()
        # self.example_input_array = torch.rand(1, 28 * 28 // 16**2, embedding_dim) # TODO: Use a real example input array
    
    def forward(self, x:Float32[Tensor, "B N hidding_dim"]) -> Float32[Tensor, "B N hidding_dim"]:
        q, k, v = self.to_qkv(x).chunk(3, dim=-1) # (B, N, embedding_dim)
        x = x + self.dropout(self.attention(q, k, v)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.feed_forward(x))
        x = self.norm2(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, image_size:int, patch_size:int, embedding_dim:int, drop_rate:float=0.0, num_blocks:int=12):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels=3, out_channels=embedding_dim, kernel_size=patch_size, flatten=True)

        self.position_embedding = PositionEmbedding(num_patches=(image_size//patch_size)**2, embedding_dim=embedding_dim)
        
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0.0 else nn.Identity()

        self.transformer_blocks = nn.Sequential(*[TransformerBlock(embedding_dim=embedding_dim, mlp_hidding_dim=embedding_dim*4, drop_rate=drop_rate) for _ in range(num_blocks)])

        self.N = (image_size//patch_size)**2
    def forward(self, images:Float32[Tensor, "B C H W"]):
        x = self.patch_embedding(images) # (B, H*W//kernel_size**2, embedding_dim)
        x = self.position_embedding(x) # (B, H*W//kernel_size**2, embedding_dim)
        
        x = self.pos_drop(x) # (B, H*W//kernel_size**2, embedding_dim)
        x = self.transformer_blocks(x) # (B, H*W//kernel_size**2, embedding_dim)
        return x # (B, H*W//kernel_size**2, embedding_dim)

class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.vision_transformer = VisionTransformer(image_size=256, patch_size=16, embedding_dim=768, drop_rate=0.1, num_blocks=4)
        self.decoder1 = nn.Linear(768, 1)
        
        self.decoder2 = nn.Linear(self.vision_transformer.N, 1)
        self.example_input_array = torch.rand(1, 3, 256, 256) # TODO: Use a real example input array
        self.criterion = nn.MSELoss()
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = torch.stack([x for _ in range(3)], dim=1).squeeze(2)

        x = self.forward(x)
        loss = self.criterion(x, y.float())
        self.log("train_loss", loss)
        return loss

    def forward(self, images: Float32[Tensor, "B C H W"]):
        x = self.vision_transformer(images)
        x = self.decoder1(x)
        x = torch.nn.functional.relu(x).squeeze(-1)
        x = self.decoder2(x)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
if __name__ == "__main__":
    model = LitAutoEncoder()
    model(model.example_input_array)
    model.to_onnx("model.onnx", input_names=["input"], output_names=["output"])
    print(model)