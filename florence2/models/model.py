import torch
import torch.nn as nn
from florence2.models.backbone import davit


class Florence2(nn.Module):
    r"""Florence 2 model

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dims (tuple(int)): Patch embedding dimension. Default: (64, 128, 192, 256)
        num_heads (tuple(int)): Number of attention heads in different layers. Default: (4, 8, 12, 16)
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        attention_types (tuple(str)): Dual attention types.
        ffn (bool): If False, pure attention network without FFNs
        overlapped_patch (bool): If True, use overlapped patch division during patch merging.
    """

    def __init__(self, backbone_name):
        super().__init__()

        if backbone_name == "base":
            self.backbone = davit.DaViT_base(pretrained=False, return_feature=True)
        elif backbone_name == "large":
            self.backbone = davit.DaViT_large_window12_384(
                pretrained=False, return_feature=True
            )
        else:
            assert False, f"backbone_name is {backbone_name} which is not defined"

    def forward(self, x):
        x = self.backbone(x)
        return x


if __name__ == "__main__":
    florence_model = Florence2("base")
    x = torch.rand([1, 3, 224, 224])
    y = florence_model(x)
    print("output", y.shape)
    print("input", x.shape)
