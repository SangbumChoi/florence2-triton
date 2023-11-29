from torch import nn


class LinearProjection(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        """Minor change can be applied into linear projection layer such as disable bias."""
        self.linear_projection = nn.Linear(in_features, hidden_features, bias=False)
        self.norm1 = nn.LayerNorm(hidden_features)

    def forward(self, x):
        x = self.linear_projection(x)
        x = self.norm1(x)
        return x
