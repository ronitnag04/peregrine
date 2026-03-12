import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Iterable

class BilinearInteraction(nn.Module):
    """
    Replaces nn.Bilinear with XLA-compatible equivalent.
    Equivalent to: out = prog @ W @ config^T (per output unit)
    but decomposed into two linear projections + hadamard,
    which XLA can handle without 3D weight tensors.
    """
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj_prog   = nn.Linear(in_dim, out_dim, bias=False)
        self.proj_config = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, prog: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        # Equivalent to bilinear without 3D weight tensor
        return self.proj_prog(prog) * self.proj_config(config) + self.bias


class PeregrineMLModel(nn.Module):
    def __init__(
        self,
        prog_size: int,
        config_size: int,
        prog_embed_dim: int = 128,
        config_embed_dim: int = 64,
        interaction_dim: int = 128,
        hidden_dims: Iterable[int] = [256, 128, 64],
        output_size: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Separately embed program and config into a common interaction space
        self.prog_encoder = nn.Sequential(
            nn.Linear(prog_size, prog_embed_dim),
            nn.LayerNorm(prog_embed_dim),
            nn.ReLU(),
            nn.Linear(prog_embed_dim, interaction_dim),
        )
        self.config_encoder = nn.Sequential(
            nn.Linear(config_size, config_embed_dim),
            nn.LayerNorm(config_embed_dim),
            nn.ReLU(),
            nn.Linear(config_embed_dim, interaction_dim),
        )

        # Bilinear interaction: captures prog_i * config_j cross terms
        # out[k] = prog_embed^T W_k config_embed, for k in interaction_dim
        self.bilinear = BilinearInteraction(interaction_dim, interaction_dim)

        # Head receives: prog_embed || config_embed || bilinear_interaction
        head_input_dim = interaction_dim + interaction_dim + interaction_dim
        layers = []
        in_dim = head_input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_size))
        self.head = nn.Sequential(*layers)

    def forward(self, prog: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        prog_embed = self.prog_encoder(prog)      # [B, interaction_dim]
        config_embed = self.config_encoder(config) # [B, interaction_dim]

        # Element-wise product: cheapest explicit interaction, surprisingly effective
        hadamard = prog_embed * config_embed       # [B, interaction_dim]

        # Bilinear: richer pairwise interactions across all (i,j) pairs
        bilinear_out = self.bilinear(prog_embed, config_embed)  # [B, interaction_dim]

        # Combine all three views
        combined = torch.cat([hadamard, bilinear_out, prog_embed + config_embed], dim=-1)
        return self.head(combined)