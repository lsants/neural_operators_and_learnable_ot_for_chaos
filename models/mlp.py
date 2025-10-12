import torch
from typing import Callable, Optional


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        dropout_rates: Optional[list[float]] = None,
        batch_normalization: Optional[list[bool]] = None,
        layer_normalization: Optional[list[bool]] = None,
     ):
        super(MLP, self).__init__()
        layers = [input_dim] + hidden_layers + [output_dim]
        self.linears = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        n_layers = len(layers) - 1

        dropout_rates = dropout_rates or [0.0] * n_layers
        batch_normalization = batch_normalization or [False] * n_layers
        layer_normalization = layer_normalization or [False] * n_layers

        self.activation = activation

        for i in range(n_layers):
            self.linears.append(
                module=torch.nn.Linear(
                    in_features=layers[i], out_features=layers[i + 1],)
            )
            self.linears.append(
                module=self._build_norm(
                    out_features=layers[i + 1], use_bn=batch_normalization[i], use_ln=layer_normalization[i])
            )
            self.dropouts.append(
                torch.nn.Dropout(
                    dropout_rates[i]) if dropout_rates[i] > 0 else torch.nn.Identity()
            )

    def _build_norm(self, out_features: int, use_bn: bool, use_ln: bool) -> torch.nn.Module:
        if use_bn and use_ln:
            raise ValueError("Choose either batch_norm or layer_norm, not both.")
        if use_bn:
            return torch.nn.BatchNorm1d(out_features)
        if use_ln:
            return torch.nn.LayerNorm(out_features)
        return torch.nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = inputs
        last_idx = len(self.linears) - 1
        for i, (linear, norm, drop) in enumerate(
            zip(self.linears, self.norms, self.dropouts)
        ):
            out = linear(out)
            if i != last_idx:
                out = norm(out)
                out = self.activation(out)
                out = drop(out)
        return out