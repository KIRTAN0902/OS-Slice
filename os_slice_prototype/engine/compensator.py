"""
OS-Slice Localized Error Compensation Module
Executes diagonal ridge-inspired damping on surviving active operational weights
after structured zero-masking.
Note: Full covariance-based OBS inversion is not implemented in this module;
the implemented method is diagonal ridge-inspired damping on down_proj and o_proj.
"""

from typing import List, Dict, Any, Optional
import torch
from rich.console import Console

console = Console()


class OperationalCompensator:
    """
    Applies layer-wise diagonal ridge-inspired damping to active MLP down_proj
    and attention o_proj weights:
        damping = 1.0 + ridge_lambda * (pruned_count / total_dim)
        W_active <- W_active / damping

    Guarantees:
    - Sliced (zeroed) parameters remain strictly zero.
    - Protected outer layers (0, 1, L-2, L-1) remain untouched.
    - Returns exact before/after L2 norm deltas for every modified layer.
    """

    def __init__(self, config: Any):
        self.config = config
        self.device = config.device

    def compensate(
        self,
        sliced_model_wrapper: Any,
        calibration_dataset: Optional[List[Dict[str, torch.Tensor]]] = None,
        steps: Optional[int] = None,
        ridge_lambda: float = 1e-3,
    ) -> Dict[str, Any]:
        """
        Executes fast, layer-wise diagonal ridge-inspired damping on active
        MLP down_proj and attention o_proj weights.
        """
        console.print("[bold cyan]Executing OS-Slice Diagonal Ridge-Inspired Compensation...[/bold cyan]")
        model = sliced_model_wrapper.model
        head_mask = sliced_model_wrapper.head_mask.to(self.device)
        mlp_mask = sliced_model_wrapper.mlp_mask.to(self.device)

        layers = None
        for attr in ["model.layers", "transformer.h", "layers"]:
            curr = model
            found = True
            for part in attr.split("."):
                if hasattr(curr, part):
                    curr = getattr(curr, part)
                else:
                    found = False
                    break
            if found:
                layers = curr
                break

        if layers is None:
            raise AttributeError("Unable to locate transformer layers in model.")

        num_layers = len(layers)
        num_heads = model.config.num_attention_heads
        hidden_size = model.config.hidden_size
        head_dim = hidden_size // num_heads
        intermediate_size = model.config.intermediate_size

        compensated_layers = 0
        layer_reports: List[Dict[str, Any]] = []

        with torch.no_grad():
            for l in range(2, num_layers - 2):
                layer_info: Dict[str, Any] = {"layer": l}

                mlp = getattr(layers[l], "mlp", None) or getattr(layers[l], "feed_forward", None)
                m_m = mlp_mask[l]
                pruned_m = (m_m == 0).nonzero(as_tuple=True)[0]
                active_m = (m_m == 1).nonzero(as_tuple=True)[0]

                if mlp is not None and len(pruned_m) > 0 and len(active_m) > 0:
                    damping_m = 1.0 + (ridge_lambda * (len(pruned_m) / intermediate_size))
                    scale_m = 1.0 / damping_m
                    before_norm_m = float(mlp.down_proj.weight.data[:, active_m].float().norm().item())
                    mlp.down_proj.weight.data[:, active_m] *= scale_m
                    after_norm_m = float(mlp.down_proj.weight.data[:, active_m].float().norm().item())
                    compensated_layers += 1
                    layer_info["mlp_active_channels"] = int(len(active_m))
                    layer_info["mlp_pruned_channels"] = int(len(pruned_m))
                    layer_info["mlp_damping_factor"] = round(damping_m, 8)
                    layer_info["mlp_scale_applied"] = round(scale_m, 8)
                    layer_info["mlp_down_proj_norm_before"] = round(before_norm_m, 6)
                    layer_info["mlp_down_proj_norm_after"] = round(after_norm_m, 6)
                    layer_info["mlp_down_proj_norm_delta"] = round(after_norm_m - before_norm_m, 6)

                self_attn = getattr(layers[l], "self_attn", None) or getattr(layers[l], "attention", None)
                h_m = head_mask[l]
                pruned_h = (h_m == 0).nonzero(as_tuple=True)[0]
                active_h = (h_m == 1).nonzero(as_tuple=True)[0]

                if self_attn is not None and len(pruned_h) > 0 and len(active_h) > 0:
                    damping_h = 1.0 + (ridge_lambda * (len(pruned_h) / num_heads))
                    scale_h = 1.0 / damping_h
                    active_cols = torch.cat([
                        torch.arange(int(h * head_dim), int((h + 1) * head_dim), device=self_attn.o_proj.weight.device)
                        for h in active_h
                    ])
                    before_norm_h = float(self_attn.o_proj.weight.data[:, active_cols].float().norm().item())
                    self_attn.o_proj.weight.data[:, active_cols] *= scale_h
                    after_norm_h = float(self_attn.o_proj.weight.data[:, active_cols].float().norm().item())
                    layer_info["attn_active_heads"] = int(len(active_h))
                    layer_info["attn_pruned_heads"] = int(len(pruned_h))
                    layer_info["attn_damping_factor"] = round(damping_h, 8)
                    layer_info["attn_scale_applied"] = round(scale_h, 8)
                    layer_info["attn_o_proj_norm_before"] = round(before_norm_h, 6)
                    layer_info["attn_o_proj_norm_after"] = round(after_norm_h, 6)
                    layer_info["attn_o_proj_norm_delta"] = round(after_norm_h - before_norm_h, 6)

                layer_reports.append(layer_info)

        console.print(
            f"[bold green]Compensation Complete:[/bold green] Applied diagonal ridge-inspired damping "
            f"across {compensated_layers} intermediate layers."
        )

        return {
            "method": "diagonal_ridge_damping",
            "method_label": "Diagonal Ridge-Inspired Damping (full covariance OBS not implemented)",
            "ridge_lambda": ridge_lambda,
            "protected_layers": [0, 1, num_layers - 2, num_layers - 1] if num_layers >= 4 else [],
            "compensated_layers": compensated_layers,
            "layer_reports": layer_reports,
            "status": "Diagonal Ridge-Inspired Damping Applied",
        }
