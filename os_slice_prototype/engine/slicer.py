"""
OS-Slice Structural Parameter Slicer Module
Applies layer-aware structural masking to attention heads and MLP channels
based on operational Fisher Information saliency, producing a compact OS-Slice Micro-Model.
"""

from typing import Dict, Any, Tuple, Optional
import copy
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from rich.console import Console

console = Console()


class OSSliceModel(nn.Module):
    """
    Encapsulates the base model with applied operational structural masks.
    Enforces that non-salient generalist parameters are excised from execution.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        head_mask: torch.Tensor,
        mlp_mask: torch.Tensor,
        slicing_ratio: float,
    ):
        super().__init__()
        self.model = base_model
        self.config = base_model.config
        self.register_buffer("head_mask", head_mask)  # [L, H]
        self.register_buffer("mlp_mask", mlp_mask)    # [L, intermediate_size]
        self.slicing_ratio = slicing_ratio

        # Calculate exact parameter statistics
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.active_params = self._compute_active_parameters()
        self.sliced_params = self.total_params - self.active_params
        self.effective_sparsity = self.sliced_params / self.total_params

    def _compute_active_parameters(self) -> int:
        """Calculates exact count of non-zero active operational weights."""
        active = 0
        layers = self._get_layers()
        num_layers = len(layers)
        hidden_size = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        head_dim = hidden_size // num_heads
        intermediate_size = self.config.intermediate_size

        for l_idx, layer in enumerate(layers):
            h_active = int(self.head_mask[l_idx].sum().item())
            m_active = int(self.mlp_mask[l_idx].sum().item())

            # Active Q and Out projection weights for active heads
            active += h_active * head_dim * hidden_size * 2
            # K and V shared across KV heads
            active += (self.config.num_key_value_heads * head_dim * hidden_size * 2)
            # Active MLP weights (Gate, Up, Down)
            active += m_active * hidden_size * 3

        # Add embeddings, layer norms, and lm_head
        non_layer_params = self.total_params - (
            num_layers * (
                (hidden_size * hidden_size * 2) + 
                (self.config.num_key_value_heads * head_dim * hidden_size * 2) +
                (intermediate_size * hidden_size * 3)
            )
        )
        return max(active + non_layer_params, 0)

    def _get_layers(self):
        for attr in ["model.layers", "transformer.h", "layers"]:
            curr = self.model
            found = True
            for part in attr.split("."):
                if hasattr(curr, part):
                    curr = getattr(curr, part)
                else:
                    found = False
                    break
            if found:
                return curr
        raise AttributeError("Unable to locate transformer layers in model.")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        kwargs.setdefault("repetition_penalty", 1.05)
        return self.model.generate(*args, **kwargs)


class StructuralSlicer:
    """
    Executes parameter slicing by ranking components by operational Fisher saliency
    and applying structured zeroes across non-salient attention heads and MLP channels.
    """

    def __init__(self, config: Any):
        self.config = config

    def compute_structural_masks(
        self,
        head_saliency: torch.Tensor,
        mlp_saliency: torch.Tensor,
        slicing_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determines layer-aware structural masks:
        - Outer layers (Layer 0, 1, L-2, L-1) are preserved to protect token embedding/unembedding.
        - Intermediate layers (Layer 2 to L-3) are sliced proportionally to operational Fisher saliency.
        """
        num_layers = head_saliency.size(0)
        num_heads = head_saliency.size(1)
        intermediate_size = mlp_saliency.size(1)

        head_mask = torch.ones_like(head_saliency)
        mlp_mask = torch.ones_like(mlp_saliency)

        start_layer = 2
        end_layer = num_layers - 2

        for l in range(start_layer, end_layer):
            # 1. Attention Heads in layer l
            h_scores = head_saliency[l]
            k_h = max(int(num_heads * slicing_ratio), 1)
            pruned_h = torch.topk(h_scores, k_h, largest=False).indices
            head_mask[l, pruned_h] = 0.0

            # 2. MLP Channels in layer l
            m_scores = mlp_saliency[l]
            k_m = int(intermediate_size * slicing_ratio)
            pruned_m = torch.topk(m_scores, k_m, largest=False).indices
            mlp_mask[l, pruned_m] = 0.0

        return head_mask, mlp_mask

    def apply_slicing(
        self,
        model: PreTrainedModel,
        head_saliency: torch.Tensor,
        mlp_saliency: torch.Tensor,
        slicing_ratio: Optional[float] = None,
        in_place: bool = False,
    ) -> OSSliceModel:
        """
        Physically zeroes out non-operational attention heads and MLP channels
        and applies variance-preserving compensation.
        """
        if slicing_ratio is None:
            slicing_ratio = self.config.default_slicing_ratio

        console.print(f"[bold yellow]Applying OS-Slice with {slicing_ratio * 100:.1f}% Parameter Slicing...[/bold yellow]")

        target_model = model if in_place else copy.deepcopy(model)
        target_model.eval()

        head_mask, mlp_mask = self.compute_structural_masks(head_saliency, mlp_saliency, slicing_ratio)

        # Locate layers
        layers = None
        for attr in ["model.layers", "transformer.h", "layers"]:
            curr = target_model
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

        num_heads = target_model.config.num_attention_heads
        hidden_size = target_model.config.hidden_size
        head_dim = hidden_size // num_heads
        intermediate_size = target_model.config.intermediate_size

        with torch.no_grad():
            for l_idx, layer in enumerate(layers):
                # 1. Mask Attention Heads
                self_attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
                if self_attn is not None:
                    h_m = head_mask[l_idx].to(target_model.device)
                    pruned_h = (h_m == 0).nonzero(as_tuple=True)[0]
                    active_h = (h_m == 1).nonzero(as_tuple=True)[0]

                    if len(pruned_h) > 0:
                        for h in pruned_h:
                            q_start, q_end = int(h * head_dim), int((h + 1) * head_dim)
                            self_attn.q_proj.weight.data[q_start:q_end, :] = 0.0
                            if hasattr(self_attn.q_proj, "bias") and self_attn.q_proj.bias is not None:
                                self_attn.q_proj.bias.data[q_start:q_end] = 0.0
                            self_attn.o_proj.weight.data[:, q_start:q_end] = 0.0

                        # Variance-preserving compensation on surviving active heads
                        h_scale = (num_heads / len(active_h)) ** 0.5
                        for h in active_h:
                            q_start, q_end = int(h * head_dim), int((h + 1) * head_dim)
                            self_attn.o_proj.weight.data[:, q_start:q_end] *= h_scale

                # 2. Mask MLP Channels
                mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
                if mlp is not None:
                    m_m = mlp_mask[l_idx].to(target_model.device)
                    pruned_m = (m_m == 0).nonzero(as_tuple=True)[0]
                    active_m = (m_m == 1).nonzero(as_tuple=True)[0]

                    if len(pruned_m) > 0:
                        mlp.gate_proj.weight.data[pruned_m, :] = 0.0
                        mlp.up_proj.weight.data[pruned_m, :] = 0.0
                        mlp.down_proj.weight.data[:, pruned_m] = 0.0

                        # Variance-preserving compensation on surviving active MLP channels
                        m_scale = (intermediate_size / len(active_m)) ** 0.5
                        mlp.down_proj.weight.data[:, active_m] *= m_scale

        sliced_model = OSSliceModel(target_model, head_mask, mlp_mask, slicing_ratio)
        console.print(
            f"[bold green]OS-Slice Slicing Complete:[/bold green] "
            f"Active: {sliced_model.active_params / 1e6:.1f}M / {sliced_model.total_params / 1e6:.1f}M params "
            f"({sliced_model.effective_sparsity * 100:.1f}% reduction)"
        )
        return sliced_model
