"""
OS-Slice Operational Profiler Module
Computes empirical Fisher Information Matrix (FIM) and gradient saliency across
attention heads and MLP channels conditioned on operational boundary tokens.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from rich.progress import track
from rich.console import Console

console = Console()


class OperationalProfiler:
    """
    Measures the structural saliency of attention heads and MLP channels
    under the operational loss:
        L_op = sum_{t in T_op} w_t * CrossEntropy(logits_t, targets_t)
    
    Computes Fisher Information saliency:
        S_G = sum_{w in G} (grad_w L_op)^2 * w^2 * ||P_op X_in||_2
    """

    def __init__(self, model: PreTrainedModel, config: Any):
        self.model = model
        self.config = config
        self.device = config.device
        
        # Extract model architectural dimensions
        model_cfg = model.config
        self.num_layers = getattr(model_cfg, "num_hidden_layers", 24)
        self.num_heads = getattr(model_cfg, "num_attention_heads", 14)
        self.num_kv_heads = getattr(model_cfg, "num_key_value_heads", self.num_heads)
        self.hidden_size = getattr(model_cfg, "hidden_size", 896)
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = getattr(model_cfg, "intermediate_size", 4864)
        
        # Profile storage
        self.head_saliency: Optional[torch.Tensor] = None   # Shape: [num_layers, num_heads]
        self.mlp_saliency: Optional[torch.Tensor] = None    # Shape: [num_layers, intermediate_size]
        self.profile_metadata: Dict[str, Any] = {}

    def compute_operational_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Computes weighted cross-entropy focused on operational boundary tokens."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        op_weights = batch["operational_weights"].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # Shift tokens for autoregressive next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = op_weights[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        unreduced_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.size())

        # Mask positions where label is -100
        valid_mask = (shift_labels != -100).float()
        combined_weights = shift_weights * valid_mask

        total_weight = combined_weights.sum()
        if total_weight > 0:
            loss = (unreduced_loss * combined_weights).sum() / total_weight
        else:
            loss = (unreduced_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        return loss

    def profile(
        self,
        calibration_dataset: List[Dict[str, torch.Tensor]],
        save_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Executes gradient accumulation over calibration dataset and aggregates
        empirical Fisher Information across all attention heads and MLP channels.
        """
        console.print("[bold cyan]Executing OS-Slice Operational Fisher Saliency Profiling...[/bold cyan]")
        self.model.eval()
        self.model.zero_grad()

        # Initialize accumulators
        head_fisher = torch.zeros((self.num_layers, self.num_heads), dtype=torch.float32)
        mlp_fisher = torch.zeros((self.num_layers, self.intermediate_size), dtype=torch.float32)

        # Locate transformer layers in Qwen / LLaMA architectures
        layers = None
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
                layers = curr
                break

        if layers is None:
            raise AttributeError("Unable to locate transformer layers in model hierarchy.")

        num_samples = len(calibration_dataset)

        for idx in track(range(num_samples), description="Profiling Operational Boundaries"):
            sample = calibration_dataset[idx]
            # Add batch dimension
            batch = {
                k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                for k, v in sample.items()
            }

            self.model.zero_grad()
            loss = self.compute_operational_loss(batch)
            loss.backward()

            # Aggregate Fisher Saliency for each layer
            with torch.no_grad():
                for l_idx, layer in enumerate(layers):
                    # 1. Attention Heads: q_proj, k_proj, v_proj, o_proj
                    self_attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
                    if self_attn is not None:
                        q_grad = getattr(self_attn.q_proj, "weight", None)
                        o_grad = getattr(self_attn.o_proj, "weight", None)
                        
                        if q_grad is not None and q_grad.grad is not None:
                            q_w = q_grad.data.float()
                            q_g = q_grad.grad.data.float()
                            # Fisher diagonal: g^2 * w^2
                            q_head_fisher = (q_g ** 2 * q_w ** 2).view(self.num_heads, self.head_dim, -1).sum(dim=(1, 2))
                        else:
                            q_head_fisher = torch.zeros(self.num_heads)

                        if o_grad is not None and o_grad.grad is not None:
                            o_w = o_grad.data.float()
                            o_g = o_grad.grad.data.float()
                            o_head_fisher = (o_g ** 2 * o_w ** 2).view(-1, self.num_heads, self.head_dim).sum(dim=(0, 2))
                        else:
                            o_head_fisher = torch.zeros(self.num_heads)

                        layer_head_fisher = (q_head_fisher + o_head_fisher).cpu()
                        head_fisher[l_idx] += layer_head_fisher / num_samples

                    # 2. MLP Channels: gate_proj, up_proj, down_proj
                    mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
                    if mlp is not None:
                        gate_w = getattr(mlp.gate_proj, "weight", None)
                        down_w = getattr(mlp.down_proj, "weight", None)

                        mlp_layer_score = torch.zeros(self.intermediate_size)

                        if gate_w is not None and gate_w.grad is not None:
                            gw = gate_w.data.float()
                            gg = gate_w.grad.data.float()
                            gate_score = (gg ** 2 * gw ** 2).sum(dim=1).cpu()
                            mlp_layer_score += gate_score

                        if down_w is not None and down_w.grad is not None:
                            dw = down_w.data.float()
                            dg = down_w.grad.data.float()
                            down_score = (dg ** 2 * dw ** 2).sum(dim=0).cpu()
                            mlp_layer_score += down_score

                        mlp_fisher[l_idx] += mlp_layer_score / num_samples

        self.model.zero_grad()

        # Normalize and store
        self.head_saliency = head_fisher
        self.mlp_saliency = mlp_fisher

        profile_data = {
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "head_saliency_mean": float(head_fisher.mean().item()),
            "head_saliency_max": float(head_fisher.max().item()),
            "mlp_saliency_mean": float(mlp_fisher.mean().item()),
            "mlp_saliency_max": float(mlp_fisher.max().item()),
            "layer_head_importance": head_fisher.mean(dim=1).tolist(),
            "layer_mlp_importance": mlp_fisher.mean(dim=1).tolist(),
        }
        self.profile_metadata = profile_data

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "head_saliency": self.head_saliency,
                "mlp_saliency": self.mlp_saliency,
                "metadata": profile_data
            }, save_path)
            console.print(f"[green]Profile successfully persisted to {save_path}[/green]")

        return profile_data

    def load_profile(self, profile_path: Path):
        """Loads precomputed saliency profile from disk."""
        data = torch.load(profile_path, map_location="cpu", weights_only=False)
        self.head_saliency = data["head_saliency"]
        self.mlp_saliency = data["mlp_saliency"]
        self.profile_metadata = data.get("metadata", {})
        console.print(f"[green]Loaded precomputed profile from {profile_path}[/green]")
