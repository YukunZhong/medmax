"""
MoDE Adapters: V-Adapter (LoRA) and T-MoE (MoE-LoRA)

Per-token modality split in every MLP linear layer:
  - Image tokens (vocab id in [4, 8195])  → V-Adapter (single LoRA)
  - Text tokens (everything else)          → T-MoE (dense soft-routing MoE-LoRA)

Backbone (ChameleonMLP) is fully frozen.  Only adapter params are trained.

Paper refs: Fig 3, Eq (2)(5)(6)(7)(8)(10)(11)(12).
"""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Chameleon image token vocab range (IMGIMG tokens): [4, 8195]
IMAGE_TOKEN_MIN: int = 4
IMAGE_TOKEN_MAX: int = 8195


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def build_image_mask(input_ids: torch.LongTensor) -> torch.BoolTensor:
    """Return bool tensor [B, S], True where token id is an image token."""
    return (input_ids >= IMAGE_TOKEN_MIN) & (input_ids <= IMAGE_TOKEN_MAX)


# ---------------------------------------------------------------------------
# Base LoRA building block
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Single LoRA adapter: Δ = B · A(x) · (alpha / rank)

    A is initialised with Kaiming uniform; B is zero-initialised so the
    adapter starts as an identity delta.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scale


# ---------------------------------------------------------------------------
# T-MoE: dense soft-routing MoE-LoRA for text tokens
# ---------------------------------------------------------------------------

class TMoE(nn.Module):
    """Text-side MoE-LoRA with dense softmax routing.

    Paper Eq (6)-(8): router computes g_j = softmax(x W_g)_j over all E
    experts.  Each expert is a LoRA; outputs are weighted-summed.
    No top-k masking and no load-balancing auxiliary loss (per the paper's
    actual formulae — equations do not include a sparsity term).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        num_experts: int = 4,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            LoRALinear(in_features, out_features, rank, alpha)
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(in_features, num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, in_features]
        gates = F.softmax(self.router(x), dim=-1)               # [N, E]
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # [N, E, out]
        return (gates.unsqueeze(-1) * expert_outs).sum(dim=1)   # [N, out]


# ---------------------------------------------------------------------------
# MoDEChameleonMLP: drop-in replacement for ChameleonMLP
# ---------------------------------------------------------------------------

class MoDEChameleonMLP(nn.Module):
    """Replaces a ChameleonMLP layer with modality-split adapters.

    For each of gate_proj / up_proj / down_proj the forward is:
        h = base_linear(x) + adapter_delta(x)

    where adapter_delta routes image tokens through the V-Adapter (LoRA)
    and text tokens through T-MoE.  In KD mode (kd_mode=True) the T-MoE
    branch is skipped entirely so that no gradient from the KD loss flows
    into T-MoE parameters.

    The modality mask and kd_mode flag are read from ``mode_context``, a
    plain dict that MoDEModel writes before every forward call:
        mode_context['image_mask']  : BoolTensor [B, S]
        mode_context['kd_mode']     : bool (default False)
    """

    def __init__(
        self,
        original_mlp: nn.Module,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        num_experts: int = 4,
        mode_context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        hidden: int = original_mlp.hidden_size
        intermediate: int = original_mlp.intermediate_size

        # Freeze backbone
        self.base = original_mlp
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.act_fn = original_mlp.act_fn

        # V-Adapters (image tokens)
        self.va_gate = LoRALinear(hidden, intermediate, lora_rank, lora_alpha)
        self.va_up   = LoRALinear(hidden, intermediate, lora_rank, lora_alpha)
        self.va_down = LoRALinear(intermediate, hidden, lora_rank, lora_alpha)

        # T-MoE (text tokens)
        self.tm_gate = TMoE(hidden, intermediate, lora_rank, lora_alpha, num_experts)
        self.tm_up   = TMoE(hidden, intermediate, lora_rank, lora_alpha, num_experts)
        self.tm_down = TMoE(intermediate, hidden, lora_rank, lora_alpha, num_experts)

        self.mode_context: Dict[str, Any] = mode_context if mode_context is not None else {}

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _compute_delta(
        self,
        x_flat: torch.Tensor,         # [N, in_features]
        vadapter: LoRALinear,
        tmoe: TMoE,
        mask_flat: Optional[torch.BoolTensor],  # [N]
        kd_mode: bool,
        out_features: int,
    ) -> torch.Tensor:
        """Compute adapter delta for one linear projection.

        Returns tensor of shape [N, out_features].
        """
        N = x_flat.shape[0]
        delta = torch.zeros(N, out_features, dtype=x_flat.dtype, device=x_flat.device)

        if mask_flat is not None:
            img_idx = mask_flat.nonzero(as_tuple=True)[0]
            if img_idx.numel() > 0:
                delta[img_idx] = vadapter(x_flat[img_idx])

            if not kd_mode:
                txt_idx = (~mask_flat).nonzero(as_tuple=True)[0]
                if txt_idx.numel() > 0:
                    delta[txt_idx] = tmoe(x_flat[txt_idx])
        else:
            # No mask available: treat all tokens as text (fallback)
            if not kd_mode:
                delta = tmoe(x_flat)

        return delta

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image_mask: Optional[torch.BoolTensor] = self.mode_context.get("image_mask")
        kd_mode: bool = self.mode_context.get("kd_mode", False)

        B, S, H = x.shape
        x_flat = x.view(B * S, H)

        if image_mask is not None:
            # image_mask may come from a different device; move to match x
            mask_flat = image_mask.to(x.device).view(B * S)
        else:
            mask_flat = None

        intermediate_size = self.base.intermediate_size

        # --- gate_proj + adapter ---
        gate_base  = self.base.gate_proj(x)                           # [B,S,I]
        gate_delta = self._compute_delta(
            x_flat, self.va_gate, self.tm_gate, mask_flat, kd_mode, intermediate_size
        ).view(B, S, intermediate_size)
        gate_out = gate_base + gate_delta

        # --- up_proj + adapter ---
        up_base  = self.base.up_proj(x)
        up_delta = self._compute_delta(
            x_flat, self.va_up, self.tm_up, mask_flat, kd_mode, intermediate_size
        ).view(B, S, intermediate_size)
        up_out = up_base + up_delta

        # --- gated activation (SiLU/GELU) ---
        intermediate = self.act_fn(gate_out) * up_out                 # [B,S,I]

        # --- down_proj + adapter ---
        down_base  = self.base.down_proj(intermediate)                # [B,S,H]
        inter_flat = intermediate.view(B * S, intermediate_size)
        down_delta = self._compute_delta(
            inter_flat, self.va_down, self.tm_down, mask_flat, kd_mode, H
        ).view(B, S, H)
        out = down_base + down_delta

        return out

    # ------------------------------------------------------------------
    # Parameter group helpers
    # ------------------------------------------------------------------

    def vadapter_parameters(self):
        """Yield V-Adapter (image-side) parameters."""
        for m in (self.va_gate, self.va_up, self.va_down):
            yield from m.parameters()

    def tmoe_parameters(self):
        """Yield T-MoE (text-side) parameters."""
        for m in (self.tm_gate, self.tm_up, self.tm_down):
            yield from m.parameters()


# ---------------------------------------------------------------------------
# Model-level injection
# ---------------------------------------------------------------------------

def insert_mode_adapters(
    model: nn.Module,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    num_experts: int = 4,
) -> Dict[str, Any]:
    """Replace every ChameleonMLP in the model with MoDEChameleonMLP.

    Returns a shared ``mode_context`` dict that callers must populate with
    ``image_mask`` (and optionally ``kd_mode``) before each forward pass.

    All backbone parameters (including those inside the replaced MLPs) are
    frozen.  Only the newly-added adapter parameters are trainable.
    """
    # Freeze the full backbone first
    for p in model.parameters():
        p.requires_grad_(False)

    mode_context: Dict[str, Any] = {"image_mask": None, "kd_mode": False}

    # Infer device and dtype from backbone parameters
    try:
        first_param = next(model.parameters())
        target_device = first_param.device
        target_dtype  = first_param.dtype
    except StopIteration:
        target_device = torch.device("cpu")
        target_dtype  = torch.float32

    # ChameleonForConditionalGeneration → .model.layers[i].mlp
    decoder_layers = model.model.layers
    n_replaced = 0
    for layer in decoder_layers:
        original_mlp = layer.mlp
        new_mlp = MoDEChameleonMLP(
            original_mlp=original_mlp,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            num_experts=num_experts,
            mode_context=mode_context,
        )
        # Move newly-created adapter params to the same device/dtype as backbone
        new_mlp.to(device=target_device, dtype=target_dtype)
        layer.mlp = new_mlp
        n_replaced += 1

    print(f"[MoDE] Replaced {n_replaced} MLP layers with MoDEChameleonMLP.")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[MoDE] Trainable: {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / n_total:.2f}%)")

    return mode_context


def get_adapter_param_groups(model: nn.Module):
    """Return (vadapter_params, tmoe_params) lists for separate optimisers."""
    va_params, tm_params = [], []
    for module in model.modules():
        if isinstance(module, MoDEChameleonMLP):
            va_params.extend(module.vadapter_parameters())
            tm_params.extend(module.tmoe_parameters())
    return va_params, tm_params
