"""
MoDE Adapters with LCE/BiLCE-LoRA: V-Adapter (LoRA) and T-MoE (MoE-LoRA)

Per-token modality split in every MLP linear layer:
  - Image tokens (vocab id in [4, 8195])  → V-Adapter (single LoRA)
  - Text tokens (everything else)          → T-MoE (dense soft-routing MoE-LoRA)

Variants:
  use_lce=True   → LCE-LoRA:  base + γ·extension
  use_bilce=True → BiLCE-LoRA: common + (1-γ)·short-private + γ·long-private

Backbone (ChameleonMLP) is fully frozen.  Only adapter params are trained.
"""

import math
from typing import Any, Dict, List, Optional

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


def _spectral_norm_estimate(H: torch.Tensor, num_iters: int = 3) -> torch.Tensor:
    """Estimate the largest singular value of *H* via power iteration.

    H should be in float32 for numerical stability.  Returns a scalar tensor.
    """
    _T, d = H.shape
    v = torch.randn(d, device=H.device, dtype=H.dtype)
    v = v / (v.norm() + 1e-12)
    for _ in range(num_iters):
        u = H @ v
        u = u / (u.norm() + 1e-12)
        v = H.t() @ u
        v = v / (v.norm() + 1e-12)
    return (H @ v).norm()


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
# LCE-LoRA: dual-branch LoRA with length-content gating
# ---------------------------------------------------------------------------

class LCELoRALinear(nn.Module):
    """Dual-branch LoRA for LCE: base (always on) + extension (gated by γ).

    Δ = (α / r) * ( B_b · A_b(x)  +  γ · B_e · A_e(x) )
    where r = base_rank + ext_rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        base_rank: int,
        ext_rank: int,
        alpha: int = 16,
    ):
        super().__init__()
        self.base_rank = base_rank
        self.ext_rank = ext_rank
        total_rank = base_rank + ext_rank
        self.scale = alpha / total_rank

        # Base branch (always active)
        self.base_A = nn.Linear(in_features, base_rank, bias=False)
        self.base_B = nn.Linear(base_rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.base_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.base_B.weight)

        # Extension branch (gated by γ)
        self.ext_A = nn.Linear(in_features, ext_rank, bias=False)
        self.ext_B = nn.Linear(ext_rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.ext_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.ext_B.weight)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        x:     [N, in_features]
        gamma: [N, 1]  sample-level gate value
        """
        base_out = self.base_B(self.base_A(x))
        ext_out = self.ext_B(self.ext_A(x))
        return (base_out + gamma * ext_out) * self.scale


# ---------------------------------------------------------------------------
# T-MoE: dense soft-routing MoE-LoRA for text tokens
# ---------------------------------------------------------------------------

class TMoE(nn.Module):
    """Text-side MoE-LoRA with dense softmax routing.

    Router computes g_j = softmax(x W_g)_j over all E experts.
    Each expert is a LoRA; outputs are weighted-summed.
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
# LCE-TMoE: T-MoE with LCE-LoRA experts
# ---------------------------------------------------------------------------

class LCETMoE(nn.Module):
    """Text-side MoE-LoRA where each expert is an LCE dual-branch LoRA."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        base_rank: int,
        ext_rank: int,
        alpha: int = 16,
        num_experts: int = 4,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            LCELoRALinear(in_features, out_features, base_rank, ext_rank, alpha)
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(in_features, num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        x:     [N, in_features]
        gamma: [N, 1]  sample-level gate value
        """
        gates = F.softmax(self.router(x), dim=-1)                          # [N, E]
        expert_outs = torch.stack([e(x, gamma) for e in self.experts], dim=1)  # [N, E, out]
        return (gates.unsqueeze(-1) * expert_outs).sum(dim=1)              # [N, out]


# ---------------------------------------------------------------------------
# BiLCE-LoRA: three-branch LoRA (common + short-private + long-private)
# ---------------------------------------------------------------------------

class BiLCELoRALinear(nn.Module):
    """Three-branch LoRA for BiLCE: symmetric protection for short & long text.

    Δ = (α/r) * ( B_c·A_c(x)  +  (1-γ)·B_s·A_s(x)  +  γ·B_l·A_l(x) )
    where r = common_rank + short_rank + long_rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        common_rank: int,
        short_rank: int,
        long_rank: int,
        alpha: int = 16,
    ):
        super().__init__()
        self.common_rank = common_rank
        self.short_rank = short_rank
        self.long_rank = long_rank
        total_rank = common_rank + short_rank + long_rank
        self.scale = alpha / total_rank

        # Common branch (always active)
        self.common_A = nn.Linear(in_features, common_rank, bias=False)
        self.common_B = nn.Linear(common_rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.common_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.common_B.weight)

        # Short-private branch (gated by 1-γ)
        self.short_A = nn.Linear(in_features, short_rank, bias=False)
        self.short_B = nn.Linear(short_rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.short_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.short_B.weight)

        # Long-private branch (gated by γ)
        self.long_A = nn.Linear(in_features, long_rank, bias=False)
        self.long_B = nn.Linear(long_rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.long_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.long_B.weight)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        x:     [N, in_features]
        gamma: [N, 1]  sample-level gate value in (0, 1)
        """
        common_out = self.common_B(self.common_A(x))
        short_out = self.short_B(self.short_A(x))
        long_out = self.long_B(self.long_A(x))
        return (common_out + (1 - gamma) * short_out + gamma * long_out) * self.scale


class BiLCETMoE(nn.Module):
    """Text-side MoE-LoRA where each expert is a BiLCE three-branch LoRA."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        common_rank: int,
        short_rank: int,
        long_rank: int,
        alpha: int = 16,
        num_experts: int = 4,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            BiLCELoRALinear(in_features, out_features,
                            common_rank, short_rank, long_rank, alpha)
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(in_features, num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        x:     [N, in_features]
        gamma: [N, 1]  sample-level gate value
        """
        gates = F.softmax(self.router(x), dim=-1)
        expert_outs = torch.stack(
            [e(x, gamma) for e in self.experts], dim=1,
        )
        return (gates.unsqueeze(-1) * expert_outs).sum(dim=1)


# ---------------------------------------------------------------------------
# Length-Content Gate:  γ = σ( wᵀ LN(h̄) + b_ℓ )
# ---------------------------------------------------------------------------

class LengthContentGate(nn.Module):
    """Sample-level gate combining content embedding and length bucket bias.

    γ(x) = σ( wᵀ LayerNorm(h̄) + b_ℓ )
    where h̄ is the mean-pooled instruction embedding and ℓ is the length bucket.
    """

    def __init__(self, hidden_size: int, num_buckets: int = 4):
        super().__init__()
        self.num_buckets = num_buckets
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.w = nn.Parameter(torch.zeros(hidden_size))
        self.bucket_bias = nn.Parameter(torch.zeros(num_buckets))
        nn.init.normal_(self.w, std=0.02)

    def forward(self, h_bar: torch.Tensor, bucket_ids: torch.LongTensor) -> torch.Tensor:
        """
        h_bar:      [B, hidden_size]  mean-pooled instruction hidden states
        bucket_ids: [B]               length bucket indices (0 .. num_buckets-1)
        Returns:    [B]               gate values in (0, 1)
        """
        normed = self.layer_norm(h_bar)                    # [B, d]
        content_score = (normed * self.w).sum(dim=-1)      # [B]
        length_bias = self.bucket_bias[bucket_ids]         # [B]
        return torch.sigmoid(content_score + length_bias)  # [B]


# ---------------------------------------------------------------------------
# MoDEChameleonMLP: drop-in replacement for ChameleonMLP
# ---------------------------------------------------------------------------

class MoDEChameleonMLP(nn.Module):
    """Replaces a ChameleonMLP layer with modality-split adapters.

    For each of gate_proj / up_proj / down_proj the forward is:
        h = base_linear(x) + adapter_delta(x)

    where adapter_delta routes image tokens through the V-Adapter (LoRA)
    and text tokens through T-MoE (or LCE-TMoE when use_lce=True).
    In KD mode (kd_mode=True) the T-MoE branch is skipped entirely so that
    no gradient from the KD loss flows into text-side parameters.

    The modality mask, kd_mode flag, and optional gamma are read from
    ``mode_context``:
        mode_context['image_mask']  : BoolTensor [B, S]
        mode_context['kd_mode']     : bool (default False)
        mode_context['gamma']       : Tensor [B] or None (LCE/BiLCE gate value)
    """

    def __init__(
        self,
        original_mlp: nn.Module,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        num_experts: int = 4,
        mode_context: Optional[Dict[str, Any]] = None,
        use_lce: bool = False,
        base_rank: Optional[int] = None,
        ext_rank: Optional[int] = None,
        use_bilce: bool = False,
        common_rank: Optional[int] = None,
        short_rank: Optional[int] = None,
        long_rank: Optional[int] = None,
    ):
        super().__init__()
        hidden: int = original_mlp.hidden_size
        intermediate: int = original_mlp.intermediate_size

        # Freeze backbone
        self.base = original_mlp
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.act_fn = original_mlp.act_fn
        self.use_lce = use_lce
        self.use_bilce = use_bilce
        self.use_gamma = use_lce or use_bilce

        # V-Adapters (image tokens) — unchanged
        self.va_gate = LoRALinear(hidden, intermediate, lora_rank, lora_alpha)
        self.va_up   = LoRALinear(hidden, intermediate, lora_rank, lora_alpha)
        self.va_down = LoRALinear(intermediate, hidden, lora_rank, lora_alpha)

        # T-MoE (text tokens)
        if use_bilce:
            assert common_rank is not None and short_rank is not None and long_rank is not None, \
                "common_rank, short_rank, long_rank must be set when use_bilce=True"
            self.tm_gate = BiLCETMoE(hidden, intermediate, common_rank, short_rank, long_rank, lora_alpha, num_experts)
            self.tm_up   = BiLCETMoE(hidden, intermediate, common_rank, short_rank, long_rank, lora_alpha, num_experts)
            self.tm_down = BiLCETMoE(intermediate, hidden, common_rank, short_rank, long_rank, lora_alpha, num_experts)
        elif use_lce:
            assert base_rank is not None and ext_rank is not None, \
                "base_rank and ext_rank must be set when use_lce=True"
            self.tm_gate = LCETMoE(hidden, intermediate, base_rank, ext_rank, lora_alpha, num_experts)
            self.tm_up   = LCETMoE(hidden, intermediate, base_rank, ext_rank, lora_alpha, num_experts)
            self.tm_down = LCETMoE(intermediate, hidden, base_rank, ext_rank, lora_alpha, num_experts)
        else:
            self.tm_gate = TMoE(hidden, intermediate, lora_rank, lora_alpha, num_experts)
            self.tm_up   = TMoE(hidden, intermediate, lora_rank, lora_alpha, num_experts)
            self.tm_down = TMoE(intermediate, hidden, lora_rank, lora_alpha, num_experts)

        self.mode_context: Dict[str, Any] = mode_context if mode_context is not None else {}

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _compute_delta(
        self,
        x_flat: torch.Tensor,                        # [N, in_features]
        vadapter: LoRALinear,
        tmoe,                                         # TMoE or LCETMoE
        mask_flat: Optional[torch.BoolTensor],        # [N]
        kd_mode: bool,
        out_features: int,
        gamma_flat: Optional[torch.Tensor] = None,    # [N] (LCE gate per token)
    ) -> torch.Tensor:
        """Compute adapter delta for one linear projection."""
        N = x_flat.shape[0]
        delta = torch.zeros(N, out_features, dtype=x_flat.dtype, device=x_flat.device)

        if mask_flat is not None:
            img_idx = mask_flat.nonzero(as_tuple=True)[0]
            if img_idx.numel() > 0:
                delta[img_idx] = vadapter(x_flat[img_idx])

            if not kd_mode:
                txt_idx = (~mask_flat).nonzero(as_tuple=True)[0]
                if txt_idx.numel() > 0:
                    if self.use_gamma and gamma_flat is not None:
                        gamma_txt = gamma_flat[txt_idx].unsqueeze(-1)  # [N_txt, 1]
                        delta[txt_idx] = tmoe(x_flat[txt_idx], gamma_txt)
                    else:
                        delta[txt_idx] = tmoe(x_flat[txt_idx])
        else:
            # No mask available: treat all tokens as text (fallback)
            if not kd_mode:
                if self.use_gamma and gamma_flat is not None:
                    delta = tmoe(x_flat, gamma_flat.unsqueeze(-1))
                else:
                    delta = tmoe(x_flat)

        return delta

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image_mask: Optional[torch.BoolTensor] = self.mode_context.get("image_mask")
        kd_mode: bool = self.mode_context.get("kd_mode", False)
        gamma: Optional[torch.Tensor] = self.mode_context.get("gamma")  # [B] or None

        B, S, H = x.shape
        x_flat = x.view(B * S, H)

        if image_mask is not None:
            mask_flat = image_mask.to(x.device).view(B * S)
        else:
            mask_flat = None

        # Expand sample-level gamma [B] to token-level [B*S]
        if gamma is not None:
            gamma_flat = gamma.to(x.device).unsqueeze(1).expand(B, S).reshape(B * S)
        else:
            gamma_flat = None

        intermediate_size = self.base.intermediate_size

        # --- gate_proj + adapter ---
        gate_base  = self.base.gate_proj(x)                           # [B,S,I]
        gate_delta = self._compute_delta(
            x_flat, self.va_gate, self.tm_gate, mask_flat, kd_mode,
            intermediate_size, gamma_flat,
        ).view(B, S, intermediate_size)
        gate_out = gate_base + gate_delta

        # --- up_proj + adapter ---
        up_base  = self.base.up_proj(x)
        up_delta = self._compute_delta(
            x_flat, self.va_up, self.tm_up, mask_flat, kd_mode,
            intermediate_size, gamma_flat,
        ).view(B, S, intermediate_size)
        up_out = up_base + up_delta

        # --- gated activation (SiLU/GELU) ---
        intermediate = self.act_fn(gate_out) * up_out                 # [B,S,I]

        # --- down_proj + adapter ---
        down_base  = self.base.down_proj(intermediate)                # [B,S,H]
        inter_flat = intermediate.view(B * S, intermediate_size)
        down_delta = self._compute_delta(
            inter_flat, self.va_down, self.tm_down, mask_flat, kd_mode,
            H, gamma_flat,
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
        """Yield T-MoE / LCE-TMoE (text-side) parameters."""
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
    use_lce: bool = False,
    base_rank: Optional[int] = None,
    ext_rank: Optional[int] = None,
    use_bilce: bool = False,
    common_rank: Optional[int] = None,
    short_rank: Optional[int] = None,
    long_rank: Optional[int] = None,
    num_buckets: int = 4,
) -> Dict[str, Any]:
    """Replace every ChameleonMLP in the model with MoDEChameleonMLP.

    Returns a shared ``mode_context`` dict that callers must populate with
    ``image_mask`` (and optionally ``kd_mode``, ``gamma``) before each
    forward pass.

    When use_lce or use_bilce is True, also creates a LengthContentGate and
    attaches it to the model as ``model.lce_gate``.
    """
    # Freeze the full backbone first
    for p in model.parameters():
        p.requires_grad_(False)

    mode_context: Dict[str, Any] = {"image_mask": None, "kd_mode": False, "gamma": None}

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
            use_lce=use_lce,
            base_rank=base_rank,
            ext_rank=ext_rank,
            use_bilce=use_bilce,
            common_rank=common_rank,
            short_rank=short_rank,
            long_rank=long_rank,
        )
        new_mlp.to(device=target_device, dtype=target_dtype)
        layer.mlp = new_mlp
        n_replaced += 1

    # LCE gate module (shared across all layers, same gate formula).
    # BiLCE uses a parameter-free Stable-Rank Prompt Gate instead.
    if use_lce:
        hidden_size = model.config.hidden_size
        lce_gate = LengthContentGate(hidden_size, num_buckets)
        lce_gate.to(device=target_device, dtype=target_dtype)
        model.lce_gate = lce_gate

    tag = "MoDE+BiLCE" if use_bilce else ("MoDE+LCE" if use_lce else "MoDE")
    print(f"[{tag}] Replaced {n_replaced} MLP layers with MoDEChameleonMLP.")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] Trainable: {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / n_total:.2f}%)")

    return mode_context


def get_adapter_param_groups(model: nn.Module):
    """Return (vadapter_params, tmoe_params) lists for separate optimisers.

    When LCE is used, tmoe_params includes the LengthContentGate parameters.
    """
    va_params, tm_params = [], []
    for module in model.modules():
        if isinstance(module, MoDEChameleonMLP):
            va_params.extend(module.vadapter_parameters())
            tm_params.extend(module.tmoe_parameters())
    if hasattr(model, "lce_gate"):
        tm_params.extend(model.lce_gate.parameters())
    return va_params, tm_params


# ---------------------------------------------------------------------------
# LCE gamma computation helpers
# ---------------------------------------------------------------------------

def compute_lce_gamma(
    model: nn.Module,
    input_ids: torch.LongTensor,
    bucket_boundaries: torch.Tensor,
    sep_token_id: int = 8710,
    pad_token_id: int = 1,
    bos_token_id: int = 0,
    instruction_only: bool = True,
) -> torch.Tensor:
    """Compute the sample-level LCE gate value γ for a batch.

    Uses the frozen embedding layer to obtain instruction token representations,
    mean-pools them, and passes through the LengthContentGate module.

    Args:
        model:             The student model (unwrapped, not DDP).
        input_ids:         [B, S] token ids.
        bucket_boundaries: 1-D sorted tensor of bucket boundaries for
                           torch.bucketize (e.g. tensor([20, 50, 100])).
        sep_token_id:      Token id that separates instruction from answer.
        pad_token_id:      Padding token id.
        bos_token_id:      Beginning-of-sequence token id.
        instruction_only:  If True, only use tokens before the first SEP for
                           gate computation (training mode).  If False, use
                           all non-image/non-special tokens (inference mode).

    Returns:
        gamma: [B] tensor in (0, 1), requires grad w.r.t. gate parameters.
    """
    device = input_ids.device
    B, S = input_ids.shape

    with torch.no_grad():
        embeds = model.model.embed_tokens(input_ids)  # [B, S, d]

    h_bars: List[torch.Tensor] = []
    lengths: List[int] = []

    for b in range(B):
        seq = input_ids[b]

        # Build mask for instruction text tokens
        if instruction_only:
            sep_positions = (seq == sep_token_id).nonzero(as_tuple=True)[0]
            sep_pos = sep_positions[0].item() if len(sep_positions) > 0 else S
            ins_mask = torch.zeros(S, dtype=torch.bool, device=device)
            ins_mask[:sep_pos] = True
        else:
            ins_mask = torch.ones(S, dtype=torch.bool, device=device)

        # Exclude image tokens, PAD, and BOS
        ins_mask &= ~((seq >= IMAGE_TOKEN_MIN) & (seq <= IMAGE_TOKEN_MAX))
        ins_mask &= (seq != pad_token_id)
        ins_mask &= (seq != bos_token_id)

        ins_idx = ins_mask.nonzero(as_tuple=True)[0]
        if ins_idx.numel() > 0:
            h_bars.append(embeds[b, ins_idx].mean(dim=0))
            lengths.append(ins_idx.numel())
        else:
            h_bars.append(embeds[b].mean(dim=0))
            lengths.append(1)

    h_bar_batch = torch.stack(h_bars)  # [B, d]
    length_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
    bucket_ids = torch.bucketize(length_tensor, bucket_boundaries.to(device))

    gamma = model.lce_gate(h_bar_batch, bucket_ids)  # [B]
    return gamma


# ---------------------------------------------------------------------------
# Stable-Rank Prompt Gate for BiLCE (parameter-free)
# ---------------------------------------------------------------------------

def compute_stable_rank_gamma(
    model: nn.Module,
    input_ids: torch.LongTensor,
    short_rank: int,
    long_rank: int,
    sep_token_id: int = 8710,
    pad_token_id: int = 1,
    bos_token_id: int = 0,
    instruction_only: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute γ(x) via the Stable-Rank Prompt Gate for BiLCE.

    Takes the hidden states entering the MLP of the **first** (frozen)
    transformer block, extracts instruction-only tokens, centres them,
    and maps the stable rank of the centred matrix to (0, 1):

        sr  = ||H̃||_F² / (||H̃||_2² + ε)
        γ   = sr / (sr + r_p)

    where r_p = short_rank + long_rank (the private rank budget).

    Because the first block's attention & layer-norms are frozen, γ is
    input-dependent but training-invariant.

    Returns:
        gamma: [B] tensor in (0, 1), detached (no learnable parameters).
    """
    device = input_ids.device
    B, S = input_ids.shape
    r_p = float(short_rank + long_rank)

    with torch.no_grad():
        embeds = model.model.embed_tokens(input_ids)          # [B, S, d]

        layer0 = model.model.layers[0]
        residual = embeds
        h = layer0.input_layernorm(embeds)
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        h, _, _ = layer0.self_attn(
            hidden_states=h,
            position_ids=position_ids,
        )
        h = residual + h
        h_pre_mlp = layer0.post_attention_layernorm(h)        # [B, S, d]

    gammas: List[torch.Tensor] = []

    for b in range(B):
        seq = input_ids[b]

        # ── instruction token mask (same logic as compute_lce_gamma) ──
        if instruction_only:
            sep_positions = (seq == sep_token_id).nonzero(as_tuple=True)[0]
            sep_pos = sep_positions[0].item() if len(sep_positions) > 0 else S
            ins_mask = torch.zeros(S, dtype=torch.bool, device=device)
            ins_mask[:sep_pos] = True
        else:
            ins_mask = torch.ones(S, dtype=torch.bool, device=device)

        ins_mask &= ~((seq >= IMAGE_TOKEN_MIN) & (seq <= IMAGE_TOKEN_MAX))
        ins_mask &= (seq != pad_token_id)
        ins_mask &= (seq != bos_token_id)

        ins_idx = ins_mask.nonzero(as_tuple=True)[0]
        if ins_idx.numel() < 2:
            gammas.append(h_pre_mlp.new_tensor(0.5))
            continue

        # Cast to float32 for numerical stability
        H = h_pre_mlp[b, ins_idx].float()                     # [T, d]

        # Centre
        H = H - H.mean(dim=0, keepdim=True)

        # Frobenius norm squared
        frob2 = (H * H).sum()

        # Spectral norm via power iteration
        sigma_max = _spectral_norm_estimate(H)

        # Stable rank
        sr = frob2 / (sigma_max * sigma_max + eps)

        # Map to [0, 1]
        gamma_val = sr / (sr + r_p)
        gammas.append(gamma_val.to(h_pre_mlp.dtype))

    return torch.stack(gammas)                                 # [B]
