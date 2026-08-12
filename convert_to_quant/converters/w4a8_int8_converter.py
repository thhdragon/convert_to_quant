"""
W4A8 INT8 Quantization Converter (AsymW4A8Int8Layout).

Implements grouped W4A8 INT8 quantization with ConvRot Hadamard rotation,
per-group FP8/FP32 relative scales, per-channel FP32 scales, optional zero-point
correction, and optional Lloyd-Max codebooks.

Based on comfy-kitchen (Comfy Org, Apache-2.0).
"""

from typing import Dict, Optional, Tuple

torch = None
try:
    import torch
except ImportError:
    pass

from ..constants import W4A8_CONVROT_GROUPSIZE, W4A8_GROUP_SIZE, W4A8_SCALE_DTYPE
from ..utils.logging import verbose
from .base_converter import BaseLearnedConverter

# Check for comfy-kitchen availability
try:
    import comfy_kitchen.tensor.w4a8_int8 as ck_w4a8

    HAS_COMFY_KITCHEN = True
except ImportError:
    HAS_COMFY_KITCHEN = False


def quantize_w4a8_int8_pytorch(
    weight: torch.Tensor,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    symmetric: bool = True,
    scale_dtype: torch.dtype = torch.float8_e4m3fn,
    codebook: bool = True,
    codebook_tensor: Optional[torch.Tensor] = None,
    stochastic_rounding: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Quantize floating weight tensor to W4A8 storage."""
    if HAS_COMFY_KITCHEN:
        return ck_w4a8.quantize_w4a8_int8_weight(
            weight,
            group_size=group_size,
            convrot_groupsize=convrot_groupsize,
            symmetric=symmetric,
            scale_dtype=scale_dtype,
            codebook=codebook,
            codebook_tensor=codebook_tensor,
            stochastic_rounding=stochastic_rounding,
        )
    raise RuntimeError("w4a8_int8 quantization requires comfy_kitchen installed.")


def dequantize_w4a8_int8_pytorch(
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    codebook: Optional[torch.Tensor] = None,
    correction: Optional[torch.Tensor] = None,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize packed W4A8 weight back into floating representation."""
    if HAS_COMFY_KITCHEN:
        return ck_w4a8.dequantize_w4a8_int8_weight(
            qdata,
            s_rel,
            s_channel,
            codebook=codebook,
            correction=correction,
            group_size=group_size,
            convrot_groupsize=convrot_groupsize,
            output_dtype=output_dtype,
        )
    raise RuntimeError("w4a8_int8 dequantization requires comfy_kitchen installed.")


class W4A8Int8Converter:
    """
    W4A8 INT8 block quantization converter.

    Quantizes weights using 16-element groups, ConvRot Hadamard rotation,
    per-channel scales, per-group FP8 relative scales, and optional codebooks.
    """

    def __init__(
        self,
        group_size: int = 16,
        convrot_groupsize: int = 256,
        symmetric: bool = True,
        scale_dtype: torch.dtype = torch.float32,
        codebook: bool = True,
        stochastic_rounding: int = 0,
    ):
        self.group_size = group_size
        self.convrot_groupsize = convrot_groupsize
        self.symmetric = symmetric
        self.scale_dtype = scale_dtype
        self.codebook = codebook
        self.stochastic_rounding = stochastic_rounding

    def convert(
        self,
        W_orig: torch.Tensor,
        key: Optional[str] = None,
        depth: int = -1,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        Dict,
    ]:
        """
        Convert weight tensor to W4A8 format.

        Returns:
            Tuple of (packed_qdata, s_rel, s_channel, correction, codebook, dequantized_weight, extra_tensors)
        """
        qdata, s_rel, s_channel, correction, codebook_tensor = quantize_w4a8_int8_pytorch(
            W_orig,
            group_size=self.group_size,
            convrot_groupsize=self.convrot_groupsize,
            symmetric=self.symmetric,
            scale_dtype=self.scale_dtype,
            codebook=self.codebook,
            stochastic_rounding=self.stochastic_rounding,
        )

        dequantized = dequantize_w4a8_int8_pytorch(
            qdata,
            s_rel,
            s_channel,
            codebook=codebook_tensor,
            correction=correction,
            group_size=self.group_size,
            convrot_groupsize=self.convrot_groupsize,
            output_dtype=W_orig.dtype,
        )

        return qdata, s_rel, s_channel, correction, codebook_tensor, dequantized, {}


class LearnedW4A8Int8Converter(BaseLearnedConverter):
    """
    Learned Rounding W4A8 INT8 converter.

    Applies W4A8 INT8 weight quantization with stochastic rounding or ALS codebook optimization.
    """

    def __init__(
        self,
        group_size: int = 16,
        convrot_groupsize: int = 256,
        symmetric: bool = True,
        scale_dtype: torch.dtype = torch.float32,
        codebook: bool = True,
        stochastic_rounding: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.group_size = group_size
        self.convrot_groupsize = convrot_groupsize
        self.symmetric = symmetric
        self.scale_dtype = scale_dtype
        self.codebook = codebook
        self.stochastic_rounding = stochastic_rounding

        verbose(f"LearnedW4A8Int8Converter initialized on device: {self.device}")
        verbose(f"  - Format: W4A8 INT8 (group_size={self.group_size}, convrot={self.convrot_groupsize}, scale_dtype={self.scale_dtype})")
        verbose(f"  - Stochastic rounding iterations: {self.stochastic_rounding} ({'disabled' if self.no_learned_rounding else 'enabled'})")

    def convert(
        self,
        W_orig: torch.Tensor,
        key: Optional[str] = None,
        depth: int = -1,
        calibration_data: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        Dict,
    ]:
        from tqdm import tqdm
        from torch.optim import AdamW, RAdam
        from ..utils.logging import info, warning

        W_dev = W_orig.to(device=self.device, dtype=torch.float32)

        # --- Initial (simple) quantization to get baseline and codebook ---------
        qdata, s_rel, s_channel, correction, codebook_tensor = quantize_w4a8_int8_pytorch(
            W_dev,
            group_size=self.group_size,
            convrot_groupsize=self.convrot_groupsize,
            symmetric=self.symmetric,
            scale_dtype=self.scale_dtype,
            codebook=self.codebook,
            stochastic_rounding=0,  # always 0 for the seed; optimization loop below handles refinement
        )

        dequantized = dequantize_w4a8_int8_pytorch(
            qdata,
            s_rel,
            s_channel,
            codebook=codebook_tensor,
            correction=correction,
            group_size=self.group_size,
            convrot_groupsize=self.convrot_groupsize,
            output_dtype=torch.float32,
        )

        # --- AdaRound optimization loop -----------------------------------------
        if (
            not self.no_learned_rounding
            and self.num_iter > 0
            and calibration_data is not None
        ):
            X = calibration_data.to(device=self.device, dtype=torch.float32)
            Y_ref = X @ W_dev.T  # target activations using original float weight

            # SVD of original weight for projection regularization
            U_k, Vh_k, _ = self._compute_svd_components(W_dev, verbose=True)

            # Build soft-rounding targets from the dequantized weight.
            # We work in the per-channel-scaled space where comfy_kitchen's ConvRot +
            # group quantization happened; we approximate that as W_dev / s_channel_bc.
            # The goal is to learn better {0,1} rounding decisions (V → sigmoid → {0,1}).
            s_channel_bc = s_channel.view(-1, 1).to(dtype=torch.float32)
            W_scaled = W_dev / s_channel_bc.clamp_min(1e-12)
            W_floor = W_scaled.floor()
            target_frac = (W_scaled - W_floor).clamp_(1e-6, 1.0 - 1e-6)

            T_start, T_end = 20.0, 2.0
            V_init = -torch.log((1.0 / target_frac) - 1.0) * T_start
            V = V_init.clone().detach().requires_grad_(True)
            del W_scaled, target_frac, V_init

            # Optimizer
            curr_lr = self.lr
            if self.optimizer_choice == "adamw":
                optimizer = AdamW([V], lr=curr_lr)
            elif self.optimizer_choice == "radam":
                optimizer = RAdam([V], lr=curr_lr)
            elif self.optimizer_choice == "prodigy":
                from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
                optimizer = ProdigyPlusScheduleFree([V], lr=curr_lr, use_schedulefree=False, use_speed=self.use_speed)
            else:
                optimizer = None  # manual SGD

            # Initial metrics for normalizing loss
            with torch.no_grad():
                init_mse = torch.nn.functional.mse_loss(X @ dequantized.T, Y_ref)
                init_svd_err = torch.linalg.norm(U_k.T @ (dequantized - W_dev) @ Vh_k.T)

            alpha_svd = 0.01 * (init_mse.item() / init_svd_err.item()) if init_svd_err.item() > 1e-8 else 0.0

            M, N = W_dev.shape
            best_loss = float("inf")
            best_V = V.detach().clone()
            plateau_counter = 0
            cooldown_counter = 0
            worse_loss_counter = 0
            loss_history = []

            schedule_name = self.lr_schedule
            effective_patience, effective_factor, effective_cooldown = None, None, None
            if schedule_name == "plateau":
                effective_patience, effective_factor, effective_cooldown = self._compute_shape_aware_plateau_params(M, N)

            decay_factor = self.lr_factor if self.lr_factor < 1.0 else 0.95
            window_size = max(5, int(2.5 / (1.0 - decay_factor)))
            loss_span_threshold = self.early_stop_loss / (1.0 - decay_factor)

            pbar = tqdm(
                range(self.num_iter),
                desc=f"    Optimizing W4A8 (AdaRound-{self.optimizer_choice}-{schedule_name})",
                leave=False,
                dynamic_ncols=True,
            )

            for i in pbar:
                if optimizer is not None:
                    optimizer.zero_grad()

                temp = T_start + (T_end - T_start) * (i / self.num_iter)
                h_V = torch.sigmoid(V / temp)
                W_q_soft = (W_floor + h_V).clamp(-7, 7)
                W_dequant_soft = W_q_soft * s_channel_bc

                Y_pred = X @ W_dequant_soft.T
                loss_mse = torch.nn.functional.mse_loss(Y_pred, Y_ref) / max(init_mse.item(), 1e-12)

                weight_err = W_dequant_soft - W_dev
                loss_svd = (
                    torch.linalg.norm(U_k.T @ weight_err @ Vh_k.T) / init_svd_err.item()
                    if alpha_svd > 0
                    else torch.tensor(0.0, device=self.device)
                )

                loss_reg = (1.0 - (2.0 * h_V - 1.0).pow(2)).mean()

                loss = loss_mse + 0.01 * loss_svd + 0.1 * loss_reg
                scaled_loss = loss * 1e5

                if optimizer is not None:
                    scaled_loss.backward()
                    if V.grad is not None:
                        V.grad.div_(1e5)
                    optimizer.step()
                else:
                    if V.grad is not None:
                        V.grad.zero_()
                    scaled_loss.backward()
                    with torch.no_grad():
                        V -= curr_lr * (V.grad / 1e5)

                current_loss_val = loss.item()
                improved = self._check_improvement(current_loss_val, best_loss)
                loss_history.append(current_loss_val)
                if len(loss_history) > window_size:
                    loss_history.pop(0)

                converged_ratio = ((torch.sigmoid(V) < 0.05) | (torch.sigmoid(V) > 0.95)).float().mean().item()
                if converged_ratio >= 0.90 and len(loss_history) == window_size:
                    loss_span = max(loss_history) - min(loss_history)
                    if loss_span < loss_span_threshold:
                        info(f"\n      - Early stop: {converged_ratio*100:.1f}% converged, span={loss_span:.2e}")
                        break

                prev_worse = worse_loss_counter
                if improved:
                    best_loss = current_loss_val
                    best_V = V.detach().clone()
                    plateau_counter = 0
                    if self.lr_adaptive_mode == "simple-reset":
                        worse_loss_counter = 0
                else:
                    worse_loss_counter += 1
                    plateau_counter += 1

                prodigy_warmup = self.optimizer_choice == "prodigy" and i < 50

                if schedule_name == "exponential":
                    if not prodigy_warmup:
                        curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                        if optimizer is not None:
                            for pg in optimizer.param_groups:
                                pg["lr"] = curr_lr
                elif schedule_name == "plateau":
                    if prodigy_warmup:
                        plateau_counter = 0
                    elif cooldown_counter > 0:
                        cooldown_counter -= 1
                    elif plateau_counter >= effective_patience:
                        if curr_lr > self.lr_min:
                            old_lr = curr_lr
                            curr_lr = max(curr_lr * effective_factor, self.lr_min)
                            if optimizer is not None:
                                for pg in optimizer.param_groups:
                                    pg["lr"] = curr_lr
                            cooldown_counter = effective_cooldown
                        plateau_counter = 0
                else:  # adaptive
                    new_lr, lr_updated = self._adaptive_lr_update_cosine(
                        curr_lr, improved, prev_worse if improved else worse_loss_counter,
                        i, (M, N), self.early_stop_lr
                    )
                    if lr_updated:
                        curr_lr = new_lr
                        if optimizer is not None:
                            for pg in optimizer.param_groups:
                                pg["lr"] = curr_lr

            pbar.close()

            # Apply best rounding decisions → re-quantize to get final hard integers
            with torch.no_grad():
                best_h = torch.sigmoid(best_V / T_end)
                W_hard = ((W_floor + best_h).round().clamp(-7, 7) * s_channel_bc)

            del V, best_V, X, Y_ref, U_k, Vh_k, W_floor, s_channel_bc, W_q_soft, W_dequant_soft

            # Re-quantize from the rounded float weight so comfy_kitchen sets up the
            # packed format, scales and optional codebook properly.
            try:
                qdata, s_rel, s_channel, correction, codebook_tensor = quantize_w4a8_int8_pytorch(
                    W_hard,
                    group_size=self.group_size,
                    convrot_groupsize=self.convrot_groupsize,
                    symmetric=self.symmetric,
                    scale_dtype=self.scale_dtype,
                    codebook=self.codebook,
                    stochastic_rounding=0,
                )
                dequantized = dequantize_w4a8_int8_pytorch(
                    qdata, s_rel, s_channel,
                    codebook=codebook_tensor, correction=correction,
                    group_size=self.group_size, convrot_groupsize=self.convrot_groupsize,
                    output_dtype=W_orig.dtype,
                )
            except Exception as e:
                warning(f"    - W4A8 re-quantization after AdaRound failed ({e}); using initial quantization.")
                # fall back to the initial result already stored in qdata/s_rel/etc.
                dequantized = dequantize_w4a8_int8_pytorch(
                    qdata, s_rel, s_channel,
                    codebook=codebook_tensor, correction=correction,
                    group_size=self.group_size, convrot_groupsize=self.convrot_groupsize,
                    output_dtype=W_orig.dtype,
                )
        else:
            if not self.no_learned_rounding and calibration_data is None:
                warning(f"    - W4A8 learned rounding skipped for {key}: no calibration data available.")
            dequantized = dequantized.to(dtype=W_orig.dtype)

        extra_tensors = {}
        if self._should_extract_lora(key, W_orig.shape, depth):
            lora_data = self._extract_error_lora(W_orig, dequantized)
            if lora_data:
                extra_tensors.update(lora_data)

        return qdata, s_rel, s_channel, correction, codebook_tensor, dequantized, extra_tensors

