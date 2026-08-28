"""Learned rounding converter for INT4 ConvRot W4A4 quantization.

This module implements advanced INT4 W4A4 quantization using learned adaptive rounding
(AdaRound) with SVD-based optimization and group-wise Hadamard rotation (ConvRot).
"""

import gc
import math
import torch
from torch.optim import AdamW, RAdam
from tqdm import tqdm

from ..constants import (
    COMPUTE_DTYPE,
    INT4_MAX,
    INT4_MIN,
    SCALE_DTYPE,
)
from ..pinned_transfer import transfer_to_gpu_pinned
from ..utils.logging import debug, info, verbose, warning
from .base_converter import BaseLearnedConverter


class LearnedRoundingConverter(BaseLearnedConverter):
    """
    Learned rounding converter for INT4 ConvRot W4A4 quantization.

    Inherits shared infrastructure from BaseLearnedConverter.
    Configured by default for INT4 W4A4 row-wise ConvRot quantization.
    """

    def __init__(
        self,
        scaling_mode: str = "row",
        block_size: int = 64,
        target_format: str = "int4",
        lr: float = 1.0,
        extract_lora: bool = False,
        lora_rank: int = 32,
        lora_depth: int = 1,
        lora_target: list[str] | None = None,
        lora_ar_threshold: float = 0.0,
        convrot: bool = True,
        convrot_group_size: int = 256,
        dynamic_convrot: bool = False,
        scale_optimization: str = "fixed",
        w4a4_untouched_activations: bool = False,
        smooth_convrot: bool = True,
        smooth_alpha: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            lr=lr,
            extract_lora=extract_lora,
            lora_rank=lora_rank,
            lora_depth=lora_depth,
            lora_target=lora_target,
            lora_ar_threshold=lora_ar_threshold,
            **kwargs,
        )

        self.block_size = block_size
        self.target_format = "int4"
        self.scaling_mode = "row"
        self.convrot = convrot
        self.convrot_group_size = convrot_group_size
        self.dynamic_convrot = dynamic_convrot
        if self.dynamic_convrot:
            self.convrot = True
        self.scale_optimization = scale_optimization
        self.w4a4_untouched_activations = w4a4_untouched_activations
        self.smooth_convrot = smooth_convrot
        self.smooth_alpha = smooth_alpha
        self.has_bias = True
        self.calib_scale = 1.0
        self.min_snr_db = kwargs.get("min_snr_db", 0.0)
        self.min_cossim = kwargs.get("min_cossim", 0.0)
        self.fallback_unresponsive = kwargs.get("fallback_unresponsive", False)
        self._last_opt_improvement = None

        verbose(f"LearnedRoundingConverter initialized for INT4 ConvRot W4A4 on device: {self.device}")
        verbose(f"  - Target format: {self.target_format}, Scaling mode: {self.scaling_mode}")
        if self.convrot:
            verbose(f"  - ConvRot Hadamard rotation enabled (group_size={self.convrot_group_size}, smooth_convrot={self.smooth_convrot})")

            verbose(f"  - ConvRot Hadamard rotation enabled (group_size={self.convrot_group_size})")

    def convert(
        self,
        W_orig: torch.Tensor,
        key: list[str] | None = None,
        depth: int = -1,
        calibration_data: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        has_bias = kwargs.get("has_bias", True)
        self.has_bias = has_bias
        self._current_extra_tensors = {}

        attempt = 1
        max_attempts = 10
        orig_top_p = self.top_p
        orig_max_k = self.max_k
        orig_min_k = self.min_k
        self.calib_scale = 1.0

        try:
            while True:
                try:
                    self._current_extra_tensors = {}
                    W_float32 = transfer_to_gpu_pinned(W_orig, self.device, COMPUTE_DTYPE)

                    if torch.all(W_float32 == 0):
                        verbose("  - Tensor is all zeros, skipping optimization.")
                        out_features, in_features = W_float32.shape
                        qdata = torch.zeros((out_features, in_features // 2), dtype=torch.int8, device=self.device)
                        dequant_scale = torch.ones(out_features, device=self.device, dtype=SCALE_DTYPE)
                        return qdata, dequant_scale, torch.zeros_like(W_float32), {}

                    W_orig_float32 = W_float32.clone() if self.extract_lora else None

                    qdata, scale, dequantized = self._convert_int4_convrot(
                        W_float32, calibration_data=calibration_data
                    )

                    extra_tensors = self._current_extra_tensors.copy()
                    self._current_extra_tensors.clear()

                    if self._should_extract_lora(key, W_orig.shape, depth):
                        lora_data = self._extract_error_lora(W_orig_float32, dequantized)
                        if lora_data:
                            extra_tensors.update(lora_data)

                    return qdata, scale, dequantized, extra_tensors

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or (
                        isinstance(e, RuntimeError)
                        and any(msg in str(e).lower() for msg in ["out of memory", "cuda out of memory", "oom"])
                    )
                    if not is_oom:
                        raise e

                    verbose(f"    - [OOM Warning] Out of memory during layer conversion (attempt {attempt}/{max_attempts}).")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    self.top_p *= 0.7
                    self.max_k = int(self.max_k * 0.7)
                    self.min_k = int(self.min_k * 0.7)
                    self.calib_scale *= 0.5

                    if attempt >= max_attempts or (self.max_k < 1 and self.min_k < 1 and self.top_p < 1e-4):
                        raise e

                    attempt += 1

        finally:
            self.top_p = orig_top_p
            self.max_k = orig_max_k
            self.min_k = orig_min_k
            self.calib_scale = 1.0

    def _convert_int4_convrot(
        self, W_float32: torch.Tensor, calibration_data: list[torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """INT4 W4A4 ConvRot quantization conversion path."""
        from ..utils.convrot import (
            build_hadamard,
            pack_int4_row_major,
            rotate_weight,
            unpack_int4_row_major,
        )

        convrot_applied = False
        layer_group_size = self.convrot_group_size
        M, N = W_float32.shape

        if self.convrot and self.scaling_mode == "row":
            if self.dynamic_convrot:
                from ..utils.convrot import find_max_compatible_group_size
                layer_group_size = find_max_compatible_group_size(N, min_group_size=self.convrot_group_size)

            if layer_group_size is not None and N % layer_group_size == 0:
                try:
                    if self.smooth_convrot:
                        from ..utils.convrot import balance_channels_smoothquant
                        if calibration_data is not None and isinstance(calibration_data, torch.Tensor):
                            calib_x = calibration_data.to(device=self.device, dtype=COMPUTE_DTYPE)
                        else:
                            calib_x = torch.randn(256, N, device=self.device, dtype=COMPUTE_DTYPE)
                        W_float32, s_c = balance_channels_smoothquant(W_float32, calib_x, alpha=self.smooth_alpha)
                        if calibration_data is not None and isinstance(calibration_data, torch.Tensor):
                            calibration_data = calibration_data / s_c.unsqueeze(0).to(device=calibration_data.device)

                    H = build_hadamard(layer_group_size, device=self.device, dtype=COMPUTE_DTYPE)
                    W_float32 = rotate_weight(W_float32, H, layer_group_size)
                    info(f"    - Applied Smooth-ConvRot Hadamard rotation for INT4 (group_size={layer_group_size}).")
                    convrot_applied = True
                except Exception as e:
                    warning(f"    - Failed to apply ConvRot for INT4: {e}")

        X_rot, Y_ref, H_mat = None, None, None
        if self.convrot and self.scaling_mode == "row" and convrot_applied:
            from ..utils.tensor_utils import prepare_calibration_data
            X_rot, Y_ref, H_mat = prepare_calibration_data(
                W_float32,
                calibration_data,
                True,
                layer_group_size,
                self.device,
                COMPUTE_DTYPE,
                calib_scale=self.calib_scale,
            )

        row_max = W_float32.abs().amax(dim=1, keepdim=True).clamp_min(1e-10)
        scale = (row_max / 7.0).squeeze(1)
        scaled_int8 = (W_float32 / scale.unsqueeze(1)).round().clamp(-7, 7).to(torch.int8)
        qdata = pack_int4_row_major(scaled_int8)

        if not self.no_learned_rounding and self.num_iter > 0:
            info(f"    - Applying learned rounding optimization for INT4 ({self.scaling_mode}-wise)...")
            if self.convrot and self.scaling_mode == "row" and X_rot is not None:
                if self.scale_optimization == "dualround":
                    info("    - Scale Optimization: DUALROUND for INT4 (Pass 1)")
                    qdata, scale = self._optimize_int4_adaround(W_float32, qdata, scale, X_rot, Y_ref)
                    row_max_opt = W_float32.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
                    scale = (row_max_opt / 7.0).squeeze(1)
                    scaled_int8 = (W_float32 / scale.unsqueeze(1)).round().clamp(-7, 7).to(torch.int8)
                    qdata = pack_int4_row_major(scaled_int8)
                    info("    - Scale Optimization: DUALROUND for INT4 (Pass 2)")
                    qdata, scale = self._optimize_int4_adaround(W_float32, qdata, scale, X_rot, Y_ref)
                else:
                    qdata, scale = self._optimize_int4_adaround(W_float32, qdata, scale, X_rot, Y_ref)

        dequantized_rot = unpack_int4_row_major(qdata).to(COMPUTE_DTYPE) * scale.unsqueeze(1)
        if convrot_applied and layer_group_size is not None:
            H = build_hadamard(layer_group_size, device=self.device, dtype=COMPUTE_DTYPE)
            dequantized = rotate_weight(dequantized_rot, H, layer_group_size)
        else:
            dequantized = dequantized_rot

        if self.has_bias and self.convrot and self.scaling_mode == "row" and X_rot is not None and Y_ref is not None:
            from ..comfy.int4_kernels import quantize_signed_int4_rowwise
            with torch.no_grad():
                if self.w4a4_untouched_activations:
                    act_dequant = X_rot
                    Y_ref_bias = Y_ref
                else:
                    qact, x_scales = quantize_signed_int4_rowwise(X_rot)
                    act_dequant = unpack_int4_row_major(qact).to(COMPUTE_DTYPE) * x_scales.unsqueeze(1)
                    Y_ref_bias = act_dequant @ W_float32.T
                Y_quant = act_dequant @ dequantized_rot.T
                bias_adj = (Y_ref_bias - Y_quant).mean(dim=0)
                self._current_extra_tensors["bias_correction"] = bias_adj.cpu()

        # Quality & Sensitivity Metrics Evaluation
        if X_rot is not None and Y_ref is not None:
            with torch.no_grad():
                if self.w4a4_untouched_activations:
                    eval_act = X_rot
                else:
                    from ..comfy.int4_kernels import quantize_signed_int4_rowwise
                    qact, x_scales = quantize_signed_int4_rowwise(X_rot)
                    eval_act = unpack_int4_row_major(qact).to(COMPUTE_DTYPE) * x_scales.unsqueeze(1)

                Y_sim = eval_act @ dequantized_rot.T
                if "bias_correction" in self._current_extra_tensors:
                    bias_adj = self._current_extra_tensors["bias_correction"].to(device=Y_sim.device, dtype=Y_sim.dtype)
                    Y_sim = Y_sim + bias_adj

                diff = Y_ref - Y_sim
                ref_norm = torch.norm(Y_ref).item()
                err_norm = torch.norm(diff).item()
                snr_db = 20.0 * math.log10(ref_norm / max(err_norm, 1e-12)) if err_norm > 0 else float("inf")
                cos_sim = torch.nn.functional.cosine_similarity(Y_ref.flatten(), Y_sim.flatten(), dim=0).item()
                nmse = (err_norm ** 2) / max(ref_norm ** 2, 1e-12)
                opt_imp = self._last_opt_improvement

                metric_str = f"SNR = {snr_db:.2f} dB, CosSim = {cos_sim:.4f}, NMSE = {nmse:.3e}"
                if opt_imp is not None:
                    metric_str += f", AdaRound Delta = {opt_imp:+.2%}"
                info(f"    - Layer 4-bit metrics: {metric_str}")

                self._current_extra_tensors["metrics"] = {
                    "snr_db": snr_db,
                    "cos_sim": cos_sim,
                    "nmse": nmse,
                    "opt_improvement": opt_imp,
                }

                should_fallback = False
                fallback_reasons = []
                if self.min_snr_db > 0 and snr_db < self.min_snr_db:
                    should_fallback = True
                    fallback_reasons.append(f"SNR {snr_db:.2f} dB < {self.min_snr_db:.2f} dB")
                if self.min_cossim > 0 and cos_sim < self.min_cossim:
                    should_fallback = True
                    fallback_reasons.append(f"CosSim {cos_sim:.4f} < {self.min_cossim:.4f}")
                if self.fallback_unresponsive and opt_imp is not None and opt_imp <= 0 and snr_db < 22.0:
                    should_fallback = True
                    fallback_reasons.append(f"Unresponsive AdaRound ({opt_imp:+.2%}) with SNR {snr_db:.2f} dB")

                if should_fallback:
                    self._current_extra_tensors["fallback"] = {
                        "should_fallback": True,
                        "reasons": fallback_reasons,
                        "snr_db": snr_db,
                        "cos_sim": cos_sim,
                        "nmse": nmse,
                    }

        self._cleanup_tensors(W_float32)
        if X_rot is not None or Y_ref is not None or H_mat is not None:
            self._cleanup_tensors(X_rot, Y_ref, H_mat)

        return qdata, scale.to(device=self.device, dtype=SCALE_DTYPE), dequantized


    def _optimize_int4_adaround(
        self,
        W_float32: torch.Tensor,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        X_rot: torch.Tensor,
        Y_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply SVD-guided AdaRound optimization over rotated parameter space for INT4 W4A4."""
        from ..utils.convrot import pack_int4_row_major, unpack_int4_row_major
        from ..comfy.int4_kernels import quantize_signed_int4_rowwise

        M, N = W_float32.shape

        with torch.no_grad():
            if self.w4a4_untouched_activations:
                act_dequant = X_rot
            else:
                qact, x_scales = quantize_signed_int4_rowwise(X_rot)
                act_dequant = unpack_int4_row_major(qact).to(COMPUTE_DTYPE) * x_scales.unsqueeze(1)
                del qact, x_scales

            # Channel feature-variance weighting for target activations
            out_var = Y_ref.pow(2).mean(dim=0).clamp_min(1e-6)
            channel_weights = (out_var / out_var.mean()).unsqueeze(0)

        U_k, Vh_k, k = self._compute_svd_components(W_float32, verbose=True)

        scale_broadcast = scale.unsqueeze(1) if scale.dim() == 1 else scale
        W_scaled = W_float32 / scale_broadcast.clamp_min(1e-12)
        W_floor = W_scaled.floor().clamp(-7, 7)

        target = (W_scaled - W_floor).clamp_(min=1e-6, max=1.0 - 1e-6)

        T_start, T_end = 20.0, 2.0
        V_init = -torch.log((1.0 / target) - 1.0) * T_start
        V = V_init.clone().detach().requires_grad_(True)
        del W_scaled, target, V_init
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        curr_lr = self.lr
        if self.optimizer_choice == "adamw":
            optimizer = AdamW([V], lr=curr_lr)
        elif self.optimizer_choice == "radam":
            optimizer = RAdam([V], lr=curr_lr)
        elif self.optimizer_choice == "prodigy":
            from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
            optimizer = ProdigyPlusScheduleFree(
                [V], lr=curr_lr, use_schedulefree=False, use_speed=self.use_speed, split_groups=False
            )
        else:
            optimizer = None

        with torch.no_grad():
            init_W_q_rounded = unpack_int4_row_major(qdata).to(COMPUTE_DTYPE)
            init_W_rounded_dequant = init_W_q_rounded * scale_broadcast
            init_mse_rounded = (channel_weights * (act_dequant @ init_W_rounded_dequant.T - Y_ref).pow(2)).mean()
            init_svd_rounded = torch.linalg.norm(U_k.T @ (init_W_rounded_dequant - W_float32) @ Vh_k.T)
            del init_W_q_rounded, init_W_rounded_dequant
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        alpha_svd = 0.01 * (init_mse_rounded.item() / init_svd_rounded.item()) if init_svd_rounded.item() > 1e-8 else 0.0

        schedule_name = self.lr_schedule
        best_loss = float("inf")
        best_V = V.detach().clone()
        best_converged_ratio = 0.0
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0

        effective_patience, effective_factor, effective_cooldown = None, None, None
        if schedule_name == "plateau":
            effective_patience, effective_factor, effective_cooldown = self._compute_shape_aware_plateau_params(M, N)

        decay_factor = self.lr_factor if self.lr_factor is not None else 0.95
        if decay_factor >= 1.0:
            decay_factor = 0.95

        window_size = max(5, int(2.5 / (1.0 - decay_factor)))
        loss_span_threshold = self.early_stop_loss / (1.0 - decay_factor)
        target_converged_ratio = 0.90

        loss_history = []
        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing INT4 (AdaRound-{self.optimizer_choice}-{schedule_name})",
            leave=False,
            dynamic_ncols=True,
        )
        for i in pbar:
            if optimizer is not None:
                optimizer.zero_grad()

            temp = T_start + (T_end - T_start) * (i / self.num_iter)
            h_V = torch.sigmoid(V / temp)
            W_q = (W_floor + h_V).clamp(-7, 7)
            W_dequant = W_q * scale_broadcast

            converged_ratio = ((torch.sigmoid(V) < 0.05) | (torch.sigmoid(V) > 0.95)).float().mean().item()

            Y_pred = act_dequant @ W_dequant.T
            loss_mse = (channel_weights * (Y_pred - Y_ref).pow(2)).mean()

            weight_error = W_dequant - W_float32
            projected_error = U_k.T @ weight_error @ Vh_k.T
            loss_svd = torch.linalg.norm(projected_error)

            # 2-Phase Regularization Penalty Schedule:
            # Phase 1 (0-25% of iterations): beta_reg = 0.0 (unconstrained loss landscape exploration)
            # Phase 2 (25-100% of iterations): smooth cosine ramp from 0.0 -> 0.1
            progress = i / max(self.num_iter, 1)
            if progress < 0.25:
                reg_weight = 0.0
            else:
                phase2_p = (progress - 0.25) / 0.75
                reg_weight = 0.1 * (0.5 * (1.0 - math.cos(math.pi * phase2_p)))

            loss_reg = (1.0 - (2.0 * h_V - 1.0).pow(2)).mean()

            loss_mse_scaled = loss_mse / max(init_mse_rounded.item(), 1e-12)
            loss_svd_scaled = loss_svd / init_svd_rounded.item() if (alpha_svd > 0 and init_svd_rounded.item() > 1e-8) else 0.0

            loss = loss_mse_scaled + 0.01 * loss_svd_scaled + reg_weight * loss_reg

            if self.optimizer_choice == "prodigy":
                if optimizer is not None:
                    loss.backward()
                    optimizer.step()
                else:
                    if V.grad is not None:
                        V.grad.zero_()
                    loss.backward()
                    with torch.no_grad():
                        V -= curr_lr * V.grad
            else:
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
            prev_worse_counter = worse_loss_counter
            improved = self._check_improvement(current_loss_val, best_loss)

            loss_history.append(current_loss_val)
            if len(loss_history) > window_size:
                loss_history.pop(0)

            if converged_ratio >= target_converged_ratio and len(loss_history) == window_size:
                loss_span = max(loss_history) - min(loss_history)
                if loss_span < loss_span_threshold:
                    info(f"\n      - Discretization early stop: {converged_ratio * 100:.2f}% parameters converged.")
                    break

            if improved:
                best_loss = current_loss_val
                best_V = V.detach().clone()
                best_converged_ratio = converged_ratio
                plateau_counter = 0
                if self.lr_adaptive_mode == "simple-reset":
                    worse_loss_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            if self.optimizer_choice != "prodigy":
                if schedule_name == "exponential":
                    curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                    if optimizer is not None:
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = curr_lr
                elif schedule_name == "plateau":
                    if cooldown_counter > 0:
                        cooldown_counter -= 1
                    elif plateau_counter >= effective_patience:
                        if curr_lr > self.lr_min:
                            curr_lr = max(curr_lr * effective_factor, self.lr_min)
                            if optimizer is not None:
                                for param_group in optimizer.param_groups:
                                    param_group["lr"] = curr_lr
                            cooldown_counter = effective_cooldown
                        plateau_counter = 0
            else:
                counter_for_update = prev_worse_counter if improved else worse_loss_counter
                new_lr, lr_updated = self._adaptive_lr_update_cosine(
                    curr_lr, improved, counter_for_update, i, (M, N), self.early_stop_lr
                )
                if lr_updated:
                    curr_lr = new_lr
                    if optimizer is not None:
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = curr_lr
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            pbar.set_postfix({"loss": f"{current_loss_val:.3e}", "best": f"{best_loss:.3e}", "lr": f"{curr_lr:.2e}"})

            if (best_loss <= self.early_stop_loss or curr_lr <= self.early_stop_lr or worse_loss_counter > self.early_stop_stall):
                break

        pbar.close()

        with torch.no_grad():
            best_V_binary = best_V.sigmoid().ge(0.5).float()
            W_quant = (W_floor + best_V_binary).clamp(-7, 7)
            opt_qdata = pack_int4_row_major(W_quant.round().to(torch.int8))

        if init_mse_rounded.item() > 1e-12 and best_loss < float("inf"):
            self._last_opt_improvement = (init_mse_rounded.item() - best_loss) / init_mse_rounded.item()
        else:
            self._last_opt_improvement = 0.0

        if not self.w4a4_untouched_activations:
            self._cleanup_tensors(act_dequant)
        self._cleanup_tensors(U_k, Vh_k, V)
        return opt_qdata, scale

