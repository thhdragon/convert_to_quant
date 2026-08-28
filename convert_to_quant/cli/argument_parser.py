"""
CLI argument parser for convert_to_quant.

Provides MultiHelpArgumentParser with categorized help sections
for INT4 ConvRot W4A4 options, filter options, and advanced parameters.
"""

import argparse
import sys

from ..constants import MODEL_FILTERS

# --- CLI Help Sections ---
EXPERIMENTAL_ARGS = {
    "int4",
    "convrot",
    "convrot_group_size",
    "dynamic_convrot",
    "w4a4_untouched_activations",
    "smooth_convrot",
    "smooth_alpha",
    "custom_layers",
    "custom_block_size",
    "custom_simple",
    "custom_heur",
    "custom_full_precision_mm",
    "custom_convrot",
    "custom_convrot_group_size",
    "layer_config",
    "layer_config_fullmatch",
    "exclude_layers",
    "block_size",
    "no_normalize_scales",
    "resume",
    "sidecar_path",
    "max_shard_size",
    "no_checkpoint",
}

FILTER_ARGS = set(MODEL_FILTERS.keys())

ADVANCED_ARGS = {
    "lr_gamma",
    "lr_patience",
    "lr_factor",
    "lr_min",
    "lr_cooldown",
    "lr_threshold",
    "lr_adaptive_mode",
    "lr_shape_influence",
    "lr_threshold_mode",
    "early_stop_loss",
    "early_stop_lr",
    "early_stop_stall",
    "scale_optimization",
}

LEARNED_ROUNDING_ARGS = {
    "full_matrix",
    "calib_samples",
    "calib_cpu",
    "optimizer",
    "num_iter",
    "lr",
    "lr_schedule",
    "top_p",
    "min_k",
    "max_k",
}

MODES_ARGS = {
    "actcal",
    "actcal_samples",
    "actcal_percentile",
    "actcal_lora",
    "actcal_seed",
    "actcal_device",
    "edit_quant",
    "remove_keys",
    "add_keys",
    "quant_filter",
    "dry_run",
    "full_precision_mm",
    "calib_data",
}


LORA_ARGS = {"extract_lora", "lora_rank", "lora_target", "lora_depth", "lora_ar_threshold", "lora_output"}


class MultiHelpArgumentParser(argparse.ArgumentParser):
    """ArgumentParser with multiple help sections."""

    def __init__(
        self, *args, experimental_args=None, filter_args=None, advanced_args=None, learned_rounding_args=None, modes_args=None,
        lora_args=None, **kwargs
    ):
        self._experimental_args = experimental_args or set()
        self._filter_args = filter_args or set()
        self._advanced_args = advanced_args or set()
        self._learned_rounding_args = learned_rounding_args or set()
        self._modes_args = modes_args or set()
        self._lora_args = lora_args or set()
        self._all_actions = []
        super().__init__(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        action = super().add_argument(*args, **kwargs)
        if hasattr(self, "_all_actions"):
            self._all_actions.append(action)
        return action

    def parse_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]

        if "--help-learned" in args or "-hl" in args:
            self._print_learned_help()
            sys.exit(0)
        elif "--help-experimental" in args or "-he" in args:
            self._print_experimental_help()
            sys.exit(0)
        elif "--help-filters" in args or "-hf" in args:
            self._print_filters_help()
            sys.exit(0)
        elif "--help-advanced" in args or "-ha" in args:
            self._print_advanced_help()
            sys.exit(0)
        elif "--help-modes" in args or "-hm" in args:
            self._print_modes_help()
            sys.exit(0)
        elif "--help-lora" in args or "-hlr" in args:
            self._print_lora_help()
            sys.exit(0)
        return super().parse_args(args, namespace)

    def _get_dest_name(self, action):
        return action.dest

    def _format_action_help(self, action):
        opts = ", ".join(action.option_strings) if action.option_strings else action.dest
        help_text = action.help or ""
        if help_text == argparse.SUPPRESS:
            return None

        if action.default is not None and action.default != argparse.SUPPRESS:
            if action.default is not False and action.default != "":
                if isinstance(action.default, str):
                    help_text += f" (default: '{action.default}')"
                else:
                    help_text += f" (default: {action.default})"

        if action.choices:
            choices_str = ", ".join(str(c) for c in action.choices)
            help_text += f" [choices: {choices_str}]"

        return f"  {opts:30s} {help_text}"

    def _print_learned_help(self):
        print("Learned Rounding Optimization Options (INT4 ConvRot W4A4)")
        print("=" * 60)
        print()
        for action in self._all_actions:
            if self._get_dest_name(action) in self._learned_rounding_args:
                line = self._format_action_help(action)
                if line:
                    print(line)
        print()

    def _print_experimental_help(self):
        print("INT4 ConvRot W4A4 Quantization Features")
        print("=" * 60)
        print()
        for action in self._all_actions:
            if self._get_dest_name(action) in self._experimental_args:
                line = self._format_action_help(action)
                if line:
                    print(line)
        print()

    def _print_filters_help(self):
        print("Model-Specific Exclusion Filters")
        print("=" * 60)
        print()
        for name, cfg in MODEL_FILTERS.items():
            print(f"  --{name:25s} {cfg.get('help', '')}")
        print()

    def _print_advanced_help(self):
        print("Advanced LR Tuning & Early Stopping Options")
        print("=" * 60)
        print()
        for action in self._all_actions:
            if self._get_dest_name(action) in self._advanced_args:
                line = self._format_action_help(action)
                if line:
                    print(line)
        print()

    def _print_modes_help(self):
        print("Utility & Calibration Modes")
        print("=" * 60)
        print()
        for action in self._all_actions:
            if self._get_dest_name(action) in self._modes_args:
                line = self._format_action_help(action)
                if line:
                    print(line)
        print()

    def _print_lora_help(self):
        print("Error Correction LoRA Options")
        print("=" * 60)
        print()
        for action in self._all_actions:
            if self._get_dest_name(action) in self._lora_args:
                line = self._format_action_help(action)
                if line:
                    print(line)
        print()

    def format_help(self):
        formatter = self._get_formatter()

        standard_actions = []
        for action in self._actions:
            dest = self._get_dest_name(action)
            if dest not in self._experimental_args and dest not in self._filter_args and dest not in self._advanced_args and dest not in self._modes_args and dest not in self._learned_rounding_args and dest not in self._lora_args:
                standard_actions.append(action)

        formatter.add_usage(self.usage, standard_actions, self._mutually_exclusive_groups)
        formatter.add_text(self.description)

        formatter.start_section("Standard Options")
        formatter.add_arguments(standard_actions)
        formatter.end_section()

        formatter.add_text("")
        formatter.add_text("Additional Help Sections:")
        formatter.add_text("  --help-learned, -hl         Show learned rounding optimization options")
        formatter.add_text("  --help-experimental, -he    Show INT4 ConvRot options")
        formatter.add_text("  --help-filters, -hf         Show model-specific exclusion filters")
        formatter.add_text("  --help-advanced, -ha        Show advanced LR tuning and early stopping")
        formatter.add_text("  --help-modes, -hm           Show calibration and utility modes")
        formatter.add_text("  --help-lora, -hlr           Show Error Correction LoRA extraction options")
        formatter.add_text(self.epilog)

        return formatter.format_help()
