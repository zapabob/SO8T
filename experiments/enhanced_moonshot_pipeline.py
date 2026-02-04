# -*- coding: utf-8 -*-
"""
Enhanced Moonshot Pipeline (cleaned)

Purpose:
- Provide a stable, minimal implementation used by the integrated 2025–2026 pipeline.
- Keep hooks for mHC / GRPO / SO8T / GRAPE / imatrix / FlashAttention.
- Avoid mojibake and keep Windows-safe defaults.
"""

from __future__ import annotations

import gc
import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available

try:
    from src.models.grape_position_encoding import (
        GrapePatchConfig,
        patch_rotary_embeddings,
        enable_additive_grape,
        patch_attention_with_additive_grape,
    )
except Exception:
    GrapePatchConfig = None
    patch_rotary_embeddings = None
    enable_additive_grape = None
    patch_attention_with_additive_grape = None

try:
    from src.models.so8t_residual_adapter import inject_nkat_to_all_layers
except Exception:
    inject_nkat_to_all_layers = None

try:
    from src.models.mhc_manifold import apply_mhc_projection_to_model
except Exception:
    apply_mhc_projection_to_model = None

try:
    from src.utils.artifact_qa import collect_artifacts, summarize_artifacts, write_report
except Exception:
    collect_artifacts = None
    summarize_artifacts = None
    write_report = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class EnhancedMoonshotPipeline:
    def __init__(self, boreas_model_path: str, output_root: Optional[str] = None) -> None:
        self.boreas_model_path = boreas_model_path
        self.output_root = Path(output_root) if output_root else Path("models")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.tokenizer = None

    def _maybe_run_unsloth(self, phase: str) -> bool:
        """Optionally run the Unsloth training script for a given phase."""
        if os.getenv("SO8T_USE_UNSLOTH") != "1":
            return False
        if os.getenv("SO8T_DRYRUN") == "1":
            logger.info("Dry-run mode: skipping Unsloth %s phase", phase)
            return False
        script = Path("scripts/training/train_unsloth_so8t.py")
        if not script.exists():
            logger.warning("Unsloth training script not found: %s", script)
            return False

        cmd = ["py", "-3", str(script), "--phase", phase]
        if os.getenv("SO8T_MCP_API_SKILL") == "1":
            cmd.append("--mcp-api-skill")
        if os.getenv("SO8T_RECOVER") == "1":
            cmd.append("--recover")
        if os.getenv("SO8T_TRAINING_CONFIG"):
            cmd.extend(["--config", os.getenv("SO8T_TRAINING_CONFIG")])

        logger.info("Running Unsloth training: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("Unsloth phase %s failed: %s", phase, exc)
            return False

    # ------------------------------------------------------------------
    # Model load / cleanup
    # ------------------------------------------------------------------
    def load_boreas_model(self) -> None:
        """Load base model and enable FlashAttention2 if available."""
        if os.getenv("SO8T_DRYRUN") == "1" or os.getenv("SO8T_SKIP_MODEL_LOAD") == "1":
            logger.info("Dry-run mode: skipping model load")
            return
        logger.info("Loading base model: %s", self.boreas_model_path)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.boreas_model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.boreas_model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            # Optional Unsloth patch (if available)
            try:
                from unsloth import FastLanguageModel

                self.model = FastLanguageModel.for_inference(self.model)
                logger.info("Unsloth patch applied")
            except Exception as exc:
                logger.info("Unsloth patch skipped: %s", exc)
            if is_flash_attn_2_available():
                try:
                    self.model.config.use_flash_attention_2 = True
                    logger.info("FlashAttention-2 enabled")
                except Exception as exc:
                    logger.warning("FlashAttention-2 enable failed: %s", exc)
        except Exception as exc:
            logger.warning("Model load skipped (non-fatal): %s", exc)

    def _cleanup_resources(self) -> None:
        """Release GPU/CPU memory safely."""
        try:
            self.model = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning("Cleanup warning: %s", exc)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _touch_marker(self, name: str) -> Path:
        marker = self.output_root / f"{name}.done"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok", encoding="utf-8")
        return marker

    def _run_artifact_qa(self, label: str, directory: Path, patterns: List[str]) -> Optional[Path]:
        if collect_artifacts is None or summarize_artifacts is None or write_report is None:
            logger.debug("Artifact QA utilities unavailable")
            return None
        if not directory.exists():
            logger.warning("QA directory missing: %s", directory)
            return None
        artifacts = collect_artifacts(directory, patterns)
        if not artifacts:
            logger.warning("No artifacts found for QA in %s", directory)
            return None
        report = {
            "label": label,
            "directory": str(directory),
            "artifacts": summarize_artifacts(artifacts),
        }
        report_path = Path("results") / "qa" / f"{label}_report.json"
        write_report(report_path, report)
        logger.info("Artifact QA report written: %s", report_path)
        return report_path

    # ------------------------------------------------------------------
    # Training / integration stubs (safe for orchestration)
    # ------------------------------------------------------------------
    def execute_sft_rlpo_integration(self, target_datasets: Optional[List[Path]] = None) -> None:
        logger.info("SFT/RLPO integration (stub) starting")
        if self._maybe_run_unsloth("sft"):
            self._touch_marker("sft_rlpo")
            logger.info("SFT/RLPO integration completed via Unsloth")
            return
        if target_datasets:
            logger.info("Using %d dataset files", len(target_datasets))
        out_dir = Path("models/aegis_v25_rlpo")
        out_dir.mkdir(parents=True, exist_ok=True)
        self._touch_marker("sft_rlpo")
        logger.info("SFT/RLPO integration completed (stub)")

    def execute_so8_residual_adapter_retraining(self) -> None:
        logger.info("SO8 residual adapter retraining (stub)")
        if os.getenv("SO8T_SO8_ENABLE") == "1" and inject_nkat_to_all_layers is not None:
            if self.model is None:
                logger.warning("SO8 adapter integration skipped (model not loaded)")
            else:
                mode = os.getenv("SO8T_SO8_MODE", "mlp_only")
                target_layers = os.getenv("SO8T_SO8_LAYERS", "middle")
                try:
                    inject_nkat_to_all_layers(self.model, target_layers=target_layers, mode=mode)
                    logger.info("SO8 residual adapters injected (mode=%s, layers=%s)", mode, target_layers)
                except Exception as exc:
                    logger.warning("SO8 residual adapter injection failed: %s", exc)
        self._touch_marker("so8_residual")

    def execute_deepseek_grpo_integration(self) -> None:
        logger.info("DeepSeek GRPO integration (stub)")
        if self._maybe_run_unsloth("grpo"):
            self._touch_marker("grpo")
            logger.info("DeepSeek GRPO integration completed via Unsloth")
            return
        self._touch_marker("grpo")

    def execute_mhc_manifold_integration(self) -> None:
        logger.info("mHC manifold integration (stub)")
        if os.getenv("SO8T_MHC_ENABLE") == "1" and apply_mhc_projection_to_model is not None:
            if self.model is None:
                logger.warning("mHC integration skipped (model not loaded)")
            else:
                targets_env = os.getenv("SO8T_MHC_TARGETS", "o_proj,down_proj,up_proj,gate_proj")
                targets = [t.strip() for t in targets_env.split(",") if t.strip()]
                blend = float(os.getenv("SO8T_MHC_BLEND", "0.1"))
                max_iter = int(os.getenv("SO8T_MHC_MAX_ITER", "20"))
                try:
                    updated = apply_mhc_projection_to_model(
                        self.model,
                        target_modules=targets,
                        max_iter=max_iter,
                        blend=blend,
                    )
                    logger.info("mHC projection applied to %d modules", len(updated))
                    report_path = Path("results") / "qa" / "mhc_projection_report.json"
                    if write_report is not None:
                        write_report(
                            report_path,
                            {
                                "targets": targets,
                                "blend": blend,
                                "max_iter": max_iter,
                                "updated_modules": updated,
                            },
                        )
                        logger.info("mHC projection report: %s", report_path)
                except Exception as exc:
                    logger.warning("mHC projection failed: %s", exc)
        self._touch_marker("mhc")

    def execute_grape_position_encoding(self, variant: str = "multiplicative") -> None:
        logger.info("GRAPE position encoding (variant=%s)", variant)
        if self.model is None or patch_rotary_embeddings is None or GrapePatchConfig is None:
            logger.warning("GRAPE patch skipped (model or patch module unavailable)")
            return
        try:
            config = GrapePatchConfig(variant=variant)
            variant_lower = (variant or "multiplicative").lower()
            did_patch = False
            if variant_lower in {"multiplicative", "commuting_ms_grape", "hybrid"}:
                patch_rotary_embeddings(self.model, config)
                did_patch = True
            if variant_lower in {"additive", "alibi", "fox", "hybrid"}:
                if enable_additive_grape is not None:
                    enable_additive_grape(self.model, config)
                    if patch_attention_with_additive_grape is not None:
                        patch_attention_with_additive_grape(self.model, config)
                    did_patch = True
            if did_patch:
                marker = "grape_additive" if variant_lower in {"additive", "alibi", "fox"} else "grape"
                self._touch_marker(marker)
        except Exception as exc:
            logger.warning("GRAPE patch failed: %s", exc)

    def execute_geometric_scaling_integration(self) -> None:
        logger.info("Geometric scaling integration (stub)")
        self._touch_marker("geometric_scaling")

    # ------------------------------------------------------------------
    # imatrix / GGUF
    # ------------------------------------------------------------------
    def execute_so8t_imatrix_quantization(self) -> None:
        """Run imatrix GGUF conversion via existing script if available."""
        logger.info("SO8T imatrix quantization")
        if os.getenv("SO8T_DRYRUN") == "1":
            logger.info("Dry-run mode: skipping imatrix conversion")
            self._touch_marker("imatrix")
            return
        script = Path("scripts/conversion/convert_aegis_v22_with_imatrix.py")
        if not script.exists():
            logger.warning("imatrix conversion script not found: %s", script)
            return

        model_dir = Path("models/aegis_v25_rlpo")
        if not model_dir.exists():
            model_dir = Path(self.boreas_model_path)

        output_dir = Path("H:/from_D/webdataset/gguf_models/aegis_v25_imatrix")
        output_dir.mkdir(parents=True, exist_ok=True)

        calib = output_dir / "imatrix_calibration.txt"
        if not calib.exists():
            calib.write_text("calibration_placeholder\n", encoding="utf-8")

        cmd = [
            "py",
            "-3",
            str(script),
            "--model-dir",
            str(model_dir),
            "--output-dir",
            str(output_dir),
            "--calibration",
            str(calib),
        ]
        try:
            subprocess.run(cmd, check=True)
            self._touch_marker("imatrix")
            self._run_artifact_qa("imatrix", output_dir, ["*.gguf", "*.bin", "*.json", "*.txt"])
        except subprocess.CalledProcessError as exc:
            logger.error("imatrix conversion failed: %s", exc)

    def execute_bf16_gguf_conversion(self) -> None:
        logger.info("BF16 GGUF conversion (stub)")
        output_dir = Path(os.getenv("SO8T_BF16_GGUF_DIR", str(self.output_root / "bf16_gguf")))
        if output_dir.exists():
            self._run_artifact_qa("bf16_gguf", output_dir, ["*.gguf"])
        self._touch_marker("bf16_gguf")

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------
    def execute_hf_upload_automation(self) -> None:
        logger.info("HF upload automation (stub)")
        self._touch_marker("hf_upload")
