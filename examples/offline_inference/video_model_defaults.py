# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

_LTX23_RAW_CHECKPOINT_MODEL_IDS = {
    "lightricks/ltx-2.3",
}


def _normalize_model_name(value: str | None) -> str:
    return (value or "").lower().replace("-", "").replace("_", "").replace(".", "")


def _is_unsupported_ltx23_raw_checkpoint_model(model: str | None) -> bool:
    return model is not None and model.lower() in _LTX23_RAW_CHECKPOINT_MODEL_IDS


def is_ltx23_model(model: str | None, model_class_name: str | None = None) -> bool:
    if _is_unsupported_ltx23_raw_checkpoint_model(model) and model_class_name is None:
        return False
    model_key = _normalize_model_name(model)
    class_key = _normalize_model_name(model_class_name)
    return "ltx23" in model_key or "ltx23" in class_key


def is_ltx2_model(model: str | None, model_class_name: str | None = None) -> bool:
    if _is_unsupported_ltx23_raw_checkpoint_model(model) and model_class_name is None:
        return False
    model_key = _normalize_model_name(model)
    class_key = _normalize_model_name(model_class_name)
    return is_ltx23_model(model, model_class_name) or "ltx2" in model_key or "ltx2" in class_key


def default_image_to_video_class_name(model: str | None, model_class_name: str | None = None) -> str | None:
    if model_class_name is not None:
        return model_class_name
    if is_ltx23_model(model):
        return "LTX23ImageToVideoPipeline"
    if is_ltx2_model(model):
        return "LTX2ImageToVideoPipeline"
    return None
