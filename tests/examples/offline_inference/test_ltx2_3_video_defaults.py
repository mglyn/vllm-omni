# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from examples.offline_inference.video_model_defaults import (
    default_image_to_video_class_name,
    is_ltx2_model,
    is_ltx23_model,
)


def test_ltx23_model_name_detection_accepts_hyphenated_model_ids():
    assert is_ltx23_model("dg845/LTX-2.3-Diffusers")
    assert is_ltx23_model("/models/ltx23-local")
    assert not is_ltx23_model("Lightricks/LTX-2.3")
    assert not is_ltx2_model("Lightricks/LTX-2.3")


def test_ltx23_image_to_video_defaults_select_i2v_pipeline():
    assert default_image_to_video_class_name("dg845/LTX-2.3-Diffusers") == "LTX23ImageToVideoPipeline"
    assert default_image_to_video_class_name("Lightricks/LTX-2.3") is None
    assert default_image_to_video_class_name("Lightricks/LTX-2") == "LTX2ImageToVideoPipeline"
