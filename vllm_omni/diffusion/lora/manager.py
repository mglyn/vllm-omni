# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import OrderedDict
from pathlib import Path
from typing import get_args

import torch
import torch.nn as nn
from safetensors.torch import load_file
from vllm.config.lora import LoRAConfig, MaxLoRARanks
from vllm.logger import init_logger
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest
from vllm.lora.utils import (
    get_adapter_absolute_path,
    get_supported_lora_modules,
    replace_submodule,
)
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear

from vllm_omni.diffusion.lora.plan import (
    DiffusionLoRAApplyPlan,
    DiffusionLoRALoadPlan,
)
from vllm_omni.diffusion.lora.utils import (
    _expand_expected_modules_for_packed_layers,
    _match_target_modules,
    from_layer_diffusion,
)
from vllm_omni.lora.types import (
    LoRAComposition,
    LoRARequestInput,
    LoRAScaleInput,
    WeightedLoRA,
    lora_composition_key,
    normalize_lora_composition,
)
from vllm_omni.lora.utils import stable_lora_int_id

logger = init_logger(__name__)


class DiffusionLoRAManager:
    """Manager for LoRA adapters in diffusion models.

    Reuses vLLM's LoRA infrastructure, adapted for diffusion pipelines.
    Uses LRU cache management similar to LRUCacheLoRAModelManager.
    """

    # Valid max allowed ranks for LoRA in vLLM
    _VALID_MAX_RANKS: list[int] = sorted(get_args(MaxLoRARanks))

    def __init__(
        self,
        pipeline: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        max_cached_adapters: int = 1,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
        prefused_loras: LoRAComposition = (),
        dynamic_loras: LoRAComposition = (),
        quantized: bool = False,
    ):
        """
        Initialize the DiffusionLoRAManager.

        Args:
            max_cached_adapters: Maximum number of LoRA adapters to keep in the
                CPU-side cache (LRU). This mirrors vLLM's `max_cpu_loras` and is
                exposed to users via `OmniDiffusionConfig.max_cpu_loras`.
        """
        self.pipeline = pipeline
        self.device = device
        self.dtype = dtype
        self._apply_plan = self._resolve_apply_plan()

        # Cache supported/expected module suffixes once, before any layer
        # replacement happens. After LoRA layers are injected, the original
        # LinearBase layers become submodules named "*.base_layer", and calling
        # vLLM's get_supported_lora_modules() again would incorrectly yield
        # "base_layer" instead of the real target module suffixes.
        self._supported_lora_modules = self._compute_supported_lora_modules()
        self._packed_modules_mapping = self._compute_packed_modules_mapping()
        self._expected_lora_modules = _expand_expected_modules_for_packed_layers(
            self._supported_lora_modules,
            self._packed_modules_mapping,
        )

        # LRU-style cache management
        startup_dynamic = dynamic_loras
        if lora_path is not None:
            legacy_request = LoRARequest(
                lora_name="static",
                lora_int_id=stable_lora_int_id(lora_path),
                lora_path=lora_path,
            )
            startup_dynamic = normalize_lora_composition(
                tuple(adapter.request for adapter in dynamic_loras) + (legacy_request,),
                tuple(adapter.scale for adapter in dynamic_loras) + (lora_scale,),
            )
        self.max_cached_adapters = max(max_cached_adapters, len(prefused_loras), len(startup_dynamic), 1)
        self._registered_adapters: dict[int, LoRAModel] = {}  # adapter_id -> LoRAModel
        self._adapter_requests: dict[int, LoRARequest] = {}
        self._active_composition: LoRAComposition = ()
        self._default_dynamic_composition: LoRAComposition = ()
        # Compatibility alias for callers/tests that inspect the historical
        # single-adapter state. It is None for zero or multiple adapters.
        self._active_adapter_id: int | None = None
        self._adapter_scales: dict[int, float] = {}  # adapter_id -> external scale

        # LRU cache tracking (adapter_id -> last_used_time)
        self._adapter_access_order: OrderedDict[int, float] = OrderedDict()
        # Pinned adapters are not evicted
        self._pinned_adapters: set[int] = set()

        # track replaced modules
        # key: full module name (component.module.path); value: LoRA layer
        self._lora_modules: dict[str, BaseLayerWithLoRA] = {}
        # Track the maximum LoRA rank we've allocated buffers for.
        self._max_lora_rank: int = 0
        self._frozen = False

        logger.info(
            "Initializing DiffusionLoRAManager: device=%s, dtype=%s, max_cached_adapters=%d, prefused=%d, dynamic=%d",
            device,
            dtype,
            self.max_cached_adapters,
            len(prefused_loras),
            len(startup_dynamic),
        )

        if prefused_loras:
            if quantized:
                raise ValueError("Prefused LoRA is not supported with quantized diffusion weights")
            self._load_composition(prefused_loras)
            self._activate_composition(prefused_loras)
            self._fuse_active_composition()
            for adapter in prefused_loras:
                self.remove_adapter(adapter.adapter_id)

        if startup_dynamic:
            self._load_composition(startup_dynamic)
            self._default_dynamic_composition = startup_dynamic
            self._pinned_adapters.update(adapter.adapter_id for adapter in startup_dynamic)
            self._activate_composition(startup_dynamic)

    def _resolve_apply_plan(self) -> DiffusionLoRAApplyPlan:
        resolver = getattr(self.pipeline, "get_lora_apply_plan", None)
        if callable(resolver):
            plan = resolver()
            if not isinstance(plan, DiffusionLoRAApplyPlan):
                raise TypeError(
                    f"{type(self.pipeline).__name__}.get_lora_apply_plan() must return "
                    f"DiffusionLoRAApplyPlan, got {type(plan)!r}"
                )
            return plan
        return DiffusionLoRAApplyPlan()

    def freeze(self) -> None:
        """Prevent graph-changing layer replacement or buffer growth."""

        self._frozen = True

    def _compute_supported_lora_modules(self) -> set[str]:
        """Compute supported LoRA module suffixes for this pipeline.

        vLLM's get_supported_lora_modules() returns suffixes for LinearBase
        modules. After this manager replaces layers with BaseLayerWithLoRA
        wrappers, those LinearBase modules become nested under ".base_layer",
        which would cause get_supported_lora_modules() to return "base_layer".
        To make adapter loading stable across multiple adapters, we also accept
        suffixes from existing BaseLayerWithLoRA wrappers and drop "base_layer"
        when appropriate.
        """
        supported = set(get_supported_lora_modules(self.pipeline))

        has_lora_wrappers = False
        for name, module in self.pipeline.named_modules():
            if isinstance(module, BaseLayerWithLoRA):
                has_lora_wrappers = True
                supported.add(name.split(".")[-1])

        if has_lora_wrappers:
            supported.discard("base_layer")

        if self._apply_plan.target_modules is not None:
            supported.update(self._apply_plan.target_modules)

        return supported

    def _component_names(self) -> tuple[str, ...]:
        if self._apply_plan.component_names is not None:
            return self._apply_plan.component_names
        default_components = ("transformer", "transformer_2", "dit", "bagel", "unet")
        declared_components = tuple(getattr(self.pipeline, "_dit_modules", ()) or ())
        extra_components = tuple(getattr(self.pipeline, "_lora_components", ()) or ())
        return tuple(dict.fromkeys((*default_components, *declared_components, *extra_components)))

    def _compute_packed_modules_mapping(self) -> dict[str, list[str]]:
        """Collect packed->sublayer mappings from the diffusion model.

        Diffusion models often use packed (fused) projections like `to_qkv` or
        `w13`, while LoRA checkpoints are typically saved against the logical
        sub-projections (e.g. `to_q`/`to_k`/`to_v`, `w1`/`w3`). Many diffusion
        model implementations already define these relationships in
        `load_weights()` via `stacked_params_mapping`. To avoid duplicating the
        mapping in multiple places, we derive packed→sublayer mappings from the
        model's `stacked_params_mapping`.
        """

        def _derive_from_stacked_params_mapping(stacked: object) -> dict[str, list[str]]:
            if not isinstance(stacked, (list, tuple)):
                return {}
            derived: dict[str, list[str]] = {}
            for item in stacked:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                packed_suffix, sub_suffix = item[0], item[1]
                if not isinstance(packed_suffix, str) or not packed_suffix:
                    continue
                if not isinstance(sub_suffix, str) or not sub_suffix:
                    continue
                # The mapping strings are usually suffix patterns (e.g. ".to_qkv"),
                # but some models scope them under submodules (e.g. ".attn1.to_qkv").
                # For LoRA we only care about the leaf module names.
                packed_name = packed_suffix.strip(".").split(".")[-1]
                sub_name = sub_suffix.strip(".").split(".")[-1]
                existing = derived.get(packed_name)
                if existing is None:
                    derived[packed_name] = [sub_name]
                elif sub_name not in existing:
                    existing.append(sub_name)
            return derived

        mapping = {
            packed_name: list(sub_names) for packed_name, sub_names in self._apply_plan.packed_modules_mapping.items()
        }
        for module in self.pipeline.modules():
            explicit = getattr(module, "packed_modules_mapping", None)
            if isinstance(explicit, dict):
                for packed_name, sub_names in explicit.items():
                    if not isinstance(packed_name, str) or not packed_name:
                        continue
                    if not isinstance(sub_names, (list, tuple)) or not all(
                        isinstance(sub_name, str) and sub_name for sub_name in sub_names
                    ):
                        continue
                    mapping.setdefault(packed_name, list(sub_names))

            derived = _derive_from_stacked_params_mapping(getattr(module, "stacked_params_mapping", None))
            for packed_name, sub_names in derived.items():
                if not isinstance(packed_name, str) or not packed_name:
                    continue
                if not isinstance(sub_names, (list, tuple)) or not all(isinstance(s, str) for s in sub_names):
                    continue
                sub_names_list = list(sub_names)
                if not sub_names_list:
                    continue

                existing = mapping.get(packed_name)
                if existing is None:
                    mapping[packed_name] = sub_names_list
                elif existing != sub_names_list:
                    logger.warning(
                        "Conflicting packed module mapping for %s: %s vs %s; using %s",
                        packed_name,
                        existing,
                        sub_names_list,
                        existing,
                    )

        return mapping

    def _get_single_file_load_plan(
        self,
        adapter_path: str,
        tensor_keys: tuple[str, ...],
    ) -> DiffusionLoRALoadPlan:
        providers = [self.pipeline]
        providers.extend(
            component
            for component_name in self._component_names()
            if isinstance((component := getattr(self.pipeline, component_name, None)), nn.Module)
        )
        plans: list[DiffusionLoRALoadPlan] = []
        for provider in providers:
            resolver = getattr(provider, "get_lora_load_plan", None)
            if not callable(resolver):
                continue
            plan = resolver(adapter_path, tensor_keys)
            if plan is None:
                continue
            if not isinstance(plan, DiffusionLoRALoadPlan):
                raise TypeError(
                    f"{type(provider).__name__}.get_lora_load_plan() must return "
                    f"DiffusionLoRALoadPlan or None, got {type(plan)!r}"
                )
            plans.append(plan)

        if not plans:
            raise ValueError(
                "Raw single-file LoRA adapters require the diffusion model to "
                "implement get_lora_load_plan(). Use a PEFT adapter directory "
                "with adapter_config.json instead."
            )
        if any(plan != plans[0] for plan in plans[1:]):
            raise ValueError("Diffusion components returned conflicting LoRA load plans")
        return plans[0]

    @staticmethod
    def _find_single_lora_file(lora_path: str) -> str | None:
        path = Path(lora_path)
        if path.is_file():
            if path.suffix != ".safetensors":
                raise ValueError(f"Raw LoRA file must use safetensors, got {path}.")
            return str(path)
        if not path.is_dir() or (path / "adapter_config.json").is_file():
            return None

        candidates = sorted(path.glob("*.safetensors"))
        if len(candidates) == 1:
            return str(candidates[0])
        if candidates:
            raise ValueError(
                f"LoRA repository {path} contains multiple safetensors files; pass the desired file path explicitly."
            )
        return None

    @staticmethod
    def _infer_single_file_rank(tensors: dict[str, torch.Tensor]) -> int:
        ranks = {int(tensor.shape[0]) for name, tensor in tensors.items() if ".lora_A" in name and tensor.ndim == 2}
        if len(ranks) != 1:
            raise ValueError(f"Raw LoRA must contain one matrix rank, found {sorted(ranks)}.")
        return ranks.pop()

    def _load_single_file_adapter(
        self,
        lora_file: str,
        lora_model_id: int,
    ) -> tuple[LoRAModel, PEFTHelper]:
        tensors = load_file(lora_file, device="cpu")
        plan = self._get_single_file_load_plan(lora_file, tuple(tensors))
        if plan.state_dict_converter is not None:
            tensors = plan.state_dict_converter(tensors)
        rank = self._infer_single_file_rank(tensors)
        config = dict(plan.peft_config)
        config["r"] = rank
        # Raw diffusion LoRAs commonly omit alpha. The neutral interpretation
        # is alpha == rank, so their internal multiplier is one.
        if config.get("lora_alpha") is None:
            config["lora_alpha"] = rank
        peft_helper = PEFTHelper.from_dict(config)
        lora_model = LoRAModel.from_lora_tensors(
            lora_model_id=lora_model_id,
            tensors=tensors,
            peft_helper=peft_helper,
            device="cpu",
            dtype=self.dtype,
            model_vocab_size=None,
            weights_mapper=plan.weights_mapper,
        )

        incomplete = [
            name for name, weights in lora_model.loras.items() if weights.lora_a is None or weights.lora_b is None
        ]
        unexpected = [name for name in lora_model.loras if name.rsplit(".", 1)[-1] not in self._expected_lora_modules]
        if incomplete or unexpected:
            raise ValueError(
                f"Raw LoRA {lora_file} is incompatible with this diffusion model: "
                f"incomplete={incomplete[:3]}, unexpected={unexpected[:3]}."
            )
        return lora_model, peft_helper

    def _get_packed_sublayer_suffixes(self, packed_module_suffix: str, n_slices: int) -> list[str] | None:
        sub_suffixes = self._packed_modules_mapping.get(packed_module_suffix)
        if not sub_suffixes:
            return None
        if len(sub_suffixes) != n_slices:
            logger.warning(
                "Packed module mapping[%s] has %d slices but layer expects %d; skipping sublayer lookup",
                packed_module_suffix,
                len(sub_suffixes),
                n_slices,
            )
            return None
        return sub_suffixes

    def set_active_adapter(
        self,
        lora_request: LoRARequestInput,
        lora_scale: LoRAScaleInput = 1.0,
    ) -> None:
        """Activate one or more request adapters as a single composition.

        An omitted request restores the deployment's default dynamic
        composition. An explicitly supplied zero-scale request disables the
        dynamic contribution for that request; prefused weights remain part of
        the base model.
        """

        composition = (
            self._default_dynamic_composition
            if lora_request is None
            else normalize_lora_composition(lora_request, lora_scale)
        )
        if lora_composition_key(composition) == lora_composition_key(self._active_composition):
            for adapter in composition:
                self._touch_adapter_info(adapter.adapter_id)
            return

        self._load_composition(composition)
        self._deactivate_all_adapters()
        self._activate_composition(composition)

    def _load_composition(self, composition: LoRAComposition) -> None:
        required_ids = {adapter.adapter_id for adapter in composition}
        resident_after_load = set(self._pinned_adapters) | required_ids
        if len(resident_after_load) > self.max_cached_adapters:
            raise ValueError(
                "LoRA composition exceeds the CPU adapter cache: "
                f"required={len(resident_after_load)}, max_cpu_loras={self.max_cached_adapters}"
            )
        previous_pins = set(self._pinned_adapters)
        try:
            # Keep the requested set resident while its later members load.
            # Otherwise LRU eviction may discard an earlier member before the
            # complete composition can be activated.
            for adapter in composition:
                if adapter.adapter_id not in self._registered_adapters:
                    self.add_adapter(adapter.request)
                else:
                    self._touch_adapter_info(adapter.adapter_id)
                self._pinned_adapters.add(adapter.adapter_id)
        finally:
            self._pinned_adapters.intersection_update(previous_pins)

    def _touch_adapter_info(self, adapter_id):
        """Update the current caching ordering info."""
        self._adapter_access_order[adapter_id] = time.time()
        self._adapter_access_order.move_to_end(adapter_id)

    def _update_adapter_scale(self, adapter_id: int, lora_scale: float) -> None:
        self._adapter_scales[adapter_id] = lora_scale

    def _load_adapter(
        self,
        lora_request: LoRARequest,
    ) -> tuple[LoRAModel, PEFTHelper]:
        if not self._expected_lora_modules:
            raise ValueError("No supported LoRA modules found in the diffusion pipeline.")

        logger.debug("Supported LoRA modules: %s", self._expected_lora_modules)

        lora_path = get_adapter_absolute_path(lora_request.lora_path)
        logger.debug("Resolved LoRA path: %s", lora_path)

        lora_file = self._find_single_lora_file(lora_path)
        if lora_file is not None:
            logger.info("Loading raw single-file LoRA from %s", lora_file)
            lora_model, peft_helper = self._load_single_file_adapter(
                lora_file,
                lora_request.lora_int_id,
            )
            for lora in lora_model.loras.values():
                lora.optimize()
            return lora_model, peft_helper

        peft_helper = PEFTHelper.from_local_dir(
            lora_path,
            max_position_embeddings=None,  # no need in diffusion
            tensorizer_config_dict=lora_request.tensorizer_config_dict,
        )

        logger.info(
            "Loaded PEFT config: r=%d, lora_alpha=%d, target_modules=%s",
            peft_helper.r,
            peft_helper.lora_alpha,
            peft_helper.target_modules,
        )

        lora_model = LoRAModel.from_local_checkpoint(
            lora_path,
            expected_lora_modules=self._expected_lora_modules,
            peft_helper=peft_helper,
            lora_model_id=lora_request.lora_int_id,
            device="cpu",  # consistent w/ vllm's behavior
            dtype=self.dtype,
            model_vocab_size=None,
            tensorizer_config_dict=lora_request.tensorizer_config_dict,
            weights_mapper=None,
        )

        logger.info(
            "Loaded LoRA model: id=%d, num_modules=%d, modules=%s",
            lora_model.id,
            len(lora_model.loras),
            list(lora_model.loras.keys()),
        )

        for lora in lora_model.loras.values():
            lora.optimize()  # ref: _create_merged_loras_inplace, internal scaling

        return lora_model, peft_helper

    def _get_packed_modules_list(self, module: nn.Module) -> list[str]:
        """Return a packed_modules_list suitable for vLLM LoRA can_replace_layer().

        Diffusion transformers frequently use packed projection layers like
        QKVParallelLinear (fused QKV). vLLM's LoRA replacement logic relies on
        `packed_modules_list` length to decide between single-slice vs packed
        LoRA layer implementations.
        """
        if isinstance(module, QKVParallelLinear):
            # Treat diffusion QKV as a 3-slice packed projection by default.
            return ["q", "k", "v"]
        if isinstance(module, MergedColumnParallelLinear):
            # 2-slice packed projection (e.g. fused MLP projections).
            return ["0", "1"]
        return []

    def _replace_layers_with_lora(self, peft_helper: PEFTHelper) -> None:
        self._ensure_max_lora_rank(peft_helper.r)

        target_modules = getattr(peft_helper, "target_modules", None)
        target_modules_list: list[str] | None = None
        target_modules_pattern: str | None = None
        if isinstance(target_modules, str) and target_modules:
            target_modules_pattern = target_modules
        elif isinstance(target_modules, list) and target_modules:
            target_modules_list = target_modules

        def _matches_target(module_name: str) -> bool:
            if target_modules_pattern is not None:
                import regex as re

                return re.search(target_modules_pattern, module_name) is not None
            if target_modules_list is None:
                return True
            return _match_target_modules(module_name, target_modules_list)

        # dummy lora config
        lora_config = LoRAConfig(
            max_lora_rank=self._max_lora_rank,
            max_loras=1,
            max_cpu_loras=self.max_cached_adapters,
            lora_dtype=self.dtype,
            fully_sharded_loras=False,
        )

        for component_name in self._component_names():
            if not hasattr(self.pipeline, component_name):
                continue
            component = getattr(self.pipeline, component_name)
            if not isinstance(component, nn.Module):
                continue

            # Collect replacements first to avoid mutating the module tree
            # while iterating over named_modules().
            pending_replacements: list[tuple[str, str, nn.Module, list[str]]] = []

            for module_name, module in component.named_modules(remove_duplicate=False):
                # Don't recurse into already-replaced LoRA wrappers. Their
                # original LinearBase lives under "base_layer", and replacing
                # that again would nest LoRA wrappers and break execution.
                if isinstance(module, BaseLayerWithLoRA) or "base_layer" in module_name.split("."):
                    continue

                full_module_name = f"{component_name}.{module_name}"
                if full_module_name in self._lora_modules:
                    logger.debug("Layer %s already replaced, skipping", full_module_name)
                    continue

                packed_modules_list = self._get_packed_modules_list(module)
                if target_modules_pattern is not None or target_modules_list is not None:
                    should_replace = _matches_target(full_module_name)
                    if not should_replace and len(packed_modules_list) > 1:
                        prefix, _, packed_suffix = full_module_name.rpartition(".")
                        sub_suffixes = self._get_packed_sublayer_suffixes(packed_suffix, len(packed_modules_list))
                        if sub_suffixes is not None:
                            for sub_suffix in sub_suffixes:
                                sub_full_name = f"{prefix}.{sub_suffix}" if prefix else sub_suffix
                                if _matches_target(sub_full_name):
                                    should_replace = True
                                    break

                    if not should_replace:
                        continue

                pending_replacements.append((module_name, full_module_name, module, packed_modules_list))

            for module_name, full_module_name, module, packed_modules_list in pending_replacements:
                if self._frozen:
                    raise ValueError(
                        f"LoRA adapter targets {full_module_name}, which was not installed before "
                        "offload/compile. Preload a compatible adapter with --dynamic-lora."
                    )
                lora_layer = from_layer_diffusion(
                    layer=module,
                    max_loras=1,
                    lora_config=lora_config,
                    packed_modules_list=packed_modules_list,
                    model_config=None,
                )

                if lora_layer is not module and isinstance(lora_layer, BaseLayerWithLoRA):
                    replace_submodule(component, module_name, lora_layer)
                    self._lora_modules[full_module_name] = lora_layer
                    logger.debug("Replaced layer: %s -> %s", full_module_name, type(lora_layer).__name__)

    def _ensure_max_lora_rank(self, min_rank: int, *, reactivate: bool = True) -> None:
        """Ensure LoRA buffers can accommodate adapters up to `min_rank`.

        We allocate per-layer LoRA buffers once when we first replace layers.
        If a later adapter has a larger rank, we need to reinitialize those
        buffers and re-apply the currently active adapter.
        """
        if min_rank <= self._max_lora_rank:
            return

        if self._frozen:
            raise ValueError(
                f"LoRA composition rank {min_rank} exceeds the startup capacity "
                f"{self._max_lora_rank}; preload the full composition with --dynamic-lora"
            )

        valid_max_rank = self._get_smallest_valid_max_rank(min_rank)

        logger.info("Increasing max LoRA rank: %d -> %d", self._max_lora_rank, valid_max_rank)
        self._max_lora_rank = valid_max_rank

        if not self._lora_modules:
            return

        lora_config = LoRAConfig(
            max_lora_rank=self._max_lora_rank,
            max_loras=1,
            max_cpu_loras=self.max_cached_adapters,
            lora_dtype=self.dtype,
            fully_sharded_loras=False,
        )

        # Recreate per-layer buffers with the new maximum rank.
        for lora_layer in self._lora_modules.values():
            lora_layer.create_lora_weights(max_loras=1, lora_config=lora_config, model_config=None)

        # Re-apply active adapter if needed (buffers were reset).
        if reactivate and self._active_composition:
            active = self._active_composition
            self._active_composition = ()
            self._active_adapter_id = None
            self._activate_composition(active)

    @classmethod
    def _get_smallest_valid_max_rank(cls, min_rank: int) -> int:
        """Given a LoRA rank, get the smallest max rank that can support it."""
        if min_rank <= 0:
            raise ValueError(f"Invalid LoRA rank: {min_rank}")

        allowed_ranks = [rank for rank in cls._VALID_MAX_RANKS if rank >= min_rank]
        if not allowed_ranks:
            raise ValueError(f"LoRA rank of {min_rank} exceeds max allowed rank of {max(cls._VALID_MAX_RANKS)}")

        return min(allowed_ranks)

    def _get_lora_weights(
        self,
        lora_model: LoRAModel,
        full_module_name: str,
    ) -> LoRALayerWeights | PackedLoRALayerWeights | None:
        """Best-effort lookup for LoRA weights by name.

        Tries:
        - Full module name (e.g. transformer.blocks.0.attn.to_qkv)
        - Relative name without the top-level component (e.g. blocks.0.attn.to_qkv)
        - Suffix-only name (e.g. to_qkv)
        """
        lora_weights = lora_model.get_lora(full_module_name)
        if lora_weights is not None:
            return lora_weights

        component_relative_name = full_module_name.split(".", 1)[-1] if "." in full_module_name else full_module_name
        lora_weights = lora_model.get_lora(component_relative_name)
        if lora_weights is not None:
            return lora_weights

        module_suffix = full_module_name.split(".")[-1]
        return lora_model.get_lora(module_suffix)

    def _get_layer_slices(
        self,
        lora_model: LoRAModel,
        full_module_name: str,
        lora_layer: BaseLayerWithLoRA,
    ) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
        n_slices = int(getattr(lora_layer, "n_slices", 1))
        lora_weights = self._get_lora_weights(lora_model, full_module_name)
        if isinstance(lora_weights, PackedLoRALayerWeights):
            return list(lora_weights.lora_a), list(lora_weights.lora_b)

        if isinstance(lora_weights, LoRALayerWeights):
            if n_slices == 1:
                return [lora_weights.lora_a], [lora_weights.lora_b]
            # Checkpoint LoRA tensors use global output dimensions. Parallel
            # wrappers expose local ``output_slices`` for their runtime buffers
            # and global ``output_sizes`` for loading; set_lora() performs the
            # rank-local slicing after this packed tensor is split.
            output_sizes = getattr(lora_layer, "output_sizes", None)
            if output_sizes is None:
                output_sizes = getattr(lora_layer, "output_slices", None)
            if output_sizes is None or lora_weights.lora_b.shape[0] != sum(output_sizes):
                raise ValueError(
                    f"Packed LoRA shape mismatch for {full_module_name}: "
                    f"B={tuple(lora_weights.lora_b.shape)}, output_sizes={output_sizes}"
                )
            return [lora_weights.lora_a] * n_slices, list(torch.split(lora_weights.lora_b, list(output_sizes), dim=0))

        if n_slices > 1:
            prefix, _, packed_suffix = full_module_name.rpartition(".")
            sub_suffixes = self._get_packed_sublayer_suffixes(packed_suffix, n_slices)
            if sub_suffixes is not None:
                lora_a: list[torch.Tensor | None] = []
                lora_b: list[torch.Tensor | None] = []
                for sub_suffix in sub_suffixes:
                    sub_name = f"{prefix}.{sub_suffix}" if prefix else sub_suffix
                    sub_lora = self._get_lora_weights(lora_model, sub_name)
                    if isinstance(sub_lora, LoRALayerWeights):
                        lora_a.append(sub_lora.lora_a)
                        lora_b.append(sub_lora.lora_b)
                    else:
                        lora_a.append(None)
                        lora_b.append(None)
                return lora_a, lora_b

        return [None] * n_slices, [None] * n_slices

    def _compose_layer_slices(
        self,
        composition: LoRAComposition,
        full_module_name: str,
        lora_layer: BaseLayerWithLoRA,
    ) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
        n_slices = int(getattr(lora_layer, "n_slices", 1))
        slice_pairs: list[list[tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in range(n_slices)]
        for adapter in composition:
            lora_model = self._registered_adapters[adapter.adapter_id]
            lora_a, lora_b = self._get_layer_slices(lora_model, full_module_name, lora_layer)
            if len(lora_a) != n_slices or len(lora_b) != n_slices:
                raise ValueError(f"LoRA slice count mismatch for {full_module_name}")
            for slice_idx, (a_tensor, b_tensor) in enumerate(zip(lora_a, lora_b, strict=True)):
                if (a_tensor is None) != (b_tensor is None):
                    raise ValueError(
                        f"LoRA adapter {adapter.request.lora_name!r} has an incomplete A/B pair "
                        f"for {full_module_name} slice {slice_idx}"
                    )
                if a_tensor is not None and b_tensor is not None:
                    slice_pairs[slice_idx].append((a_tensor, b_tensor * adapter.scale))

        composed_a: list[torch.Tensor | None] = []
        composed_b: list[torch.Tensor | None] = []
        for slice_idx, pairs in enumerate(slice_pairs):
            if not pairs:
                composed_a.append(None)
                composed_b.append(None)
                continue
            input_dims = {a.shape[1] for a, _ in pairs}
            output_dims = {b.shape[0] for _, b in pairs}
            if len(input_dims) != 1 or len(output_dims) != 1:
                raise ValueError(f"LoRA adapters have incompatible shapes for {full_module_name} slice {slice_idx}")
            composed_a.append(torch.cat([a for a, _ in pairs], dim=0))
            composed_b.append(torch.cat([b for _, b in pairs], dim=1))
        return composed_a, composed_b

    def _activate_composition(self, composition: LoRAComposition) -> None:
        if not composition:
            self._active_composition = ()
            self._active_adapter_id = None
            return

        bound: dict[str, tuple[list[torch.Tensor | None], list[torch.Tensor | None]]] = {}
        required_rank = 0
        for full_module_name, lora_layer in self._lora_modules.items():
            slices = self._compose_layer_slices(composition, full_module_name, lora_layer)
            bound[full_module_name] = slices
            for a_tensor in slices[0]:
                if a_tensor is not None:
                    required_rank = max(required_rank, a_tensor.shape[0])
        if required_rank:
            self._ensure_max_lora_rank(required_rank, reactivate=False)

        for full_module_name, lora_layer in self._lora_modules.items():
            lora_a, lora_b = bound[full_module_name]
            if not any(a is not None for a in lora_a):
                lora_layer.reset_lora(0)
            elif len(lora_a) == 1:
                assert lora_a[0] is not None and lora_b[0] is not None
                lora_layer.set_lora(index=0, lora_a=lora_a[0], lora_b=lora_b[0])
            else:
                lora_layer.set_lora(index=0, lora_a=lora_a, lora_b=lora_b)

        self._active_composition = composition
        self._active_adapter_id = composition[0].adapter_id if len(composition) == 1 else None
        for adapter in composition:
            self._update_adapter_scale(adapter.adapter_id, adapter.scale)
        logger.info("Activated LoRA composition %s", lora_composition_key(composition))

    def _activate_adapter(self, adapter_id: int, scale: float) -> None:
        """Backward-compatible single-adapter activation helper."""

        request = self._adapter_requests.get(adapter_id)
        if request is None:
            request = LoRARequest(
                lora_name=f"adapter-{adapter_id}",
                lora_int_id=adapter_id,
                lora_path=f"registered://{adapter_id}",
            )
        self._activate_composition((WeightedLoRA(request=request, scale=scale),))

    def _fuse_active_composition(self) -> None:
        """Permanently merge the active rank-local delta into dense weights."""

        if not self._active_composition:
            return
        with torch.no_grad():
            for full_module_name, lora_layer in self._lora_modules.items():
                base_layer = getattr(lora_layer, "base_layer", None)
                weight = getattr(base_layer, "weight", None)
                if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or not weight.is_floating_point():
                    raise ValueError(f"Prefused LoRA requires a dense floating-point weight for {full_module_name}")

                active_slices = getattr(lora_layer, "_diffusion_lora_active_slices", None)
                output_slices = tuple(getattr(lora_layer, "output_slices", (weight.shape[0],)))
                offset = 0
                for slice_idx, slice_size in enumerate(output_slices):
                    if active_slices is not None and not active_slices[slice_idx]:
                        offset += slice_size
                        continue
                    a_tensor = lora_layer.lora_a_stacked[slice_idx][0, 0]
                    b_tensor = lora_layer.lora_b_stacked[slice_idx][0, 0]
                    weight_slice = weight[offset : offset + slice_size]
                    merged_weight = weight_slice.float()
                    merged_weight.addmm_(b_tensor.float(), a_tensor.float())
                    weight_slice.copy_(merged_weight)
                    offset += slice_size
        self._deactivate_all_adapters()

    def _deactivate_all_adapters(self) -> None:
        if not self._active_composition:
            logger.debug("All adapters already inactive")
            return
        logger.info("Deactivating all adapters: %d layers", len(self._lora_modules))
        for lora_layer in self._lora_modules.values():
            lora_layer.reset_lora(0)
        self._active_composition = ()
        self._active_adapter_id = None
        logger.debug("All adapters deactivated")

    def _evict_for_new_adapter(self) -> None:
        """Evict unpinned registered adapters until we have room for a new
        adapter to be loaded."""
        while len(self._registered_adapters) > (self.max_cached_adapters - 1):
            # Pick LRU among non-pinned adapters
            evict_candidates = [aid for aid in self._adapter_access_order.keys() if aid not in self._pinned_adapters]
            if not evict_candidates:
                raise ValueError(
                    f"LoRA CPU cache is full ({self.max_cached_adapters}) and all adapters are pinned; "
                    "increase max_cpu_loras or unpin an adapter"
                )

            lru_adapter_id = evict_candidates[0]
            logger.info(
                "Evicting LRU adapter: id=%d (cache: %d/%d)",
                lru_adapter_id,
                len(self._registered_adapters),
                self.max_cached_adapters,
            )
            self.remove_adapter(lru_adapter_id)

    def add_adapter(self, lora_request: LoRARequest) -> bool:
        """
        Add a new adapter to the cache without activating it.
        """
        adapter_id = lora_request.lora_int_id

        if adapter_id in self._registered_adapters:
            existing = self._adapter_requests.get(adapter_id)
            if existing is not None and existing.lora_path != lora_request.lora_path:
                raise ValueError(
                    f"LoRA adapter ID {adapter_id} is already registered from {existing.lora_path!r}, "
                    f"not {lora_request.lora_path!r}"
                )
            logger.debug("Adapter %d already registered, skipping", adapter_id)
            return False

        logger.info("Adding new adapter: id=%d, name=%s", adapter_id, lora_request.lora_name)

        # evict if cache full before adding the new adapter
        # so that we don't go over capacity on the new load
        self._evict_for_new_adapter()

        lora_model, peft_helper = self._load_adapter(lora_request)
        self._replace_layers_with_lora(peft_helper)
        self._registered_adapters[adapter_id] = lora_model
        self._adapter_requests[adapter_id] = lora_request
        self._touch_adapter_info(adapter_id)

        logger.debug(
            "Adapter %d added, cache size: %d/%d", adapter_id, len(self._registered_adapters), self.max_cached_adapters
        )
        return True

    def remove_adapter(self, adapter_id: int) -> bool:
        """
        Remove an adapter from the cache.
        """
        if adapter_id not in self._registered_adapters:
            logger.debug("Adapter %d not found, cannot remove", adapter_id)
            return False

        logger.info("Removing adapter: id=%d", adapter_id)
        if any(adapter.adapter_id == adapter_id for adapter in self._active_composition):
            self._deactivate_all_adapters()

        del self._registered_adapters[adapter_id]
        self._adapter_requests.pop(adapter_id, None)
        self._adapter_scales.pop(adapter_id, None)
        self._adapter_access_order.pop(adapter_id, None)
        self._pinned_adapters.discard(adapter_id)
        logger.debug(
            "Adapter %d removed, cache size: %d/%d",
            adapter_id,
            len(self._registered_adapters),
            self.max_cached_adapters,
        )
        return True

    def list_adapters(self) -> list[int]:
        """Return list of registered adapter ids."""
        return list(self._registered_adapters.keys())

    def pin_adapter(self, adapter_id: int) -> bool:
        """Mark an adapter as pinned so it will not be evicted."""
        if adapter_id not in self._registered_adapters:
            logger.debug("Adapter %d not found, cannot pin", adapter_id)
            return False
        self._pinned_adapters.add(adapter_id)
        # Touch access order so it is most recently used
        self._adapter_access_order[adapter_id] = time.time()
        self._adapter_access_order.move_to_end(adapter_id)
        logger.info("Pinned adapter id=%d (won't be evicted)", adapter_id)
        return True
