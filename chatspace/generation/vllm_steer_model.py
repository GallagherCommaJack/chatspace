"""vLLM-based steerable language model that manages steering inside workers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from vllm import LLM, SamplingParams

from chatspace.vllm_steering import runtime as steering_runtime

from .base import SteerableModel


@dataclass
class VLLMSteeringConfig:
    """Configuration for vLLM-based steerable model."""

    model_name: str = "Qwen/Qwen3-32B"
    target_layer: int = 22
    init_scale: float = 0.0
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    dtype: str = "auto"


def _parse_dtype(dtype_str: str) -> torch.dtype:
    if not dtype_str.startswith("torch."):
        raise ValueError(f"Unexpected dtype format: {dtype_str}")
    name = dtype_str.split(".", maxsplit=1)[1]
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return dtype


class VLLMSteerModel(SteerableModel):
    """Wrap a vLLM LLM with steering vector control via worker RPCs."""

    def __init__(self, cfg: VLLMSteeringConfig, **vllm_kwargs) -> None:
        self.cfg = cfg

        llm_kwargs = {
            "tensor_parallel_size": cfg.tensor_parallel_size,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "dtype": cfg.dtype,
        }
        if cfg.max_model_len is not None:
            llm_kwargs["max_model_len"] = cfg.max_model_len
        llm_kwargs.update(vllm_kwargs)

        steering_runtime.ensure_layer_patch_installed()
        self.llm = LLM(model=cfg.model_name, **llm_kwargs)
        self._engine_client = self.llm.llm_engine.engine_core

        setup_info = self._engine_client.collective_rpc(
            steering_runtime.initialize_worker_state,
            args=(int(cfg.target_layer), float(cfg.init_scale)),
        )
        if not setup_info:
            raise RuntimeError("Failed to initialize steering state on workers.")

        first = setup_info[0]
        self.hidden_size = int(first["hidden_size"])
        self._vector_dtype = _parse_dtype(first["dtype"])
        self._cached_vector = torch.zeros(self.hidden_size, dtype=self._vector_dtype)

    def _broadcast_vector(self, vector: torch.Tensor) -> None:
        vec = vector.to(dtype=self._vector_dtype, device="cpu").contiguous()
        payload = steering_runtime.serialize_tensor(vec)
        self._engine_client.collective_rpc(
            steering_runtime.set_worker_vector, args=(payload,)
        )
        self._cached_vector = vec.clone()

    def _broadcast_clear(self) -> None:
        self._engine_client.collective_rpc(steering_runtime.clear_worker_vector)
        self._cached_vector.zero_()

    def set_target_layer(self, layer_idx: int) -> None:
        layer_idx = int(layer_idx)
        if layer_idx == self.cfg.target_layer:
            return
        self._engine_client.collective_rpc(
            steering_runtime.set_worker_layer, args=(layer_idx,)
        )
        self.cfg.target_layer = layer_idx

    def set_vector(self, vector: torch.Tensor | None) -> None:
        if vector is None:
            self._broadcast_clear()
            return

        vec = vector.detach().view(-1)
        if vec.shape[0] != self.hidden_size:
            raise ValueError(
                f"Steering vector shape mismatch: expected {(self.hidden_size,)}, got {tuple(vec.shape)}"
            )
        self._broadcast_vector(vec)

    def current_vector(self) -> torch.Tensor:
        """Return a CPU copy of the last vector broadcast to workers."""
        return self._cached_vector.clone()

    def generate(
        self,
        prompts: list[str] | str,
        sampling_params: SamplingParams | None = None,
        **kwargs: Any,
    ) -> list[str]:
        if isinstance(prompts, str):
            prompts = [prompts]
        if sampling_params is None:
            sampling_params = SamplingParams(**kwargs)
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]

    def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        sampling_params: SamplingParams | None = None,
        *,
        use_tqdm: bool = False,
        chat_options: dict[str, Any] | None = None,
        prefill_assistant: str | bool | None = None,
        **sampling_kwargs: Any,
    ) -> list[str]:
        """Execute chat-style generation with optional sampling overrides.

        Parameters
        ----------
        messages : list[dict[str, Any]] | list[list[dict[str, Any]]]
            Conversation messages using the OpenAI-style schema. A single
            conversation may be provided (list of messages) or a batch of
            conversations (list of conversation lists).
        sampling_params : SamplingParams | None
            Optional sampling configuration. If omitted, ``sampling_kwargs``
            are used to instantiate a ``SamplingParams`` object.
        use_tqdm : bool, default False
            Whether to display the progress bar during generation.
        chat_options : dict[str, Any] | None
            Additional keyword arguments forwarded to ``LLM.chat`` (for
            example ``chat_template`` or ``add_generation_prompt``).
        prefill_assistant : str | bool | None, default None
            Optional prefix inserted as the final assistant message before
            generation. When set to ``True`` the helper injects an empty
            ``<think></think>`` block compatible with the hybrid chat template.
            String inputs allow custom prefixes; any whitespace-only think blocks
            are normalized to match template formatting. The helper automatically
            strips the prefix (including the sentinel) from returned outputs. Set
            to ``None`` or ``False`` to disable prefilling.
        **sampling_kwargs : Any
            Keyword arguments used to build a ``SamplingParams`` instance when
            ``sampling_params`` is not supplied.
        """
        if sampling_params is None:
            sampling_params = SamplingParams(**sampling_kwargs)
        elif sampling_kwargs:
            raise ValueError(
                "Provide either sampling_params or sampling keyword overrides, not both."
            )

        single_conversation = (
            isinstance(messages, list)
            and (len(messages) == 0 or isinstance(messages[0], dict))
        )
        if single_conversation:
            batched_messages = [cast(list[dict[str, Any]], messages)]
        else:
            batched_messages = cast(list[list[dict[str, Any]]], messages)

        prepared_messages: list[list[dict[str, Any]]]
        chat_kwargs = dict(chat_options or {})
        chat_kwargs.setdefault("chat_template_content_format", "string")

        sentinel = "ASSISTANT_PREFILL:"
        trim_prefix: str | None = None

        def _normalize_prefill(raw: str) -> str:
            stripped = raw.strip()
            if stripped.startswith("<think>") and "</think>" in stripped:
                head, tail = stripped.split("</think>", 1)
                inside = head[len("<think>") :].strip()
                if not inside:
                    tail = tail.lstrip("\n")
                    return "<think>\n\n</think>\n\n" + tail
            return raw

        prefill_base: str | None
        if isinstance(prefill_assistant, bool):
            prefill_base = "<think>\n\n</think>\n\n" if prefill_assistant else None
        elif isinstance(prefill_assistant, str):
            prefill_base = _normalize_prefill(prefill_assistant)
        else:
            prefill_base = None

        if prefill_base is not None:
            prefill_payload = f"{prefill_base}{sentinel}"
            trim_prefix = prefill_payload
            if chat_kwargs.get("add_generation_prompt", False):
                raise ValueError(
                    "Cannot prefill assistant when add_generation_prompt=True. "
                    "Disable prefilling or override chat_options."
                )
            if chat_kwargs.get("continue_final_message") is False:
                raise ValueError(
                    "Cannot prefill assistant when continue_final_message=False. "
                    "Disable prefilling or override chat_options."
                )
            chat_kwargs.setdefault("add_generation_prompt", False)
            chat_kwargs.setdefault("continue_final_message", True)

            prepared_messages = []
            for conv in batched_messages:
                conv_copy = [dict(msg) for msg in conv]
                conv_copy.append({"role": "assistant", "content": prefill_payload})
                prepared_messages.append(conv_copy)
        elif prefill_assistant not in (None, False):
            raise TypeError("prefill_assistant must be a string, boolean, or None.")
        else:
            prepared_messages = batched_messages

        outputs = self.llm.chat(
            prepared_messages,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
            **chat_kwargs,
        )
        texts: list[str] = []
        for output in outputs:
            text = output.outputs[0].text
            if trim_prefix and text.startswith(trim_prefix):
                text = text[len(trim_prefix) :].lstrip()
            texts.append(text)
        return texts

    def save_pretrained(self, save_directory: str | Path, **_) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

        vector_path = path / "steering_vector.pt"
        torch.save({"steering_vector": self._cached_vector.clone()}, vector_path)

        config_path = path / "steering_config.json"
        with config_path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self.cfg), fh, indent=2)

    @classmethod
    def from_pretrained(
        cls, save_directory: str | Path, **vllm_kwargs: Any
    ) -> "VLLMSteerModel":
        path = Path(save_directory)
        config_path = path / "steering_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing steering configuration at {config_path}")
        with config_path.open("r", encoding="utf-8") as fh:
            cfg_dict = json.load(fh)

        cfg = VLLMSteeringConfig(**cfg_dict)
        model = cls(cfg, **vllm_kwargs)

        vector_path = path / "steering_vector.pt"
        if vector_path.exists():
            state = torch.load(vector_path, map_location="cpu")
            tensor = state.get("steering_vector")
            if tensor is None:
                raise ValueError(
                    f"steering_vector.pt missing 'steering_vector' key at {vector_path}"
                )
            model.set_vector(tensor)
        return model

    # ------------------------------------------------------------------
    # Internal debugging helpers (used in tests)
    # ------------------------------------------------------------------

    def _fetch_worker_vectors(self) -> list[torch.Tensor]:
        """Retrieve current worker vectors for validation."""
        payloads = self._engine_client.collective_rpc(
            steering_runtime.fetch_worker_vector
        )
        return [
            steering_runtime.deserialize_tensor(
                payload, device=torch.device("cpu"), dtype=self._vector_dtype
            )
            for payload in payloads
        ]
