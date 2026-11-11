#!/usr/bin/env python3
"""SFT training script for Qwen2.5-VL EnactGraph SLM."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_sft import (  # noqa: E402
    SafeWandbCallback,
    _configure_visible_devices,
    _train_test_split_with_optional_strata,
)

import torch
import yaml
from datasets import load_from_disk
from PIL import Image
from transformers import (
    AutoProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from nsdl.image_pixel_cap import apply_image_pixel_cap  # noqa: E402
from nsdl.path_remap import remap_path  # noqa: E402
from nsdl.vl_data_contract import validate_vl_sft_dataset_contract  # noqa: E402


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _fsdp_enabled(train_cfg: dict[str, Any]) -> bool:
    fsdp = train_cfg.get("fsdp")
    if isinstance(fsdp, str):
        return bool(fsdp.strip())
    return bool(fsdp)


def _distributed_world_size() -> int:
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


def _visible_cuda_device_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        visible = visible.strip()
        if not visible:
            return 0
        return len([part for part in visible.split(",") if part.strip()])
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def _maybe_reexec_with_torchrun(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    train_cfg = cfg.get("training", {})
    if not _fsdp_enabled(train_cfg):
        return
    if _distributed_world_size() > 1:
        return
    if not args.auto_torchrun:
        raise SystemExit(
            "This config enables FSDP, but the process is not distributed. "
            "Launch with torchrun, for example:\n"
            "  python -m torch.distributed.run --standalone --nproc_per_node=2 "
            f"{Path(__file__).as_posix()} --config {args.config}"
        )

    nproc = args.torchrun_nproc_per_node or _visible_cuda_device_count()
    if nproc < 2:
        raise SystemExit(
            "This config enables FSDP, which needs distributed multi-GPU training. "
            f"Only {nproc} CUDA device(s) are visible to this process. "
            "Use a multi-GPU visibility setting such as '--device 0,1', or unset "
            "CUDA_VISIBLE_DEVICES, then rerun this command."
        )

    env = os.environ.copy()
    env["NSDL_TORCHRUN_REEXEC"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    print(
        "FSDP config detected in a non-distributed process; "
        f"relaunching with torchrun on {nproc} GPUs.",
        flush=True,
    )
    os.execve(sys.executable, cmd, env)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif not isinstance(item, dict):
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(content) if content is not None else ""


class VisionChatDataCollator:
    """Build Qwen2.5-VL batches lazily from text messages and image paths."""

    def __init__(
        self,
        processor,
        *,
        max_seq_len: int,
        goal_loss_weight: float = 3.0,
        plan_loss_weight: float = 1.0,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_seq_len = int(max_seq_len)
        self.goal_loss_weight = max(0.0, float(goal_loss_weight))
        self.plan_loss_weight = max(0.0, float(plan_loss_weight))

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        processed = [self._process_feature(feature) for feature in features]
        batch = self.tokenizer.pad(
            [
                {
                    "input_ids": item["input_ids"],
                    "attention_mask": item["attention_mask"],
                }
                for item in processed
            ],
            padding=True,
            return_tensors="pt",
        )
        max_len = int(batch["input_ids"].shape[1])
        labels = []
        loss_weights = []
        for item in processed:
            row_labels = list(item["labels"])
            row_weights = list(item["loss_weights"])
            if len(row_labels) > max_len:
                row_labels = row_labels[:max_len]
            if len(row_weights) > max_len:
                row_weights = row_weights[:max_len]
            labels.append(row_labels + [-100] * (max_len - len(row_labels)))
            loss_weights.append(row_weights + [0.0] * (max_len - len(row_weights)))
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        batch["loss_weights"] = torch.tensor(loss_weights, dtype=torch.float32)

        extra_keys = sorted(
            {
                key
                for item in processed
                for key in item["extras"]
            }
        )
        for key in extra_keys:
            tensors = [item["extras"][key] for item in processed if key in item["extras"]]
            if tensors:
                batch[key] = torch.cat(tensors, dim=0)
        return batch

    def _process_feature(self, feature: dict[str, Any]) -> dict[str, Any]:
        image_path = Path(remap_path(feature["image_path"]))
        if not image_path.exists():
            raise FileNotFoundError(f"VL SFT image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")

        crops_by_message = self._load_crop_attachments(feature)
        messages, images = self._messages_with_image(
            feature["messages"], image, crops_by_message
        )
        full_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        full_inputs = self.processor(
            text=[full_text],
            images=images,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        input_ids = full_inputs["input_ids"][0].tolist()
        attention_mask = full_inputs["attention_mask"][0].tolist()
        labels, loss_weights = self._assistant_only_labels(
            messages=messages,
            full_text=full_text,
            input_ids=input_ids,
            images=images,
        )

        extras = {
            key: value
            for key, value in full_inputs.items()
            if key not in {"input_ids", "attention_mask"} and isinstance(value, torch.Tensor)
        }
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_weights": loss_weights,
            "extras": extras,
        }

    @staticmethod
    def _load_crop_attachments(feature: dict[str, Any]) -> dict[int, list[Image.Image]]:
        """P5.10.3 v3: crop pixels referenced by crop_attachments_json.

        Returns ``message_index -> [crop images in attachment order]``. Rows
        without the column (v1/v2 datasets) or with an empty list load nothing.
        """
        raw = feature.get("crop_attachments_json")
        if not raw:
            return {}
        attachments = json.loads(raw)
        crops: dict[int, list[Image.Image]] = {}
        for att in attachments:
            if not isinstance(att, dict):
                continue
            message_index = att.get("message_index")
            crop_path = att.get("image_path")
            if message_index is None or not crop_path:
                continue
            path = Path(remap_path(crop_path))
            if not path.exists():
                raise FileNotFoundError(f"VL SFT crop image not found: {path}")
            crops.setdefault(int(message_index), []).append(
                Image.open(path).convert("RGB")
            )
        return crops

    @staticmethod
    def _messages_with_image(
        messages: list[dict[str, Any]],
        image: Image.Image,
        crops_by_message: dict[int, list[Image.Image]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[Image.Image]]:
        """Render messages for the processor and collect images in order.

        Mirrors inference-side ``local_vl._build_messages`` byte-for-byte:
        the scene image rides the first user turn, and crop images ride their
        tool turn (rendered as user) with image parts BEFORE the text part.
        Without crops the rendered text is identical to the legacy behavior.
        """
        crops_by_message = crops_by_message or {}
        vl_messages: list[dict[str, Any]] = []
        images: list[Image.Image] = []
        image_added = False
        for message_index, message in enumerate(messages):
            role = str(message.get("role", "user"))
            content_text = _content_to_text(message.get("content", ""))
            if role == "tool":
                tool_name = str(message.get("name") or "tool").strip() or "tool"
                role = "user"
                content_text = f"[TOOL_RESULT {tool_name}]\n{content_text}"
            attach_scene = role == "user" and not image_added
            message_crops = crops_by_message.get(message_index, [])
            if attach_scene or message_crops:
                parts: list[dict[str, Any]] = []
                if attach_scene:
                    content_text = content_text.replace("<image>", "").strip()
                    parts.append({"type": "image", "image": image})
                    images.append(image)
                    image_added = True
                for crop in message_crops:
                    parts.append({"type": "image", "image": crop})
                    images.append(crop)
                parts.append({"type": "text", "text": content_text})
                vl_messages.append({"role": role, "content": parts})
            else:
                vl_messages.append({"role": role, "content": content_text})
        return vl_messages, images

    def _assistant_only_labels(
        self,
        *,
        messages: list[dict[str, Any]],
        full_text: str,
        input_ids: list[int],
        images: list[Image.Image],
    ) -> tuple[list[int], list[float]]:
        labels = [-100] * len(input_ids)
        loss_weights = [0.0] * len(input_ids)
        search_pos = 0
        any_span = False
        latest_goal_active_items: set[str] | None = None
        for message in messages:
            if str(message.get("role")) != "assistant":
                continue
            content_text = _content_to_text(message.get("content", ""))
            if not content_text:
                continue
            start = full_text.find(content_text, search_pos)
            if start < 0:
                continue
            end = start + len(content_text)
            search_pos = end
            if bool(message.get("loss_masked")):
                continue
            start_tok = self._token_count_for_prefix(full_text[:start], images)
            end_tok = self._token_count_for_prefix(full_text[:end], images)
            start_tok = min(start_tok, len(input_ids))
            end_tok = min(end_tok, len(input_ids))
            if end_tok <= start_tok:
                continue
            any_span = True
            labels[start_tok:end_tok] = input_ids[start_tok:end_tok]
            for pos in range(start_tok, end_tok):
                loss_weights[pos] = 1.0
            content_goal_active_items = self._latest_goal_active_items(content_text)
            if content_goal_active_items is not None:
                latest_goal_active_items = content_goal_active_items
            if self.goal_loss_weight != 1.0:
                self._apply_span_loss_weight(
                    labels=labels,
                    loss_weights=loss_weights,
                    full_text=full_text,
                    images=images,
                    content_offset=start,
                    token_bounds=(start_tok, end_tok),
                    char_spans=self._goal_block_char_spans(content_text),
                    weight=self.goal_loss_weight,
                )
            if self.plan_loss_weight != 1.0:
                self._apply_span_loss_weight(
                    labels=labels,
                    loss_weights=loss_weights,
                    full_text=full_text,
                    images=images,
                    content_offset=start,
                    token_bounds=(start_tok, end_tok),
                    char_spans=self._plan_decision_char_spans(content_text, latest_goal_active_items),
                    weight=self.plan_loss_weight,
                )

        if any_span:
            return labels, loss_weights

        # Backward-compatible fallback for legacy single-assistant datasets.
        prompt_messages = messages[:2]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=images[:1] or None,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        prompt_len = min(int(prompt_inputs["input_ids"].shape[1]), len(input_ids))
        labels = ([-100] * prompt_len + input_ids[prompt_len:])[: len(input_ids)]
        loss_weights = [0.0 if label == -100 else 1.0 for label in labels]
        return labels, loss_weights

    def _apply_span_loss_weight(
        self,
        *,
        labels: list[int],
        loss_weights: list[float],
        full_text: str,
        images: list[Image.Image],
        content_offset: int,
        token_bounds: tuple[int, int],
        char_spans: list[tuple[int, int]],
        weight: float,
    ) -> None:
        start_tok, end_tok = token_bounds
        for char_start, char_end in char_spans:
            span_start_tok = self._token_count_for_prefix(
                full_text[: content_offset + char_start],
                images,
            )
            span_end_tok = self._token_count_for_prefix(
                full_text[: content_offset + char_end],
                images,
            )
            span_start_tok = min(max(span_start_tok, start_tok), len(labels))
            span_end_tok = min(max(span_end_tok, span_start_tok), len(labels))
            span_end_tok = min(span_end_tok, end_tok)
            for pos in range(span_start_tok, span_end_tok):
                if labels[pos] != -100:
                    loss_weights[pos] = weight

    def _token_count_for_prefix(self, text: str, images: list[Image.Image]) -> int:
        """Token length of a rendered-text prefix under multi-image expansion.

        The prefix must be paired with exactly the images whose ``<|image_pad|>``
        placeholders it contains (span boundaries never split a placeholder:
        they always fall inside assistant text, which carries no image tokens).
        """
        if not text:
            return 0
        n_images = text.count("<|image_pad|>")
        prefix_images = images[:n_images]
        tokenized = self.processor(
            text=[text],
            images=prefix_images or None,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return int(tokenized["input_ids"].shape[1])

    @staticmethod
    def _goal_block_char_spans(text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        pattern = re.compile(
            r"(?ms)^\[GOAL\]\s*$.*?(?=^\[(?:ASSIGN|PLAN_SKETCH|TOOL_CALL|PLAN)\]\s*$|\Z)"
        )
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end()))
        return spans

    @staticmethod
    def _plan_block_char_spans(text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        pattern = re.compile(r"(?ms)^\[PLAN\]\s*$.*?(?=^\[[A-Z_]+\]\s*$|\Z)")
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end()))
        return spans

    @classmethod
    def _latest_goal_active_items(cls, text: str) -> set[str] | None:
        active_items: set[str] | None = None
        for goal_start, goal_end in cls._goal_block_char_spans(text):
            body = cls._block_body_after_header(text[goal_start:goal_end], "[GOAL]")
            goal = cls._decode_json_prefix(body)
            if isinstance(goal, dict) and isinstance(goal.get("active"), list):
                active_items = {
                    normalized
                    for item in goal["active"]
                    if (normalized := cls._normalize_plan_item(item))
                }
        return active_items

    @classmethod
    def _plan_decision_char_spans(
        cls,
        text: str,
        active_goal_items: set[str] | None,
    ) -> list[tuple[int, int]]:
        if active_goal_items is None:
            return []

        spans: list[tuple[int, int]] = []
        for plan_start, plan_end in cls._plan_block_char_spans(text):
            block = text[plan_start:plan_end]
            body = cls._block_body_after_header(block, "[PLAN]")
            body_offset = plan_start + len(block) - len(body)
            plan = cls._decode_json_prefix(body)
            if not isinstance(plan, list):
                continue
            op_spans = cls._json_array_object_char_spans(body)
            for op, (op_start, op_end) in zip(plan, op_spans):
                if not isinstance(op, dict) or not cls._is_weighted_plan_op(op, active_goal_items):
                    continue
                op_text = body[op_start:op_end]
                for key_start, key_end in cls._plan_decision_key_value_spans(op_text):
                    spans.append((
                        body_offset + op_start + key_start,
                        body_offset + op_start + key_end,
                    ))
        return spans

    @staticmethod
    def _block_body_after_header(block: str, header: str) -> str:
        match = re.match(rf"(?ms)^{re.escape(header)}\s*$\n?", block)
        if not match:
            return ""
        return block[match.end():]

    @staticmethod
    def _decode_json_prefix(text: str) -> Any:
        stripped = text.lstrip()
        if not stripped:
            return None
        try:
            value, _ = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError:
            return None
        return value

    @classmethod
    def _is_weighted_plan_op(cls, op: dict[str, Any], active_goal_items: set[str]) -> bool:
        op_type = str(op.get("op_type") or "").strip().lower()
        if op_type not in {"acquire", "local_acquire", "load_carrier", "grasp", "grasp_item"}:
            return False
        entities = cls._plan_op_entities(op)
        return bool(entities) and entities.issubset(active_goal_items)

    @classmethod
    def _plan_op_entities(cls, op: dict[str, Any]) -> set[str]:
        raw_entity = op.get("entity", op.get("item"))
        if isinstance(raw_entity, list):
            return {
                normalized
                for item in raw_entity
                if (normalized := cls._normalize_plan_item(item))
            }
        normalized = cls._normalize_plan_item(raw_entity)
        return {normalized} if normalized else set()

    @staticmethod
    def _normalize_plan_item(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()

    @staticmethod
    def _plan_decision_key_value_spans(op_text: str) -> list[tuple[int, int]]:
        pattern = re.compile(r'"(?:op_type|entity|item)"\s*:\s*(?:"(?:\\.|[^"\\])*"|null)')
        return [(match.start(), match.end()) for match in pattern.finditer(op_text)]

    @staticmethod
    def _json_array_object_char_spans(text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        array_start = text.find("[")
        if array_start < 0:
            return spans

        square_depth = 0
        curly_depth = 0
        object_start: int | None = None
        in_string = False
        escape = False
        for idx in range(array_start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "[":
                square_depth += 1
                continue
            if char == "]":
                if square_depth == 1 and curly_depth == 0:
                    break
                square_depth = max(0, square_depth - 1)
                continue
            if char == "{":
                if square_depth == 1 and curly_depth == 0:
                    object_start = idx
                curly_depth += 1
                continue
            if char == "}":
                if curly_depth == 0:
                    continue
                curly_depth -= 1
                if square_depth == 1 and curly_depth == 0 and object_start is not None:
                    spans.append((object_start, idx + 1))
                    object_start = None
        return spans


class WeightedTokenTrainer(Trainer):
    """Trainer with optional per-token CE weights from the collator."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # noqa: ANN001
        loss_weights = inputs.pop("loss_weights", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        if loss_weights is None or labels is None:
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].to(shift_logits.device).contiguous()
        valid = (shift_labels != -100).to(shift_weights.dtype)
        token_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)
        weighted = token_loss * shift_weights * valid
        denom = (shift_weights * valid).sum().clamp_min(1.0)
        loss = weighted.sum() / denom
        return (loss, outputs) if return_outputs else loss


def _load_vl_model(model_path: str, cfg: dict[str, Any]):
    # Dispatch on the checkpoint's own model_type so P10.S2's generation x scale
    # grid trains through one code path; also verifies the FSDP wrap class in the
    # config names this generation's decoder layer, which otherwise only shows up
    # as a silently un-sharded (OOM) run hours in.
    from nsdl.vl_model_loader import (
        fsdp_layer_cls_for_model_type,
        resolve_vl_model_class,
        resolve_vl_model_type,
    )

    model_type = resolve_vl_model_type(model_path)
    model_cls = resolve_vl_model_class(model_path)
    wrap_cls = ((cfg.get("training") or {}).get("fsdp_config") or {}).get(
        "transformer_layer_cls_to_wrap"
    )
    if wrap_cls:
        expected = fsdp_layer_cls_for_model_type(model_type)
        if str(wrap_cls) != expected:
            raise ValueError(
                f"fsdp_config.transformer_layer_cls_to_wrap={wrap_cls!r} does not match "
                f"model_type={model_type!r} (expected {expected!r})"
            )

    attn_impl = "flash_attention_2" if cfg["model"].get("flash_attn", True) else "sdpa"
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    try:
        model = model_cls.from_pretrained(model_path, **model_kwargs)
    except Exception:
        print(f"[WARN] {attn_impl} failed, falling back to sdpa")
        model_kwargs["attn_implementation"] = "sdpa"
        model = model_cls.from_pretrained(model_path, **model_kwargs)
    model.config.use_cache = False
    _apply_freezing(model, cfg.get("model", {}))
    return model


_VISION_TOWER_ATTRS = ("visual", "vision_tower", "vision_model")


def vision_tower_prefix(model) -> str:
    """Dotted prefix under which the vision tower's parameters live.

    This is looked up on the live module tree rather than hard-coded because the
    layout is not stable across generations. Qwen2/2.5-VL put the tower at the
    top level (``visual.*`` / ``model.*``); Qwen3-VL nests everything one level
    deeper (``model.visual.*`` / ``model.language_model.*``). Against Qwen3-VL a
    hard-coded ``visual.`` prefix matches *no* parameter, so every freeze flag
    silently degrades to a no-op and the run becomes a full ViT fine-tune at the
    language-model LR -- a different recipe from the Qwen2 cells it is meant to
    be comparable with. Nothing raises and the loss curve looks fine; the run
    just is not the experiment. Hence: resolve it, and fail if it cannot be.
    """
    candidates = [
        name
        for name, _ in model.named_modules()
        if name and name.rsplit(".", 1)[-1] in _VISION_TOWER_ATTRS
    ]
    if not candidates:
        raise ValueError(
            f"{type(model).__name__} exposes no vision tower under any of "
            f"{_VISION_TOWER_ATTRS}; refusing to run, because the freeze flags "
            f"would silently apply to nothing"
        )
    # Shallowest wins: the tower itself, not some same-named submodule inside it.
    return min(candidates, key=lambda n: (n.count("."), n)) + "."


def _apply_freezing(model, model_cfg: dict[str, Any]) -> None:
    freeze_vision = bool(model_cfg.get("freeze_vision_tower", True))
    freeze_projector = bool(model_cfg.get("freeze_multi_modal_projector", True))
    freeze_language = bool(model_cfg.get("freeze_language_model", False))

    vision_prefix = vision_tower_prefix(model)
    # `deepstack_merger_list` is Qwen3-VL-only: the same patch-merger class as
    # `merger`, applied to intermediate ViT levels. Counted as projector for the
    # same reason `merger` is -- it projects into the LM embedding space. Inert
    # under every current config (both flags are true), and only observable if
    # someone unfreezes exactly one of the two.
    projector_prefixes = (vision_prefix + "merger.", vision_prefix + "deepstack_merger_list.")

    counts = {"vision": 0, "projector": 0, "language": 0}
    for name, param in model.named_parameters():
        is_projector = (
            "multi_modal_projector" in name
            or "mm_projector" in name
            or name.startswith(projector_prefixes)
        )
        is_vision = name.startswith(vision_prefix) and not is_projector
        is_language = not is_vision and not is_projector
        if freeze_vision and is_vision:
            param.requires_grad = False
        if freeze_projector and is_projector:
            param.requires_grad = False
        if freeze_language and is_language:
            param.requires_grad = False
        group = "projector" if is_projector else ("vision" if is_vision else "language")
        counts[group] += param.numel()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total:.2%})")
    # Printed so a mis-resolved prefix is visible in the log's first seconds
    # rather than inferred from a memory footprint hours later.
    print(
        f"Param split under vision prefix {vision_prefix!r}: "
        + ", ".join(f"{k}={v:,}" for k, v in counts.items())
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dry_run", action="store_true", help="Run 2 steps only")
    parser.add_argument("--data-dir", default=None, help="Override data.dir")
    parser.add_argument("--model-path", default=None, help="Override model.path")
    parser.add_argument("--output-dir", default=None, help="Override training.output_dir")
    parser.add_argument("--goal-loss-weight", type=float, default=None, help="Override training.goal_loss_weight")
    parser.add_argument("--plan-loss-weight", type=float, default=None, help="Override training.plan_loss_weight")
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "CUDA device visibility for this process. Use 'cuda:0' or 'cuda:1' "
            "to run independent single-GPU trainings concurrently, or '0,1' "
            "to expose both GPUs for FSDP. Default: auto."
        ),
    )
    parser.add_argument(
        "--auto-torchrun",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Automatically relaunch FSDP configs with torchrun when this script "
            "is started as a single Python process. Default: enabled."
        ),
    )
    parser.add_argument(
        "--torchrun-nproc-per-node",
        type=int,
        default=None,
        help="Override torchrun --nproc_per_node when --auto-torchrun is enabled.",
    )
    args = parser.parse_args()
    _configure_visible_devices(args.device)

    cfg = load_config(args.config)
    if args.data_dir:
        cfg.setdefault("data", {})["dir"] = args.data_dir
    if args.model_path:
        cfg.setdefault("model", {})["path"] = args.model_path
    if args.output_dir:
        cfg.setdefault("training", {})["output_dir"] = args.output_dir
    if args.goal_loss_weight is not None:
        cfg.setdefault("training", {})["goal_loss_weight"] = float(args.goal_loss_weight)
    if args.plan_loss_weight is not None:
        cfg.setdefault("training", {})["plan_loss_weight"] = float(args.plan_loss_weight)

    _maybe_reexec_with_torchrun(args, cfg)

    model_path = cfg["model"]["path"]
    data_dir = Path(cfg["data"]["dir"])
    output_dir = cfg["training"]["output_dir"]
    max_seq_len = int(cfg["data"].get("max_seq_len", 4096))

    print(f"Model: {model_path}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # Input-resolution cap (A0b, 2026-08-27). Must be applied BEFORE the dataset is
    # tokenized: the collator materializes pixel_values from this same processor, so
    # a cap set later would leave the cached sequence lengths at full resolution.
    # Unset == full resolution == every SFT run before this date.
    applied_pixel_cap = apply_image_pixel_cap(processor, tag="sft")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    train_ds = load_from_disk(str(data_dir / "train"))
    expected_prompt_variant = cfg.get("data", {}).get("prompt_variant")
    if expected_prompt_variant:
        contract = validate_vl_sft_dataset_contract(
            train_ds,
            data_dir=data_dir / "train",
            expected_prompt_variant=expected_prompt_variant,
        )
        print(f"Data contract: {contract}")
    internal_val_ratio = float(cfg["data"].get("internal_val_ratio", 0.0))
    train_cfg = cfg["training"]
    goal_loss_weight = float(train_cfg.get("goal_loss_weight", 3.0))
    plan_loss_weight = float(train_cfg.get("plan_loss_weight", 1.0))
    print(f"Goal loss weight: {goal_loss_weight}")
    print(f"Plan loss weight: {plan_loss_weight}")
    eval_strategy = train_cfg.get("eval_strategy", "no")
    do_eval = eval_strategy != "no" and internal_val_ratio > 0.0
    val_ds = None
    if do_eval:
        seed = int(cfg["data"].get("internal_val_seed", 4242))
        splits = _train_test_split_with_optional_strata(
            train_ds,
            ratio=internal_val_ratio,
            seed=seed,
            stratify_cols=list(cfg["data"].get("internal_val_stratify", [])),
        )
        train_ds = splits["train"]
        val_ds = splits["test"]
        print(f"Train (post-split): {len(train_ds)}, Internal val: {len(val_ds)}")
    else:
        print(f"Train: {len(train_ds)} (eval disabled)")

    model = _load_vl_model(model_path, cfg)

    report_to = os.environ.get("TRAIN_REPORT_TO", train_cfg.get("report_to", "none"))
    wandb_enabled = bool(report_to and "wandb" in str(report_to).lower())
    if wandb_enabled:
        os.environ.setdefault("WANDB_MODE", "online")
        os.environ.setdefault("WANDB_CONSOLE", "off")
        os.environ.setdefault("WANDB__SERVICE_WAIT", "300")
        os.environ.setdefault("WANDB_PROJECT", train_cfg.get("wandb_project", "EnactGraph-VL-SFT"))
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
        os.environ.setdefault("WANDB_WATCH", "false")

    # `max_steps` in the YAML is opt-in and only exists for bounded runs (env
    # acceptance short runs, VRAM probes). Absent -- every production config --
    # it stays -1 and epochs decide, exactly as before.
    max_steps = 2 if args.dry_run else int(train_cfg.get("max_steps", -1))
    num_epochs = 1 if args.dry_run else train_cfg.get("num_train_epochs", 2)
    gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", True))
    extra_training_args = {}
    for optional_key in (
        "optim",
        "fsdp",
        "fsdp_config",
        "ddp_find_unused_parameters",
        "torch_empty_cache_steps",
        "save_only_model",
    ):
        if optional_key in train_cfg:
            extra_training_args[optional_key] = train_cfg[optional_key]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 2),
        learning_rate=float(train_cfg.get("learning_rate", 1e-5)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        label_smoothing_factor=0.0,
        bf16=True,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 500),
        eval_strategy=eval_strategy if do_eval else "no",
        eval_steps=train_cfg.get("eval_steps", 500) if do_eval else None,
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=do_eval and bool(train_cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss") if do_eval else None,
        greater_is_better=bool(train_cfg.get("greater_is_better", False)) if do_eval else None,
        report_to="none" if wandb_enabled else report_to,
        run_name=train_cfg.get("run_name", "slm-vl-sft"),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        remove_unused_columns=False,
        seed=int(train_cfg.get("seed", 42)),
        **extra_training_args,
    )

    callbacks = []
    if wandb_enabled:
        callbacks.append(SafeWandbCallback())
    if do_eval:
        patience = int(train_cfg.get("early_stopping_patience", 0))
        if patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = WeightedTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=VisionChatDataCollator(
            processor,
            max_seq_len=max_seq_len,
            goal_loss_weight=goal_loss_weight,
            plan_loss_weight=plan_loss_weight,
        ),
        tokenizer=processor.tokenizer,
        callbacks=callbacks,
    )

    print("Starting VL SFT training...")
    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    if applied_pixel_cap is not None:
        # The cap rides along in the saved preprocessor_config.json, so evaluating
        # this checkpoint is resolution-matched by construction even with the env
        # var unset. Verify rather than assume: if it did not persist, every later
        # eval of this ckpt would silently run at 2,691 tokens against weights
        # trained at 646, and nothing downstream would flag it.
        saved = json.loads((Path(final_dir) / "preprocessor_config.json").read_text())
        if int(saved.get("max_pixels", -1)) != applied_pixel_cap and \
           int((saved.get("size") or {}).get("longest_edge", -1)) != applied_pixel_cap:
            raise RuntimeError(
                f"pixel cap {applied_pixel_cap} did not persist into "
                f"{final_dir}/preprocessor_config.json (got max_pixels="
                f"{saved.get('max_pixels')!r}, size={saved.get('size')!r})"
            )
        print(f"[sft] pixel cap {applied_pixel_cap} persisted into {final_dir}")
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()
