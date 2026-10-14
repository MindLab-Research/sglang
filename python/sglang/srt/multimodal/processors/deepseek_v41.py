"""DeepSeek-V4.1 image preprocessing, preserving raw token IDs for Engram."""

import asyncio
import hashlib
import logging
import os
import threading
from collections import OrderedDict
from functools import partial

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.multimodal.deepseek_v41_image_processing import (
    GPU_PLAN_KEY,
    image_token_types,
    patchify_image,
    patchify_image_rust,
    prepare_image_gpu,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
    _mm_timing_async,
)
from sglang.srt.runtime_context import get_mm
from sglang.srt.rust_extensions import load_rust_extension

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Content-addressed image preprocess cache (ported from GLM-5.3 commit
# f84fce9943 fix#2). Agent conversations re-send identical history images on
# every turn; without this the per-image resize/normalize/patchify re-runs for
# byte-identical inputs (documented at 9-20s CPU per 32-image request on 1101,
# measured with a 99.99% radix hit). Keys on the md5 content hash attached to
# each loaded image plus the preprocessing fingerprint; CPU tensors only, LRU
# by bytes under SGLANG_DSV41_PREPROCESS_CACHE_MB (default 4 GiB).
# ---------------------------------------------------------------------------
_DSV41_PP_CACHE = OrderedDict()
_DSV41_PP_LOCK = threading.Lock()
_DSV41_PP_BYTES = 0
_DSV41_PP_MAX_BYTES = (
    int(os.environ.get("SGLANG_DSV41_PREPROCESS_CACHE_MB", "4096")) * 1024 * 1024
)
_DSV41_MM_TIMING = os.environ.get("SGLANG_MM_TIMING", "0") == "1"


def _dsv41_tensor_bytes(patches) -> int:
    try:
        return int(patches.numel() * patches.element_size())
    except Exception:
        return 0


def _dsv41_pp_key(media_hash, backend, fp):
    if not media_hash:
        return None
    return hashlib.md5(f"{media_hash}|{backend}|{fp}".encode()).hexdigest()


def _dsv41_pp_get(key):
    if not key:
        return None
    with _DSV41_PP_LOCK:
        v = _DSV41_PP_CACHE.get(key)
        if v is not None:
            _DSV41_PP_CACHE.move_to_end(key)
        return v


def _dsv41_pp_put(key, value, nbytes):
    global _DSV41_PP_BYTES
    if not key or nbytes <= 0 or nbytes > _DSV41_PP_MAX_BYTES:
        return
    with _DSV41_PP_LOCK:
        old = _DSV41_PP_CACHE.pop(key, None)
        if old is not None:
            _DSV41_PP_BYTES -= old[-1]
        _DSV41_PP_CACHE[key] = (*value, nbytes)
        _DSV41_PP_BYTES += nbytes
        while _DSV41_PP_BYTES > _DSV41_PP_MAX_BYTES and len(_DSV41_PP_CACHE) > 1:
            _, ev = _DSV41_PP_CACHE.popitem(last=False)
            _DSV41_PP_BYTES -= ev[-1]


class DeepseekV41ImageProcessor(BaseMultimodalProcessor):
    models = [DeepseekV4ForCausalLM]
    preserve_processor_input_ids = True
    prefer_tokenized_input = True
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.preprocess_backend = "cpu"
        self.cpu_patchify = patchify_image
        backend = get_mm().image_processor_backend
        if backend == "pil" or get_mm().disable_fast_image_processor:
            self.preprocess_backend = "cpu"
        elif envs.SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU.get():
            self.preprocess_backend = "gpu"
        elif backend == "auto":
            # Resolved before super().__init__(), which fingerprints the backend;
            # a load failure falls back to PIL, later image errors do not.
            try:
                extension = load_rust_extension(
                    "sglang.srt.rust_extensions._multimodal"
                )
                resize_patchify = extension.dsv41.resize_patchify
            except (ImportError, OSError, RuntimeError, AttributeError) as error:
                logger.warning(
                    "V4.1 Rust image processor unavailable; using PIL: %s", error
                )
            else:
                self.preprocess_backend = "rust"
                self.cpu_patchify = partial(
                    patchify_image_rust, resize_patchify=resize_patchify
                )
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.image_token_id = hf_config.image_token_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=_processor.convert_ids_to_tokens(self.image_token_id),
            image_token_id=self.image_token_id,
        ).build(_processor)

    def preprocess_fingerprint_payload(self):
        payload = super().preprocess_fingerprint_payload()
        payload["dsv41_preprocess_backend"] = self.preprocess_backend
        return payload

    @property
    def _dsv41_pp_fp(self):
        """Stable short fingerprint of everything that changes patchify result."""
        fp = getattr(self, "_dsv41_pp_fp_cached", None)
        if fp is not None:
            return fp
        try:
            payload = self.preprocess_fingerprint_payload()
            fp = hashlib.md5(repr(sorted(payload.items())).encode()).hexdigest()[:16]
        except Exception:
            fp = "nofp"
        self._dsv41_pp_fp_cached = fp
        return fp

    @_mm_timing_async("dsv41_total")
    async def process_mm_data_async(
        self, image_data, input_text, request_obj, *args, **kwargs
    ):
        base = await self.load_mm_data(
            input_text, image_data=image_data, multimodal_tokens=self.mm_tokens
        )
        ids = (
            input_text
            if isinstance(input_text, list)
            else self._processor.encode(input_text)
        )
        # perf(dsv41-mm) [ported from f84fce9943 fix#3]: never scan input_ids
        # token-by-token in Python. For a 500K-token prompt `ids.count(...)` +
        # `for token in ids` costs seconds per request, on every request.
        # Fast path: no images -> the output ids are exactly the input ids.
        if not base.images:
            if self.image_token_id in ids:
                raise ValueError("Image placeholders and images must match")
            return MultimodalProcessorOutput(
                input_ids=list(ids),
                mm_items=self._prepare_mm_items_for_transport([]),
                im_token_id=self.image_token_id,
            )
        import numpy as _np

        _ids_arr = _np.asarray(ids, dtype=_np.int64)
        _img_pos = _np.nonzero(_ids_arr == self.image_token_id)[0]
        if len(_img_pos) != len(base.images):
            raise ValueError("Image placeholders and images must match")
        images = iter(base.images)
        tokens, items = [], []
        _prev = 0
        for _pos in _img_pos:
            _pos = int(_pos)
            if _pos > _prev:
                tokens.extend(_ids_arr[_prev:_pos].tolist())
            _prev = _pos + 1
            image = next(images)
            plan = None
            _ck = _dsv41_pp_key(
                getattr(image, "_sglang_media_hash", None),
                self.preprocess_backend,
                self._dsv41_pp_fp,
            )
            _hit = _dsv41_pp_get(_ck)
            if _hit is not None:
                patches, h, w, lh, lw, plan = _hit[0], _hit[1], _hit[2], _hit[3], _hit[4], _hit[5]
                if _DSV41_MM_TIMING:
                    print(f"[MM-T] dsv41_imgcache=HIT key={(_ck or '')[:10]}", flush=True)
            elif self.preprocess_backend == "gpu":
                patches, plan, lh, lw = await asyncio.to_thread(
                    prepare_image_gpu, image, self.hf_config
                )
                h = plan["height"] // plan["patch_size"]
                w = plan["width"] // plan["patch_size"]
                _dsv41_pp_put(_ck, (patches, h, w, lh, lw, plan), _dsv41_tensor_bytes(patches))
                if _DSV41_MM_TIMING:
                    print(f"[MM-T] dsv41_imgcache=MISS key={(_ck or 'nokey')[:10]}", flush=True)
            else:
                patches, h, w, lh, lw = await asyncio.to_thread(
                    self.cpu_patchify, image, self.hf_config
                )
                _dsv41_pp_put(_ck, (patches, h, w, lh, lw, None), _dsv41_tensor_bytes(patches))
                if _DSV41_MM_TIMING:
                    print(f"[MM-T] dsv41_imgcache=MISS key={(_ck or 'nokey')[:10]}", flush=True)
            if self.keep_mm_features_on_device:
                patches = patches.to(torch.device("cuda", self.server_args.base_gpu_id))
            metadata = {"n_vit_h": h, "n_vit_w": w}
            if plan is not None:
                metadata[GPU_PLAN_KEY] = plan
            count = len(image_token_types(lh, lw))
            start = len(tokens)
            tokens.extend([self.image_token_id] * count)
            items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=patches,
                    offsets=[(start, start + count - 1)],
                    model_specific_data=metadata,
                )
            )
        if "_prev" in dir() and _prev < len(_ids_arr):
            tokens.extend(_ids_arr[_prev:].tolist())
        return MultimodalProcessorOutput(
            input_ids=tokens,
            mm_items=self._prepare_mm_items_for_transport(items),
            im_token_id=self.image_token_id,
        )
