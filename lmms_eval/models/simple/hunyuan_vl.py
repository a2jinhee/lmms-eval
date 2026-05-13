import os
import re
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# HunYuanVL-MoT custom classes live outside lmms-eval (in the project root's modeling/).
# Set HY_EMBODIED_ROOT or PYTHONPATH to point at the project root before launching.
_hy_root = os.environ.get("HY_EMBODIED_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
if _hy_root not in sys.path:
    sys.path.insert(0, _hy_root)

try:
    from modeling import (
        HunYuanVLMoTConfig,
        HunYuanVLMoTForConditionalGeneration,
        HunYuanVLMoTProcessor,
    )

    AutoConfig.register("hunyuan_vl_mot", HunYuanVLMoTConfig, exist_ok=True)
    AutoModelForImageTextToText.register(HunYuanVLMoTConfig, HunYuanVLMoTForConditionalGeneration, exist_ok=True)
    AutoProcessor.register(HunYuanVLMoTConfig, HunYuanVLMoTProcessor, exist_ok=True)
    _has_hunyuan = True
except ImportError as e:
    eval_logger.warning(
        f"Failed to import HunYuanVL-MoT modeling classes ({e}). "
        "Set HY_EMBODIED_ROOT to the project root or add it to PYTHONPATH."
    )
    _has_hunyuan = False

_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@register_model("hunyuan_vl")
class HunYuanVL(lmms):
    """
    HY-Embodied (HunYuanVL-MoT) model for lmms-eval.

    Example:
        accelerate launch --num_processes 4 -m lmms_eval \\
            --model hunyuan_vl \\
            --model_args pretrained=tencent/HY-Embodied-0.5,device_map=auto \\
            --tasks site_bench_video --batch_size 1 --output_path ./logs/
    """

    def __init__(
        self,
        pretrained: str = "tencent/HY-Embodied-0.5",
        device: str = "cuda",
        device_map: str = "auto",
        batch_size: int = 1,
        use_cache: bool = True,
        max_num_frames: int = 32,
        enable_thinking: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        if not _has_hunyuan:
            raise ImportError("HunYuanVL-MoT modeling classes are required. Set HY_EMBODIED_ROOT to the project root.")

        accelerator = Accelerator()
        self.accelerator = accelerator

        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        self._model = AutoModelForImageTextToText.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            attn_implementation="eager",
        ).eval()

        self._processor = HunYuanVLMoTProcessor.from_pretrained(pretrained)
        self._tokenizer = self._processor.tokenizer
        self._config = self._model.config

        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.max_num_frames = int(max_num_frames)
        self.enable_thinking = enable_thinking if isinstance(enable_thinking, bool) else enable_thinking == "True"
        self._max_length = 2048

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for HunYuanVL.")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _load_video_frames(self, video_path: str) -> np.ndarray:
        """Uniformly sample self.max_num_frames frames from a video file."""
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        indices = np.linspace(0, total - 1, min(self.max_num_frames, total), dtype=int)
        indices = np.unique(indices)
        return vr.get_batch(indices).asnumpy()  # (T, H, W, C)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tok_decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be Union[str, list], got {type(until)}")

            assert self.batch_size_per_gpu == 1, "HunYuanVL currently supports batch_size=1 only."
            context = contexts[0]
            visuals = self.flatten([doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id])

            # Build message content and separately collect video/image tensors
            content = []
            video_inputs = []
            image_inputs = []
            for visual in visuals:
                if isinstance(visual, str) and visual.endswith(_VIDEO_EXTENSIONS):
                    content.append({"type": "video", "video": visual})
                    try:
                        video_inputs.append(self._load_video_frames(visual))
                    except Exception as e:
                        eval_logger.warning(f"Failed to load video {visual}: {e}")
                elif isinstance(visual, PIL.Image.Image):
                    content.append({"type": "image", "image": visual})
                    image_inputs.append(visual)
            content.append({"type": "text", "text": context})
            messages = [{"role": "user", "content": content}]

            if self.accelerator.is_local_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Messages for doc ID {doc_id[0]}:\n{messages}")

            try:
                # Render text prompt (inserts video/image token placeholders)
                text = self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )

                # Tokenize and process visuals separately so we control frame count
                inputs = self._processor(
                    text=[text],
                    images=image_inputs if image_inputs else None,
                    videos=video_inputs if video_inputs else None,
                    return_tensors="pt",
                )

                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self._device)

                max_new_tokens = gen_kwargs.get("max_new_tokens", 1024)
                temperature = gen_kwargs.get("temperature", 0.0)
                do_sample = temperature > 0

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        use_cache=self.use_cache,
                        temperature=temperature if do_sample else None,
                        do_sample=do_sample,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                output_ids = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
                text_output = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]

                if self.enable_thinking:
                    m = re.search(r"<answer>(.*?)</answer>", text_output, re.DOTALL)
                    if m:
                        text_output = m.group(1).strip()

                for term in until:
                    if term:
                        text_output = text_output.split(term)[0]

            except Exception as e:
                eval_logger.error(f"Error during generation for doc ID {doc_id[0]}: {e}", exc_info=True)
                text_output = ""

            res.append(text_output)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_output)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for HunYuanVL.")
