import os
import time
from typing import Any, Dict, List

from PIL import Image
from vllm import LLM, SamplingParams


os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class InternVL3:
    def __init__(self, model_path, args):
        super().__init__()

        self.model_path = model_path
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens
        self.max_image_num = getattr(args, "max_image_num", 1)
        self.tensor_parallel_size = int(getattr(args, "tensor_parallel_size", 1))
        self.adapter_path = getattr(args, "adapter_path", None)

        gpu_memory_utilization = float(getattr(args, "gpu_memory_utilization", 0.85))

        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            trust_remote_code=True,
            tensor_parallel_size=self.tensor_parallel_size,
            enforce_eager=True,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": max(1, self.max_image_num)},
        )

        print(f"adapter_path:{self.adapter_path}")

    def _load_image(self, image: Any):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image

    def _build_prompt_and_mm_data(self, messages: Dict[str, Any]) -> Dict[str, Any]:
        system_message = messages.get("system", "")
        prompt_text = messages.get("prompt", "")

        images: List[Any] = []
        user_parts: List[str] = []

        if "image" in messages:
            images.append(self._load_image(messages["image"]))
            user_parts.append("<IMG_CONTEXT>")

        elif "images" in messages:
            for image in messages["images"]:
                images.append(self._load_image(image))
                user_parts.append("<IMG_CONTEXT>")

        user_parts.append(prompt_text)
        user_content = "\n".join([x for x in user_parts if x is not None and x != ""])

        parts: List[str] = []
        if system_message:
            parts.append(f"<|im_start|>system\n{system_message}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n{user_content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        prompt = "".join(parts)

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": {},
        }

        if images:
            llm_inputs["multi_modal_data"]["image"] = images

        return llm_inputs

    def _build_sampling_params(self) -> SamplingParams:
        if self.temperature == 0:
            return SamplingParams(
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=self.repetition_penalty,
                max_tokens=self.max_new_tokens,
            )

        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_new_tokens,
        )

    def generate_output(self, messages: Dict[str, Any]) -> str:
        llm_inputs = self._build_prompt_and_mm_data(messages)
        sampling_params = self._build_sampling_params()
        outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    def generate_outputs(self, messages_list: List[Dict[str, Any]]):
        llm_inputs_list = []
        for messages in messages_list:
            llm_inputs_list.append(self._build_prompt_and_mm_data(messages))

        sampling_params = self._build_sampling_params()

        start_time = time.perf_counter()
        outputs = self.llm.generate(llm_inputs_list, sampling_params=sampling_params)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time = total_time / max(1, len(outputs))

        res = []
        sub_total_time = []

        for idx, output in enumerate(outputs, start=1):
            text = output.outputs[0].text
            print(f"idx:{idx},result-------------:{text}")
            res.append(text)
            sub_total_time.append(avg_time)

        return res, sub_total_time