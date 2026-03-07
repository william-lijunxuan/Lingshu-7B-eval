import time
import torch

from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .utils import load_image


class InternVL3:
    def __init__(self, model_path, args):
        super().__init__()

        self.llm = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )

        self.generation_config = {
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }

        self.adapter_path = getattr(args, "adapter_path", None)
        print(f"adapter_path:{self.adapter_path}")

        if self.adapter_path is not None and not (
            isinstance(self.adapter_path, str) and self.adapter_path.strip().lower() in {"none", "null", ""}
        ):
            from peft import PeftModel

            self.llm = PeftModel.from_pretrained(self.llm, self.adapter_path)
            print("----------------------Use GRPO weights---------------------------")
            self.llm = self.llm.merge_and_unload()
            self.llm.eval()

    def _get_vision_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def process_messages(self, messages):
        prompt = ""
        pixel_values = None

        if "messages" in messages:
            for message in messages["messages"]:
                role = message["role"]
                content = message["content"]
                prompt += f"{role}: {content}\n"

            return {
                "prompt": prompt.strip(),
                "pixel_values": None,
            }

        if "system" in messages:
            prompt += messages["system"] + "\n"

        text = messages["prompt"]

        if "image" in messages:
            prompt += "<image>\n" + text
            image = messages["image"]
            pixel_values = load_image(image).to(torch.bfloat16).to(self._get_vision_device())

        elif "images" in messages:
            images = messages["images"]
            for i, _ in enumerate(images):
                prompt += f"<image_{i + 1}>: <image>\n"
            prompt += text

            image_tensors = []
            for image in images:
                image_tensors.append(
                    load_image(image).to(torch.bfloat16).to(self._get_vision_device())
                )
            pixel_values = torch.cat(image_tensors, dim=0)

        else:
            prompt += text
            pixel_values = None

        return {
            "prompt": prompt,
            "pixel_values": pixel_values,
        }

    def generate_output(self, messages):
        llm_inputs = self.process_messages(messages)
        question = llm_inputs["prompt"]
        pixel_values = llm_inputs["pixel_values"]

        do_sample = self.generation_config["temperature"] != 0
        gen_config = {
            "max_new_tokens": self.generation_config["max_new_tokens"],
            "repetition_penalty": self.generation_config["repetition_penalty"],
            "do_sample": do_sample,
        }

        if do_sample:
            gen_config["temperature"] = self.generation_config["temperature"]
            gen_config["top_p"] = self.generation_config["top_p"]

        response, history = self.llm.chat(
            self.tokenizer,
            pixel_values,
            question,
            gen_config,
            history=None,
            return_history=True,
        )
        return response

    def generate_outputs(self, messages_list):
        res = []
        sub_total_time = []

        for idx, messages in enumerate(messages_list, start=1):
            start_time = time.perf_counter()
            result = self.generate_output(messages)
            end_time = time.perf_counter()
            delta = end_time - start_time

            print(f"idx:{idx},result-------------:{result},total_time:{delta}")
            res.append(result)
            sub_total_time.append(delta)

        return res, sub_total_time