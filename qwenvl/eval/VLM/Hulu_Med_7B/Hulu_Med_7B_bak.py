from transformers import AutoModelForCausalLM, AutoProcessor
import transformers.image_utils as image_utils
import torch, gc, time


class Hulu_Med_7B:
    def __init__(self, model_path, args):
        try:
            from transformers.video_utils import VideoInput
            image_utils.VideoInput = VideoInput
        except Exception:
            pass

        super().__init__()

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens
        self.adapter_path = getattr(args, "adapter_path", None)

        print(f"adapter_path:{self.adapter_path}")
        if self.adapter_path is not None and not (
            isinstance(self.adapter_path, str)
            and self.adapter_path.strip().lower() in {"none", "null", ""}
        ):
            from peft import PeftModel

            self.llm = PeftModel.from_pretrained(self.llm, self.adapter_path)
            print("----------------------Use fine-tuned weights---------------------------")
            self.llm.eval()

    def _build_conversation(self, messages):
        conversation = []
        # User message
        content = []

        # Single image
        if "image" in messages:
            img = messages["image"]
            if isinstance(img, str):
                image_obj = {"image_path": img}
            elif isinstance(img, dict) and "image_path" in img:
                image_obj = img
            else:
                # Fallback: pass raw object
                image_obj = img

            content.append(
                {
                    "type": "image",
                    "image": image_obj,
                }
            )

        # Multiple images
        elif "images" in messages:
            for img in messages["images"]:
                if isinstance(img, str):
                    image_obj = {"image_path": img}
                elif isinstance(img, dict) and "image_path" in img:
                    image_obj = img
                else:
                    image_obj = img

                content.append(
                    {
                        "type": "image",
                        "image": image_obj,
                    }
                )

        # Text prompt
        prompt_text = messages.get("prompt", "")
        content.append(
            {
                "type": "text",
                "text": prompt_text,
            }
        )

        conversation.append(
            {
                "role": "user",
                "content": content,
            }
        )

        return conversation

    def process_messages(self, messages):
        conversation = self._build_conversation(messages)

        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Move tensors to model device
        inputs = {
            k: v.to(self.llm.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        return inputs

    # def generate_output(self, messages):
    #     inputs = self.process_messages(messages)
    #     torch.cuda.empty_cache()
    #     do_sample = False if self.temperature == 0 else True
        # output_ids = self.llm.generate(**inputs,temperature=self.temperature,top_p=self.top_p,repetition_penalty=self.repetition_penalty,max_new_tokens=self.max_new_tokens,do_sample = do_sample,use_cache=False)
        # input_ids = inputs["input_ids"]
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, output_ids)
        # ]
        # outputs = self.processor.batch_decode(
        #     generated_ids_trimmed,
        #     skip_special_tokens=True,
        #     use_think=False,
        # )[0].strip()
        #
        # del inputs, output_ids, generated_ids_trimmed
        # gc.collect()
        # torch.cuda.empty_cache()

        # output_ids = self.llm.generate(**inputs,temperature=self.temperature,top_p=self.top_p,repetition_penalty=self.repetition_penalty,max_new_tokens=self.max_new_tokens,do_sample = do_sample,use_cache=False)
        # outputs = self.processor.batch_decode(
        #     output_ids,
        #     skip_special_tokens=True,
        #     use_think=False
        # )[0].strip()
        # print(f"Hulu_med_7B_outputs:{outputs}")
        # del inputs, output_ids, outputs
        # gc.collect()
        # torch.cuda.empty_cache()
        # return outputs

    def generate_output(self, messages):
        inputs = self.process_messages(messages)
        torch.cuda.empty_cache()
        do_sample = False if self.temperature == 0 else True

        output_ids = self.llm.generate(
            **inputs,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            use_cache=False,
        )
        outputs = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            use_think=False,
        )[0].strip()
        print(f"Hulu_med_7B_outputs:{outputs}")

        # Free GPU-heavy tensors only
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs
    def generate_outputs(self, messages_list):
        res = []
        sub_total_time = []

        for idx, messages in enumerate(messages_list, start=1):
            start_times = time.perf_counter()
            result = self.generate_output(messages)
            end_times = time.perf_counter()
            delta = end_times - start_times
            print(f"idx:{idx}, result-------------:{result}, total_time:{delta}")
            res.append(result)
            sub_total_time.append(delta)

        return res, sub_total_time
