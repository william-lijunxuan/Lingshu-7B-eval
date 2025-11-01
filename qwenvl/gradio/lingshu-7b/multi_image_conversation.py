# gradio_vlm_app_multi.py
# Local VLM demo (Qwen2.5-VL + optional PEFT) with multi-image input.
# Comments are in English.

import os
import sys
import traceback
import socket
from typing import List, Union

import torch
import gradio as gr
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# Silence TensorFlow logs if accidentally imported by other libs
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# -------------------------------
# Configuration
# -------------------------------
BASE_MODEL = "/home/william/model/Lingshu-7B"  # change to local path
ADAPTER_DIR = "/home/william/model/Lingshu-7B-Finetuning/qwenvl/scripts/output"  # change to local path
USE_ADAPTER = True
ATTN_IMPL_CANDIDATES = ["flash_attention_2", "sdpa"]

DEFAULT_MAX_NEW = 256
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9

# -------------------------------
# Utilities
# -------------------------------
def log_exception(prefix: str, e: BaseException):
    print(f"[{prefix}] {type(e).__name__}: {e}", file=sys.stderr)
    traceback.print_exc()

def humanize_error(e: BaseException) -> str:
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return "CUDA out of memory. Reduce max_new_tokens or the number/size of images."
    return f"{type(e).__name__}: {e}"

def find_free_port(start=7860, end=7900) -> int:
    import socket as _s
    for p in range(start, end + 1):
        with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as s:
            s.setsockopt(_s.SOL_SOCKET, _s.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return 0

def _pick_attn_impl() -> str:
    for impl in ATTN_IMPL_CANDIDATES:
        if impl == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
                return "flash_attention_2"
            except Exception:
                continue
        if impl == "sdpa":
            return "sdpa"
    return "sdpa"

def _ensure_pil(img: Union[str, Image.Image]) -> Image.Image:
    """Accept either a PIL.Image or a path and return PIL.Image."""
    if isinstance(img, Image.Image):
        return img
    return Image.open(img).convert("RGB")

# -------------------------------
# Model loading (singleton)
# -------------------------------
_model = None
_processor = None

def load_pipeline():
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor

    try:
        attn_impl = _pick_attn_impl()
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            dtype=dtype,                         # use dtype (torch_dtype is deprecated)
            attn_implementation=attn_impl,
            device_map=device_map,
        )
    except Exception as e:
        log_exception("load_base_model", e)
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                BASE_MODEL, dtype=torch.float32, attn_implementation="sdpa", device_map=None
            )
        except Exception as ee:
            log_exception("load_base_model_fallback", ee)
            raise RuntimeError("Failed to load base model.") from ee

    try:
        processor = AutoProcessor.from_pretrained(BASE_MODEL, use_fast=True)
    except Exception as e:
        log_exception("load_processor", e)
        raise RuntimeError("Failed to load processor.") from e

    if USE_ADAPTER and os.path.isdir(ADAPTER_DIR):
        try:
            peft = PeftModel.from_pretrained(model, ADAPTER_DIR)
            model = peft.merge_and_unload()
        except Exception as e:
            log_exception("merge_adapter", e)

    model.eval()
    _model, _processor = model, processor
    return _model, _processor

# -------------------------------
# Inference helper (multi-image)
# -------------------------------
def generate_once(
    images: List[Union[str, Image.Image]],
    text_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    """
    Multi-image + text -> model output string.
    """
    try:
        model, processor = load_pipeline()
    except Exception as e:
        return f"[Error] {humanize_error(e)}"

    # Normalize to PIL list
    pil_images = []
    for img in images or []:
        try:
            pil_images.append(_ensure_pil(img))
        except Exception as e:
            log_exception("load_image", e)

    if not pil_images:
        return "[Error] No valid image found."

    try:
        # Build messages: multiple images before the text prompt
        contents = [{"type": "image", "image": im} for im in pil_images]
        if text_prompt and text_prompt.strip():
            contents.append({"type": "text", "text": text_prompt.strip()})

        messages = [{"role": "user", "content": contents}]

        # Tokenization / vision packing
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "do_sample": True if float(temperature) > 0 else False,
        }

        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    out_ids = model.generate(**inputs, **gen_kwargs)
            else:
                out_ids = model.generate(**inputs, **gen_kwargs)

        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        output_text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text

    except Exception as e:
        log_exception("inference", e)
        return f"[Error] {humanize_error(e)}"

# -------------------------------
# Gradio UI (Blocks) - multi-image
# -------------------------------
def submit_fn(chat_history, files, user_text, max_new_tokens, temperature, top_p):
    """
    files: list of paths (gr.Files) or list of tempfile objects; use their .name or path.
    """
    try:
        file_paths = []
        if files:
            # gr.Files in v4 returns a list of tempfile.NamedTemporaryFile or dicts with 'name'
            for f in files:
                if isinstance(f, dict) and "name" in f:
                    file_paths.append(f["name"])
                elif hasattr(f, "name"):
                    file_paths.append(f.name)
                elif isinstance(f, str):
                    file_paths.append(f)

        if not file_paths and (not user_text or not user_text.strip()):
            return chat_history, gr.update(value="")

        chat_history = chat_history + [[user_text or "", None]]

        output = generate_once(
            images=file_paths,
            text_prompt=user_text or "",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        chat_history[-1][1] = output
        return chat_history, gr.update(value="")
    except Exception as e:
        log_exception("submit_fn", e)
        chat_history = chat_history + [[user_text or "", f"[Error] {humanize_error(e)}"]]
        return chat_history, gr.update(value="")

def clear_fn():
    return [], None, "", None

with gr.Blocks(title="Skinalor", css="footer {visibility: hidden}") as demo:
    gr.Markdown("### Skinalor AI Specialist — Image + Text — Multi-Image + Text")

    with gr.Row():
        with gr.Column(scale=1):
            # Accept multiple images
            files_in = gr.Files(label="Upload images", file_types=["image"], file_count="multiple")
            # Optional preview gallery
            gallery = gr.Gallery(label="Preview", columns=4, height=220)

            user_in = gr.Textbox(label="Prompt", placeholder="Describe these images.", lines=3)
            with gr.Row():
                max_new = gr.Slider(16, 1024, value=DEFAULT_MAX_NEW, step=8, label="max_new_tokens")
            with gr.Row():
                temperature = gr.Slider(0.0, 1.5, value=DEFAULT_TEMPERATURE, step=0.05, label="temperature")
                top_p = gr.Slider(0.1, 1.0, value=DEFAULT_TOP_P, step=0.05, label="top_p")

            with gr.Row():
                btn_run = gr.Button("Generate", variant="primary")
                btn_clear = gr.Button("Clear")

        with gr.Column(scale=1):
            chat = gr.Chatbot(label="Conversation", height=520, show_copy_button=True)
            note = gr.Markdown(
                "Tip: You can upload multiple images. If VRAM is limited, try fewer images or a lower max_new_tokens."
            )

    # Simple preview: map Files -> Gallery
    def _preview(files):
        preview_items = []
        if files:
            for f in files:
                if isinstance(f, dict) and "name" in f:
                    preview_items.append(f["name"])
                elif hasattr(f, "name"):
                    preview_items.append(f.name)
                elif isinstance(f, str):
                    preview_items.append(f)
        return preview_items

    files_in.change(_preview, inputs=files_in, outputs=gallery)

    state = gr.State([])

    btn_run.click(
        submit_fn,
        inputs=[state, files_in, user_in, max_new, temperature, top_p],
        outputs=[chat, user_in],
    ).then(lambda h: h, inputs=chat, outputs=state)

    btn_clear.click(clear_fn, outputs=[chat, files_in, user_in, gallery]).then(lambda: [], outputs=state)

demo.queue()

if __name__ == "__main__":
    port = find_free_port(7860, 7860) or find_free_port(7861, 7900)
    try:
        demo.launch(server_name="127.0.0.1", server_port=port, share=False)
    except OSError:
        demo.launch(server_name="127.0.0.1", server_port=0, share=False)
