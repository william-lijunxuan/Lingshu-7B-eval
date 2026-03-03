# gradio_vlm_app.py
# Minimal local VLM demo for Qwen2.5-VL + PEFT adapter (merged) with robust try/except.
# Comments are in English. No cloud calls; everything runs locally.

import os
import sys
import traceback
import socket
import torch
import gradio as gr
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
USE_ADAPTER = True  # set False to run base model only
ATTN_IMPL_CANDIDATES = ["flash_attention_2", "sdpa"]

DEFAULT_MAX_NEW = 256
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9

# -------------------------------
# Utilities
# -------------------------------
def log_exception(prefix: str, e: BaseException):
    """Print a concise error to stderr and a full traceback for debugging."""
    print(f"[{prefix}] {type(e).__name__}: {e}", file=sys.stderr)
    traceback.print_exc()

def humanize_error(e: BaseException) -> str:
    """Return a short, user-facing error string."""
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return (
            "CUDA out of memory. Reduce max_new_tokens, use smaller images, "
            "or free VRAM before retrying."
        )
    return f"{type(e).__name__}: {e}"

def find_free_port(start=7860, end=7900) -> int:
    """Find a free port between [start, end]. Return 0 to let OS decide if none found."""
    for p in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return 0

def _pick_attn_impl() -> str:
    """Prefer FA2 if available; otherwise use SDPA."""
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

# -------------------------------
# Model loading (singleton)
# -------------------------------
_model = None
_processor = None

def load_pipeline():
    """Load model/processor once with defensive error handling."""
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor

    try:
        attn_impl = _pick_attn_impl()
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None

        # Use `dtype` (torch_dtype is deprecated warnings in some versions)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            dtype=dtype,
            attn_implementation=attn_impl,
            device_map=device_map,
        )
    except Exception as e:
        log_exception("load_base_model", e)
        # Last-resort fallback: try SDPA on CPU if GPU load failed
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                BASE_MODEL, dtype=torch.float32, attn_implementation="sdpa", device_map=None
            )
        except Exception as ee:
            log_exception("load_base_model_fallback", ee)
            raise RuntimeError("Failed to load base model. Check BASE_MODEL path and dependencies.") from ee

    try:
        processor = AutoProcessor.from_pretrained(BASE_MODEL, use_fast=True)
    except Exception as e:
        log_exception("load_processor", e)
        raise RuntimeError("Failed to load processor. Check BASE_MODEL path.") from e

    if USE_ADAPTER and os.path.isdir(ADAPTER_DIR):
        try:
            peft = PeftModel.from_pretrained(model, ADAPTER_DIR)
            model = peft.merge_and_unload()
        except Exception as e:
            log_exception("merge_adapter", e)
            # Continue with base model if adapter merge fails
    else:
        # Adapter dir missing: continue with base model
        pass

    model.eval()
    _model, _processor = model, processor
    return _model, _processor

# -------------------------------
# Inference helper
# -------------------------------
def generate_once(image, text_prompt, max_new_tokens, temperature, top_p):
    """
    Run a single round: image + text -> model output string.
    Wrapped with try/except to return readable errors to the UI.
    """
    try:
        model, processor = load_pipeline()
    except Exception as e:
        return f"[Error] {humanize_error(e)}"

    try:
        # Build chat messages in Qwen format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},  # PIL.Image or path is acceptable
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        # Prepare inputs
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
                # autocast can still raise OOM; the error will be caught below
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    out_ids = model.generate(**inputs, **gen_kwargs)
            else:
                out_ids = model.generate(**inputs, **gen_kwargs)

        # Trim prompt tokens for clean decoding
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        output_text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text

    except Exception as e:
        log_exception("inference", e)
        return f"[Error] {humanize_error(e)}"

# -------------------------------
# Gradio UI (Blocks)
# -------------------------------
def submit_fn(chat_history, image, user_text, max_new_tokens, temperature, top_p):
    """Event handler with try/except around all IO paths."""
    try:
        if image is None and (user_text is None or user_text.strip() == ""):
            return chat_history, gr.update(value="")

        # Append user turn (text only in Chatbot; image is shown in the Image component)
        chat_history = chat_history + [[user_text or "", None]]

        output = generate_once(
            image=image,
            text_prompt=user_text or "",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        chat_history[-1][1] = output
        return chat_history, gr.update(value="")
    except Exception as e:
        log_exception("submit_fn", e)
        # Keep UI responsive and show an error message in the conversation
        chat_history = chat_history + [[user_text or "", f"[Error] {humanize_error(e)}"]]
        return chat_history, gr.update(value="")

def clear_fn():
    return [], None, ""

with gr.Blocks(title="Skinalor", css="footer {visibility: hidden}") as demo:
    gr.Markdown("### Skinalor AI Specialist — Image + Text")
    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="Image", height=360)
            user_in = gr.Textbox(label="Prompt", placeholder="Describe this image.", lines=3)
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
                "Model: Qwen2.5-VL (merged adapter). "
                "If VRAM is low, reduce max_new_tokens. Queue is enabled."
            )

    state = gr.State([])  # chat history

    btn_run.click(
        submit_fn,
        inputs=[state, image_in, user_in, max_new, temperature, top_p],
        outputs=[chat, user_in],
    ).then(lambda h: h, inputs=chat, outputs=state)

    btn_clear.click(clear_fn, outputs=[chat, image_in, user_in]).then(lambda: [], outputs=state)

# Gradio v4: no concurrency_count/max_size args in queue()
demo.queue()

if __name__ == "__main__":
    # Try preferred port; fall back to a free one if occupied
    port = find_free_port(7860, 7860)  # try 7860 first
    if port == 0:
        port = find_free_port(7861, 7900)
    try:
        demo.launch(server_name="127.0.0.1", server_port=port, share=False)
    except OSError:
        # Last resort: let OS assign an ephemeral port
        demo.launch(server_name="127.0.0.1", server_port=0, share=False)
