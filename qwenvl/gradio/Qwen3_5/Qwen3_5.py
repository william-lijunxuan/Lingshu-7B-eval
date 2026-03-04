import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from PIL import Image
import requests
from io import BytesIO

# ============ 配置 ============
model_path = "/mnt/d/skinalor/model/Qwen3.5-4B"  # 您的本地模型路径
device = "cuda" if torch.cuda.is_available() else "cpu"
image_path = "/mnt/d/skinalor/dataset/skin/Derm1M/clean/edu/978-3-642-97931-6_00100_01864.png"

# ============ 加载模型和处理器 ============
print(f"Loading model from {model_path}...")
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_path, use_fast=True,trust_remote_code=True)

# ============ 准备输入 ============

image = Image.open(image_path).convert("RGB")

# 构建消息（Qwen VL 格式）
messages = [
    {
        "role": "system",
        "content": "Do not output reasoning steps.\nDo not include analysis, thoughts, explanations, or markdown.\nOutput valid JSON only.\nNo code block fences."
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": "**Instruction (system)**\nYou are a dermatology vision–language assistant. Given one clinical or dermoscopic image and optional user text, infer the most likely disease name. If the image is not a lesion photo (e.g., poster, icon, cartoon) or is too poor-quality to assess, return “Not applicable”. Use only visual evidence and the user text; do not invent details.\n\n**image or clinical description**\nBehçet's syndrome is a rare condition mainly affecting young men. It should be considered in any patient with recurrent oral and genital ulceration. The ulcers tend to be larger, deeper, and last longer than aphthous ulcers. There may also be one or more of the following: iritis, arthritis, thrombophlebitis, sterile pustules on the skin, erythema nodosum, and meningoencephalitis. Patients with these symptoms should be referred to a hospital for treatment.\n\n**body location**\noral\n\n**Task (user)**\nAnswer the question: “What is the name of the disease shown in the image?”\nReturn a single word or short phrase for the primary answer, and provide top-3 possible diseases with probabilities. Answer in English.\n\n**Output rules**\n1. Output strict JSON only, no extra text.\n2. `answer` must be one word or a short phrase.\n3. `top3` has exactly 3 items, each item includes fields `disease`, `prob`, and `reason`; the list is sorted by `prob` (0–1) in descending order, and the three `prob` values sum to 1.0 (±0.01). The `reason` is a concise morphological justification (e.g., region, color/shape/border/texture, elevation, perilesional skin).\n4. Keep reasoning concise and purely morphological (region, color/shape/border/texture, elevation, perilesional skin). No treatment advice.\n\n**JSON schema**\n{\n  \"answer\": \"<single word or short phrase>\",\n  \"top3\": [\n    {\"disease\": \"<name>\", \"prob\": 0.00, \"reason\": \"<short morphological rationale>\"},\n    {\"disease\": \"<name>\", \"prob\": 0.00, \"reason\": \"<short morphological rationale>\"},\n    {\"disease\": \"<name>\", \"prob\": 0.00, \"reason\": \"<short morphological rationale>\"}\n  ],\n  \"reason\": {\n    \"region\": \"<if discernible>\",\n    \"lesion_morphology\": {\n      \"size_mm\": \"<if scale visible, else null>\",\n      \"shape\": \"<round/oval/irregular>\",\n      \"border\": \"<well-defined/ill-defined; smooth/notched>\",\n      \"colour\": \"<uniform/variegated + hues>\",\n      \"surface\": \"<smooth/scaly/crusted/ulcerated/verrucous>\"\n    },\n    \"elevation\": \"<flat/slightly raised/plaque/nodule/depressed/NA>\",\n    \"perilesional_skin\": \"<erythema/induration/atrophy/scale/bleeding/NA>\"\n  },\n  \"quality_flags\": {\n    \"non_lesion_image\": false,\n    \"low_resolution_or_glare\": false,\n    \"occlusion\": false\n  }\n}\n"
            }
        ]
    }
]
print("prompt",messages)
# ============ 推理 ============
print("Generating response...")
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,enable_thinking=False
)
inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt",
    padding=True
).to(device)

# 生成配置（可根据需要调整）
generation_config = {
    "max_new_tokens": 512,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "do_sample": True,
    "pad_token_id": processor.tokenizer.pad_token_id
}
print("generation_config:", generation_config)

with torch.no_grad():
    output_ids = model.generate(**inputs, **generation_config)

# ============ 解码输出 ============
# 提取新生成的 tokens（去掉输入部分）
input_len = inputs["input_ids"].shape[1]
generated_ids = output_ids[0][input_len:]
response = processor.decode(generated_ids, skip_special_tokens=True)

print("\n" + "="*50)
print("🤖 Model Response:")
print("="*50)
print(response)