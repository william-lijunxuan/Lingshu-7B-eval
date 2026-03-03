import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from PIL import Image
import requests
from io import BytesIO

# ============ 配置 ============
model_path = "/root/model/Qwen3.5-4B"  # 您的本地模型路径
device = "cuda" if torch.cuda.is_available() else "cpu"
image_url = "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"

# ============ 加载模型和处理器 ============
print(f"Loading model from {model_path}...")
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# ============ 准备输入 ============
# 下载并处理图片
response = requests.get(image_url)
image = Image.open(BytesIO(response.content)).convert("RGB")

# 构建消息（Qwen VL 格式）
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": "The centres of the four illustrated circles are in the corners of the square. The two big circles touch each other and also the two little circles. With which factor do you have to multiply the radii of the little circles to obtain the radius of the big circles?\nChoices:\n(A) $\\frac{2}{9}$\n(B) $\\sqrt{5}$\n(C) $0.8 \\cdot \\pi$\n(D) 2.5\n(E) $1+\\sqrt{2}$\n\nPlease answer with the correct choice letter."
            }
        ]
    }
]

# ============ 推理 ============
print("Generating response...")
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
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
    "pad_token_id": processor.tokenizer.pad_token_id,
}

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