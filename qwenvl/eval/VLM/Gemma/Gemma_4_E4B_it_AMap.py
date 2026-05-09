from transformers import AutoProcessor, AutoModelForCausalLM
import torch

MODEL_ID = "/home/william/model/gemma-4-E4B-it"

# IMAGE = "/home/william/dataset/skin/Derm1M/IIYI/7_3.png"
IMAGE = "/home/william/model/X-AnyLabeling/DeweiMap4.tif"
prompt = (
    "请描述下图片的内容，使用中文回答，图中像管道一样的细管，是石油管道，统计图中有几处石油管道？有很多石油管道被植被给覆盖住了，注意有的是石油管道的影子，影子不能统计，这种的按照一条石油管道进行统计"
)

device = torch.device("cuda:0")
# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
).to(device)
# Prompt
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Write a short joke about saving RAM."},
# ]
messages = [
    {
        "role": "user", "content": [
            {"type": "image", "image": IMAGE},
            {"type": "text", "text": prompt}
        ]
    }
]

# Process input
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]


print("model.device =", model.device)
print("input_ids.device =", inputs["input_ids"].device)
print("model dtype =", model.dtype)
print("first parameter dtype =", next(model.parameters()).dtype)
print("input_ids dtype =", inputs["input_ids"].dtype)
print("before generate allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
# Generate output
outputs = model.generate(**inputs, max_new_tokens=1024)
print("after generate allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
print("after generate reserved:", torch.cuda.memory_reserved() / 1024**3, "GB")
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
print(response)
# Parse output
processor.parse_response(response)
