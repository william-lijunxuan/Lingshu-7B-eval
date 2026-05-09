from transformers import AutoProcessor, AutoModelForCausalLM
import torch

MODEL_ID = "/root/model/gemma-4-E4B-it"

IMAGE = "/root/dataset/skin/Derm1M/IIYI/7_3.png"
prompt = (
    "You are a board‐certified dermatology AI specialist. A patient has just uploaded an image of a skin lesion. "
    "Carefully examine the lesion’s visual features—color, shape, borders, surface texture, and anatomic location—and "
    "then compose a single, fully descriptive diagnostic sentence in English. Mirror the expert style by:\n"
    "            1. Opening with a concise description of the key visual finding (e.g. “The red, smooth, exophytic nodule with a slightly narrowed base…”).\n"
    "            2. Stating the most likely diagnosis (e.g. “…may indicate squamous cell carcinoma.”).\n"
    "            3. Optionally noting any next steps for confirmation (e.g. “Further biopsy is recommended to confirm the diagnosis.”).\n\n"
    "            Example output (for a smooth red papule on the lip):\n"
    "            “The red, smooth, dome-shaped papule on the lip, with slight keratosis and prominent capillaries, is most consistent with basal cell carcinoma; a skin biopsy is advised for confirmation.“"
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
