from transformers import AutoModelForCausalLM, AutoProcessor
import transformers.image_utils as image_utils

# Monkey patch VideoInput for newer transformers versions
try:
    from transformers.video_utils import VideoInput
    image_utils.VideoInput = VideoInput
except Exception:
    # If anything goes wrong here, just skip; image-only inference will still work
    pass

import torch

model_path = "/root/model/Hulu-Med-7B"
image_path = "/root/dataset/skin/Derm1M/IIYI/7_3.png"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
)
tokenizer = processor.tokenizer


prompt = ('''
**Instruction (system)**
You are a dermatology vision–language assistant. Given one clinical or dermoscopic image and optional user text, infer the most likely disease name. If the image is not a lesion photo (e.g., poster, icon, cartoon) or is too poor-quality to assess, return “Not applicable”. Use only visual evidence and the user text; do not invent details.
**image or clinical description**
The user's chief complaint is related to a skin condition they are experiencing, specifically described as '皮肿淀粉样变' (Amyloidosis of the skin). They have uploaded a photo depicting their skin condition. Other forum users responded to the post discussing similar instances and treatments. One user noted they saw an identical case during their advanced studies. Another user inquired about effective medications for this skin condition.
**Task (user)**
Answer the question: “What is the name of the disease shown in the image?\n”
Return a single word or short phrase for the primary answer, and provide top-3 possible diseases with probabilities.Answer in English.
**Output rules**
1.Output strict JSON only, no extra text.
2.answer must be one word or a short phrase.
3.top3 has exactly 3 items, sorted by prob (0–1), and the three probs sum to 1.0 (±0.01).
4.If the image is not a real lesion or is unreadable, set answer to "Not applicable" and return an empty array for top3.
5.Keep reasoning concise and purely morphological (region, color/shape/border/texture, elevation, perilesional skin). No treatment advice.
**JSON schema**
{
  "answer": "<single word or short phrase>",
  "top3": [
    {"disease": "<name>", "prob": 0.00},
    {"disease": "<name>", "prob": 0.00},
    {"disease": "<name>", "prob": 0.00}
  ],
  "reason": {
    "region": "<if discernible>",
    "lesion_morphology": {
      "size_mm": "<if scale visible, else null>",
      "shape": "<round/oval/irregular>",
      "border": "<well-defined/ill-defined; smooth/notched>",
      "colour": "<uniform/variegated + hues>",
      "surface": "<smooth/scaly/crusted/ulcerated/verrucous>"
    },
    "elevation": "<flat/slightly raised/plaque/nodule/depressed/NA>",
    "perilesional_skin": "<erythema/induration/atrophy/scale/bleeding/NA>"
  },
  "quality_flags": {
    "non_lesion_image": false,
    "low_resolution_or_glare": false,
    "occlusion": false
  }
}
''')

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": {
                    "image_path": image_path,
                },
            },
            {
                "type": "text",
                "text": prompt,
            },
        ],
    }
]

inputs = processor(
    conversation=conversation,
    add_system_prompt=True,
    add_generation_prompt=True,
    return_tensors="pt",
)

inputs = {
    k: v.to(model.device) if isinstance(v, torch.Tensor) else v
    for k, v in inputs.items()
}

if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.6,
        pad_token_id=tokenizer.eos_token_id,
    )

outputs = processor.batch_decode(
    output_ids,
    skip_special_tokens=True,
    use_think=False,
)[0].strip()

print(outputs)
