from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from peft import PeftModel
model_path = "/root/model/Lingshu-7B"
ADAPTER = "/root/model/Lingshu-7B-Finetuning/qwenvl/scripts/outputqwen3vl"
IMAGE = "/root/dataset/skin/Derm1M/IIYI/7_3.png"
# prompt = (
#     "You are a board‐certified dermatology AI specialist. A patient has just uploaded an image of a skin lesion. "
#     "Carefully examine the lesion’s visual features—color, shape, borders, surface texture, and anatomic location—and "
#     "then compose a single, fully descriptive diagnostic sentence in English. Mirror the expert style by:\n"
#     "            1. Opening with a concise description of the key visual finding (e.g. “The red, smooth, exophytic nodule with a slightly narrowed base…”).\n"
#     "            2. Stating the most likely diagnosis (e.g. “…may indicate squamous cell carcinoma.”).\n"
#     "            3. Optionally noting any next steps for confirmation (e.g. “Further biopsy is recommended to confirm the diagnosis.”).\n\n"
#     "            Example output (for a smooth red papule on the lip):\n"
#     "            “The red, smooth, dome-shaped papule on the lip, with slight keratosis and prominent capillaries, is most consistent with basal cell carcinoma; a skin biopsy is advised for confirmation.“"
# )

# prompt = ("You are a board‐certified dermatology AI specialist. A patient has just uploaded an image of a skin lesion. "
#      "Carefully examine the lesion’s visual features—color, shape, borders, surface texture, and anatomic location—and "
#           "Answer the question using a single word or phrase and put the answer in one")
prompt = ('''
**Instruction (system)**
You are a dermatology vision–language assistant. Given one clinical or dermoscopic image and optional user text, infer the most likely disease name. If the image is not a lesion photo (e.g., poster, icon, cartoon) or is too poor-quality to assess, return “Not applicable”. Use only visual evidence and the user text; do not invent details.
**image or clinical description**
The user's chief complaint is related to a skin condition they are experiencing, specifically described as '皮肿淀粉样变' (Amyloidosis of the skin). They have uploaded a photo depicting their skin condition. Other forum users responded to the post discussing similar instances and treatments. One user noted they saw an identical case during their advanced studies. Another user inquired about effective medications for this skin condition.
**Task (user)**
Answer the question: “What is the name of the disease shown in the image?\n<image>”
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


# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_path,use_fast=True)
# model = PeftModel.from_pretrained(model, ADAPTER)
# model = model.merge_and_unload()
# model.eval()
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": IMAGE,
            },
            {"type": "text", "text": prompt},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)


#baseline So, let\'s see. The image shows a close-up of a person\'s lower legs. The legs are covered with what looks like a white, powdery or scaly rash. The person is wearing denim shorts, and you can see part of their shoes—black sneakers, maybe. The background is a tiled floor, light-colored tiles. There\'s a date stamp in the bottom right corner: 2013-10-09, and some Chinese text that says "爱发网" which is a website name. The skin on the legs has a lot of white spots or flakes, which might be a

# fine tune So, let\'s describe the image. First, the main subject is a person\'s lower legs, from the knees down to the feet. The person is wearing dark denim shorts, so we can see their thighs and legs. The legs have a lot of what looks like white or light gray flakes or patches, which might be a skin condition, maybe something like psoriasis or eczema. The skin texture appears rough with these patches. The background is a light-colored tiled floor, and the person is standing. There\'s a timestamp in the bottom right corner: "2013 10 09" and some Chinese

# So, let\'s describe the image. First, it\'s a close-up of a person\'s lower legs. The person is wearing denim shorts that are rolled up at the bottom, so you can see the thighs and lower legs. The skin on the legs has a lot of white, flaky patches, which might be a rash or a skin condition like eczema or psoriasis. The legs are standing on a light-colored tiled floor, probably in a room. There\'s a black shoe on the right foot. In the bottom right corner, there\'s a timestamp: "2013 10 09" and



# baseline Got it, let's analyze the image. The patient's legs have a diffuse, white, scaly eruption. The color is grayish-white, the surface is scaly and rough, borders are not sharply defined, location is on the lower legs. This looks like tinea corporis (ringworm), but wait, tinea usually has more ring-like patterns. Wait, no, maybe it's lichenified or maybe a fungal infection. Wait, the appearance is scaly, hyperkeratotic, with a white, fine scale. Alternatively, could be chronic eczema or psoriasis. But psoriasis often has well
# fine tune  Got it, let\'s analyze the image. The patient\'s legs have a skin lesion. Looking at the visual features: the lesions are on the lower legs, they appear as multiple small, white or pale, rough patches. Wait, the description says "sweat" but the image shows a rash-like appearance. Wait, the user provided an image link, but in the description, it\'s "leg skin with white speckles". Wait, the example was about a papule, but here it\'s bilateral lower legs with a white, scaly or rough texture. Wait, the key is to describe color, shape, borders