import open_clip
from PIL import Image
import torch

# DEFAULT_IMAGE = "/home/william/dataset/skin/SkinCAP/skincap/55.png"
# DEFAULT_IMAGE = "/home/william/dataset/skin/Derm1M/IIYI/7_3.png" # amyloidosis
DEFAULT_IMAGE = "/home/william/dataset/skin/Derm1M/twitter/151_0.png" # herpes simplex virus
# Load model with huggingface checkpoint
model, _, preprocess = open_clip.create_model_and_transforms(
    'hf-hub:redlessone/DermLIP_ViT-B-16'
)
model.eval()

# Initialize tokenizer
tokenizer = open_clip.get_tokenizer('hf-hub:redlessone/DermLIP_ViT-B-16')

# Read example image
image = preprocess(Image.open(DEFAULT_IMAGE)).unsqueeze(0)

# Define disease labels (example: PAD dataset classes)
PAD_CLASSNAMES = [
    "eczema", "atopic dermatitis", "psoriasis", "seborrheic dermatitis",
    "tinea corporis", "tinea pedis", "tinea cruris", "tinea versicolor",
    "acne vulgaris", "rosacea", "urticaria", "impetigo", "cellulitis",
    "folliculitis", "herpes zoster", "herpes simplex", "vitiligo",
    "melanoma", "basal cell carcinoma", "squamous cell carcinoma",
    "nevus", "dermatofibroma", "keloid", "lipoma", "seborrheic keratosis",
    "contact dermatitis", "allergic contact dermatitis", "lichen planus",
    "pityriasis rosea", "intertrigo", "hidradenitis suppurativa",
    "pityrosporum folliculitis", "keratosis pilaris", "actinic keratosis",
]

# Build text prompts
template = lambda c: f'This is a skin image of {c}'
text = tokenizer([template(c) for c in PAD_CLASSNAMES])

# Inference
with torch.no_grad(), torch.autocast("cuda"):
    # Encode image and text
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# Get prediction
final_prediction = PAD_CLASSNAMES[torch.argmax(text_probs[0])]
print(f'This image is diagnosed as {final_prediction}.')
print("Label probabilities:", text_probs)
