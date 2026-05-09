from ultralytics.models.sam import SAM3SemanticPredictor
from PIL import Image
import numpy as np

# Initialize predictor with configuration
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    # model="/root/model/Sam3/sam3.pt",
    model="/root/model/Sam/Sam3/sam3.1_multiplex.pt",
    half=True,  # Use FP16 for faster inference
    save=True,
)
# predictor = SAM3SemanticPredictor(overrides=overrides, bpe_path="/root/model/Sam3/bpe_simple_vocab_16e6.txt.gz")
predictor = SAM3SemanticPredictor(overrides=overrides)

img_path = "/root/model/X-AnyLabeling/map.png"
# img_path = "/root/model/X-AnyLabeling/DeweiMap4.tif"
# img = Image.open(img_path).convert("RGB")
# img = np.array(img)
# Set image once for multiple queries
predictor.set_image(img_path)

# Query with multiple text prompts
# results = predictor(text=["oil pipeline", "long narrow pipeline", "underground pipeline trace","white oil pipeline","brown oil pipeline","white or brown oil pipeline"])
# results = predictor(text=[
#     "thin long pipe",
#     "narrow pipeline on ground",
#     "long cylindrical object"
# ])
# Works with descriptive phrases
# results = predictor(text=["pipeline under trees", "person with blue cloth"])

# Query with a single concept
results = predictor(text=["pipeline"])