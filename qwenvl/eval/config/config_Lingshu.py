import os
from  datetime import datetime

today_str  = datetime.now().strftime("%Y%m%d_%H%M%S")

# RAG config --MMSkinQA
# RAG_FLAG="True"
# DATASET_NAME ="MMSkinQA"
# PROJECT_ROOT = "/mnt/d/skinalor/model/Skinalor/RAG/RAGDataSet"
# DB_DIR = os.path.join(PROJECT_ROOT, DATASET_NAME)
# CHROMA_PERSIST_PATH = os.path.join(DB_DIR, "chroma_db_skin")
# CHROMA_COLLECTION_NAME = r"skin_cases_multivector_"+DATASET_NAME

# RAG config --SkinCAP
# RAG_FLAG="True"
RAG_FLAG="False"
DATASET_NAME ="Derm1m"
PROJECT_ROOT = "/mnt/d/skinalor/model/Skinalor/RAG/RAGDataSet"
DB_DIR = os.path.join(PROJECT_ROOT, DATASET_NAME)
CHROMA_PERSIST_PATH = os.path.join(DB_DIR, "chroma_db_skin")
CHROMA_COLLECTION_NAME = "skin_cases_multivector"+DATASET_NAME

# embedding model config
EMBEDDING_MODEL_NAME = 'openai/clip-vit-base-patch32'

DATASETS_PATH = "redlessone"
# EVAL_DATASETS = "SkinCAP,SkinCAP,SKINgpt,MMSkinQA"
# EVAL_DATASETS = "SKINgpt,MMSkinQA"
EVAL_DATASETS = "Derm1m"


EVAL_LOCAL_DATASETS_FLAG ="True"
# EVAL_LOCAL_DATASETS_FILE ="/mnt/d/skinalor/dataset/skin/SkinCAP/SkinCAP_20250712_121252.json,/mnt/d/skinalor/dataset/skin/SkinCAP/SkinCAP_20250712_013256.json,/mnt/d/skinalor/dataset/skin/SKINgpt/20250711055029_SKINgpt_multiple_choice_QA.json,/mnt/d/skinalor/dataset/skin/MM-SkinQA/MM-SkinQA_20250711213519.json"
# EVAL_LOCAL_DATASETS_FILE ="/mnt/d/skinalor/dataset/skin/SKINgpt/20250711055029_SKINgpt_multiple_choice_QA.json,/mnt/d/skinalor/dataset/skin/MM-SkinQA/MM-SkinQA_20250711213519.json"
EVAL_LOCAL_DATASETS_FILE ="/mnt/d/skinalor/dataset/skin/Derm1M/Derm1M_train.jsonl"

EVAL_DATASET_PATH = "/mnt/d/skinalor/dataset/skin/Derm1M"
OUTPUT_PATH = f"eval_results/Lingshu-7B/{today_str}"
# VLM model path
MODEL_PATH = "/mnt/d/skinalor/model/Lingshu-7B"
MODEL_NAME="LingShu"
# ADAPTER_PATH = "/mnt/d/skinalor/model/Lingshu-7B-Finetuning/qwenvl/scripts/output"
ADAPTER_PATH = None

#vllm setting
CUDA_VISIBLE_DEVICES="0"
TENSOR_PARALLEL_SIZE="1"
USE_VLLM="True"

#Eval setting
SEED=42
REASONING="False"
TEST_TIMES=1


# Eval LLM setting
MAX_NEW_TOKENS=1024
MAX_IMAGE_NUM=6
TEMPERATURE=0.7
TOP_P=0.0001
REPETITION_PENALTY=1

# LLM judge setting
USE_LLM_JUDGE="False"
# gpt api model name
GPT_MODEL="gpt-4.1-2025-04-14"
OPENAI_API_KEY=""



















# PROJECT_ROOT = "/mnt/d/skinalor/model/Skinalor/RAG"
# DB_DIR = os.path.join(PROJECT_ROOT, "db")
# CHROMA_PERSIST_PATH = os.path.join(DB_DIR, "chroma_db_skin")
# CHROMA_COLLECTION_NAME = "skin_cases_multivector"

