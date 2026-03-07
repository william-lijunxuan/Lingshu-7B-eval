# Lingshu-7B-eval
Evaluate the model

# download code
```bash
cd /root/model
git clone https://github.com/william-lijunxuan/Lingshu-7B-eval.git
git checkout server
git pull  
cd Lingshu-7B-eval
```
# run 
## eval finetuning

```bash
conda activate lingshu
cd qwenvl/eval
bash shell/Lingshu/lingshu_7b.sh
```


## eval baseline
Change the ADAPTER_PATH in shell/Lingshu/lingshu_7b.sh to None
```bash
conda activate lingshu
cd qwenvl/eval
bash shell/Lingshu/lingshu_7b.sh
```
# Run lingshu-7b locally using gradio
```bash
conda activate lingshu
cd qwenvl/eval/gradio/lingshu-7b
python multi_image_conversation.py
```

# Run qwen3.5
```bash
cd model
git clone https://huggingface.co/Qwen/Qwen3.5-4B
# open new terminal
conda activate lingshu
pip install -U transformers
cd model/Lingshu-7B-eval/qwenvl/eval
git stash save"RL_eval"
git pull
bash shell/qwen3_5_4b/qwen3_5_4b.sh
```

# Run Hulu_Med_30A3
```bash
cd /Lingshu-7B-eval
git stash save"RL_eval"
git pull
cd Lingshu-7B-eval/qwenvl/eval
bash shell/Hulu_Med/Hulu_Med_30A3.sh
```

# Run MediX_R1
```bash
cd /Lingshu-7B-eval
git stash save"RL_eval"
git pull
cd Lingshu-7B-eval/qwenvl/eval
bash shell/MediX_R1/MediX_R1_8b.sh
```

# Run lingshu-I-8B(InternVL3)
```bash
cd /Lingshu-7B-eval
git stash save"RL_eval"
git pull
cd Lingshu-7B-eval/qwenvl/eval
bash shell/Lingshu/lingshu-I-8B.sh
```