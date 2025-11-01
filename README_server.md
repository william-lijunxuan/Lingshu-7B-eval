# Lingshu-7B-eval
Evaluate the model

# download code
```bash
cd /mnt/d/skinalor/model
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