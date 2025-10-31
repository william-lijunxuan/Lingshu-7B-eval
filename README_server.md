# Lingshu-7B-eval
Evaluate the model

# download code
```bash
cd /mnt/d/skinalor/model
git clone https://github.com/william-lijunxuan/Lingshu-7B-eval.git
git checkout server
git pull  
```
# run 
## eval baseline
```bash
conda activate lingshu
cd qwenvl/eval
bash shell/Lingshu/lingshu_7b.sh
```
## eval finetuning

```bash
cd qwenvl/eval
bash shell/Lingshu/lingshu_7b.sh
```