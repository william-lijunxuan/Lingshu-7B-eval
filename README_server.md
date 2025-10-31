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
## eval finetuning

```bash
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
