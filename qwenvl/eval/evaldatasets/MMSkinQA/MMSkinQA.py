from ..utils.base_dataset import BaseDataset



import os
from datasets import Dataset
from datasets import load_dataset
from tqdm import tqdm
import json
from datasets import Dataset
from PIL import Image
from mathruler.grader import extract_boxed_content
from ..utils.utils import load_and_maybe_compress,save_json,extract,judger,get_compare_messages,judge_open_end_vqa,judge_judgement,judge_judgement_close_options,judge_close_end_vqa,judge_close_end_vqa_json
from distutils.util import strtobool
class MMSkinQA(BaseDataset):
    def __init__(self,idx,model,dataset_path,output_path):
        self.model = model
        self.idx = idx
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "HemanthKumarK/SKINgpt"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
        self.eval_local_datasets_flag = bool(strtobool(os.environ.get("eval_local_datasets_flag",True)))
        self.eval_local_datasets_file = str(os.environ.get("eval_local_datasets_file", "/mnt/d/skinalor/dataset/skin/MM-SkinQA/MM-SkinQA_20250711213519.json"))


    def load_data(self):
        # load local datasets
        if self.eval_local_datasets_flag:
            with open(self.eval_local_datasets_file.split(',')[self.idx]) as f:
                records = json.load(f)
            train_ds = Dataset.from_list(records)
            dataset = train_ds.select(range(300))
            # dataset = train_ds
            for idx,sample in tqdm(enumerate(dataset)):
                if idx % self.num_chunks == self.chunk_idx:
                    sample = self.construct_messages(sample)
                    self.samples.append(sample)
            return self.samples
        else:
            # load hugging face datasets
            dataset_path = self.dataset_path
            dataset = load_dataset(dataset_path, split="train")

            for idx, sample in tqdm(enumerate(dataset)):
                if idx % self.num_chunks == self.chunk_idx:
                    sample = self.construct_messages(sample)
                    self.samples.append(sample)
            return self.samples

    def construct_messages(self,sample):
        prompt_text = f"""
        \n{sample["question"]}\n
        """
        img_path = os.path.join("/mnt/d/skinalor/dataset/skin/MM-SkinQA", sample["image_name"])
        image_name = sample["image_name"]
        print(f"\nimage_name:{image_name}\n\n")
        print(f"\nprompt_text:{prompt_text}\n\n")
        image = Image.open(img_path).convert("RGB")

        # RAG
        # if bool(strtobool(os.environ.get("rag_flag", True))):
        #     chroma_persist_path = os.environ.get("chroma_persist_path")
        #     chroma_collection_name = os.environ.get("chroma_collection_name")
        #     embedding_model_name = os.environ.get("embedding_model_name")
        #     retriever = SkinCaseRetriever(
        #         chroma_persist_path,
        #         chroma_collection_name,
        #         embedding_model_name
        #     )
        #     retrieved_cases = retriever.retrieve_similar_cases(image,prompt_text)
        #
        #     ids = retrieved_cases.get('ids', [[]])[0]
        #     metadatas = retrieved_cases.get('metadatas', [[]])[0]
        #
        #     if not ids:
        #         print("No similar cases were found in the database.\n")
        #     else:
        #         prompt_text += "For your reference, here are some similar cases from a medical database:\n"
        #         for i, meta in enumerate(metadatas):
        #             prompt_text += f"\n--- Reference Case {i + 1} ---\n"
        #             disease = meta.get("disease")
        #             if disease:
        #                 prompt_text += f"Diagnosis: {disease}\n"
        #             prompt_text += f"Description: {meta.get('description', 'N/A')}\n"
        #
        #     print("--- Generated Prompt Text ---")
        #     print(prompt_text)

        messages = {"prompt": prompt_text, "image": image}
        sample["messages"] = messages
        del sample["image_name"]
        return sample


    def cal_metrics(self,out_samples):
        messages_list = []

        metrics = {
            "total metrics": {
                "total": 0,
                "right": 0
            },
            "open": {
                "total": 0,
                "right": 0,
                "bleu1": 0,
                "bleu2": 0,
                "bleu3": 0,
                "bleu4": 0,
                "rouge1": 0,
                "rouge2": 0,
                "rougel": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "em": 0,
            },
            "close": {
                "total": 0,
                "right": 0
            }
        }

        open_id = []
        for i, out_sample in tqdm(enumerate(out_samples)):
            response = out_sample["response"]
            print(f"response:{response}")
            if extract_boxed_content(response) != "None":
                response = extract_boxed_content(response)
            elif "<answer>" in response:
                response = extract(response, "answer")

            answer = out_sample["answer"]
            question = out_sample["question"]
            question_type = out_sample["question_type"]

            answer = answer.lower().strip()
            response = response.lower().strip()

            metrics["total metrics"]["total"] += 1
            if question_type =="close":
                metrics["close"]["total"] += 1
                correct = judge_judgement_close_options(answer, response)
                out_samples[i]["correct"] = correct
                if correct:
                    metrics["close"]["right"] += 1
                    metrics["total metrics"]["right"] += 1
            else:
                metrics["open"]["total"] += 1

                c_metrics = judge_open_end_vqa(answer, response)
                out_samples[i]["correct"] = c_metrics["em"]
                out_samples[i]["metrics"] = c_metrics
                if c_metrics["em"]:
                    metrics["total metrics"]["right"] += 1
                    metrics["open"]["right"] += 1
                for metric in c_metrics:
                    metrics["open"][metric] += c_metrics[metric]

                if os.environ.get("use_llm_judge", "False") == "True":
                    messages = get_compare_messages(question, response, answer)
                    messages_list.append(messages)
                    open_id.append(i)

        if os.environ.get("use_llm_judge", "False") == "True":
            metrics["total metrics"]["right"] = 0
            metrics["open"]["right"] = 0
            metrics["close"]["right"] = 0
            llm = judger
            results = llm.generate_outputs(messages_list)
            for i, result in zip(open_id, results):
                result = extract(result, "judge")
                result = True if result == "0" else False
                out_samples[i]["correct"] = result
                if result:
                    metrics["open"]["right"] += 1
                    metrics["total metrics"]["right"] += 1

        total_total = metrics["total metrics"]["total"]
        if total_total > 0:
           metrics["total metrics"]["acc"] = metrics["total metrics"]["right"] / total_total
        else:
            metrics["total metrics"]["acc"] = 0.0

        open_total = metrics["open"]["total"]
        if open_total > 0:
            metrics["open"]["acc"] = metrics["open"]["right"] / open_total
        else:
            metrics["open"]["acc"] = 0.0

        close_total = metrics["close"]["total"]
        if close_total > 0:
            metrics["close"]["acc"] = metrics["close"]["right"] / close_total
        else:
            metrics["close"]["acc"] = 0.0
        if open_total > 0:
            for metric in metrics["open"]:
                if metric not in ["right", "total", "acc"]:
                    metrics["open"][metric] = metrics["open"][metric] / open_total
        else:
            for metric in metrics["open"]:
                if metric not in ["right", "total", "acc"]:
                    metrics["open"][metric] = 0.0

        total_time = float(os.environ.get("total_time"))
        total_samples = metrics["total metrics"]["total"]
        avg_time = total_time / total_samples if total_samples > 0 else 0.0


        metrics["total_time_s"] = total_time
        metrics["avg_time_per_sample_s"] = avg_time
        return metrics, out_samples