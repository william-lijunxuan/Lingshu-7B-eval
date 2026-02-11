import json
import math
import sys
import copy
import re
import os
import json
import difflib
import asyncio
import random
from PIL import Image

from itertools import chain
from math import log
from multiprocessing import Pool
from functools import partial
from tqdm.asyncio import tqdm_asyncio
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from mathruler.grader import extract_boxed_content
import nltk
from tqdm.auto import tqdm
import torch
from packaging import version
from collections import defaultdict, Counter
from openai import AzureOpenAI, OpenAI,AsyncAzureOpenAI,AsyncOpenAI
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoModel, AutoTokenizer, BertConfig, GPT2Tokenizer, RobertaTokenizer,
                          RobertaConfig, XLMConfig, XLNetConfig)
from transformers import __version__ as trans_version
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)
import unicodedata
from nltk.translate.meteor_score import meteor_score
from bert_score.scorer import BERTScorer


_BERT_SCORER = None


def get_bert_scorer():
    global _BERT_SCORER
    if _BERT_SCORER is None:
        _BERT_SCORER = BERTScorer(
            lang="en",
            rescale_with_baseline=False,
            idf=False,
            batch_size=64,
            nthreads=4,
        )
    return _BERT_SCORER
def tokenize(text):
    text = text.lower().replace(".", " .").split(" ")
    return text

def bleu(pred,target,n):
    weights=[1/n for _ in range(n)]
    tokenized_target = tokenize(target)
    tokenized_pred = tokenize(pred)
    return sentence_bleu([tokenized_target], tokenized_pred, weights=weights)

def rouge(pred,target):
    rouge_scorer = Rouge()
    rouge_scores = rouge_scorer.get_scores(pred.lower(), target.lower())
    return rouge_scores

# METEOR
def calculate_meteor(predictions, ground_truths):
    score = meteor_score([predictions.split()], ground_truths.split())
    return score
    # meteor_scores = [nltk.translate.meteor_score.meteor_score([truth.split()], pred.split()) for pred, truth in
    #                  zip(predictions, ground_truths)]
    # return sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0


# # BERTScore
def calculate_bertscore(predictions, ground_truths):
    """
    Wrapper around BERTScorer.score that:
    - accepts either str or List[str]
    - reuses a global BERTScorer instance
    - returns the mean F1 score as a scalar float
    """
    # Normalize input types: allow str or List[str]
    if isinstance(predictions, str):
        predictions_list = [predictions]
    else:
        predictions_list = list(predictions)

    if isinstance(ground_truths, str):
        ground_truths_list = [ground_truths]
    else:
        ground_truths_list = list(ground_truths)

    if len(predictions_list) != len(ground_truths_list):
        raise ValueError(
            f"Length mismatch in calculate_bertscore: "
            f"{len(predictions_list)} preds vs {len(ground_truths_list)} refs"
        )


    predictions_list = [(p or "").strip() for p in predictions_list]
    ground_truths_list = [(g or "").strip() for g in ground_truths_list]


    if all(p == "" for p in predictions_list) or all(g == "" for g in ground_truths_list):
        print(f"predictions:{predictions_list}, ground_truths:{ground_truths_list}")
        print("P_valid:0.0, R_valid:0.0, F_valid:0.0 (empty texts)")
        return 0.0

    scorer = get_bert_scorer()
    P_valid, R_valid, F_valid = scorer.score(predictions_list, ground_truths_list)

    p_mean = P_valid.mean().item()
    r_mean = R_valid.mean().item()
    f_mean = F_valid.mean().item()

    print(f"predictions:{predictions_list}, ground_truths:{ground_truths_list}")
    print(f"P_valid:{p_mean}, R_valid:{r_mean}, F_valid:{f_mean}")

    return f_mean


def get_compare_messages(question,response,answer):
    prompt = f"""
Your task is to determine whether the user's answer is correct based on the provided questions and standard answers (for example, if the user expresses a similar meaning to the standard answer, or another interpretation of the standard answer, it is considered correct.)

The question is: {question}

The standard answer: {answer}

The user's answer: {response}

Please strictly follow the following format for output(0 represents correct, 1 represents incorrect):
<think>{{your concise think step}}</think>
<judge>{{0/1}}</judge>

for example:
<think>The standard answer is right, and the user's answer is right frontal lobe, they express the same meaning, so it is correct.</think>
<judge>0</judge>
    """
    messages = [{"role":"user","content":prompt}]
    return messages


def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()

def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = 0
    highest_similarity = 0
    
    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)
        
        # If the current string is more similar than the previous most similar string, update the variables
        if similarity > highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity
    
    # Return the index of the most similar string
    return most_similar_index

def judge_multi_choice(choices,answer,response,alphas = None):
    response = response.lower()
    if response.split("\n\n")[0] in [chr(ord('a') + i) for i in range(len(choices))]:
        response = response.split("\n\n")[0]
    elif response.split("\n\n")[-1].split(".")[0] in [chr(ord('a') + i) for i in range(len(choices))]:
        response = response.split("\n\n")[-1].split(".")[0]
    
    response = parse_response(response)
    alphas = [chr(ord('a') + i) for i in range(len(choices))]
    choices = [choice.lower() for choice in choices]
    flag = False
    response = response.strip().lower()
    response = response.replace("\n","")
    split_response = response.split(".")[0]
    split_response = split_response.split(":")[-1]
    answer = answer.strip().lower()
    
    if len(split_response) > 300:
        flag = False
    # letter,letter.  choice,choice
    if split_response == answer:
        flag = True
    
    # letter,choice
    elif split_response in alphas:
        if choices[ord(split_response)-ord("a")]== answer:
            flag = True
    
    # choice letter
    elif split_response in choices:
        if answer in alphas and split_response == choices[ord(answer)-ord("a")]:
            flag = True
    # unparsed
    else:
        index = find_most_similar_index(choices,response)
        if alphas[index] == answer or choices[index] == answer:
            flag = True
    return flag


def parse_response(response):
    response = response.lower()
    if "boxed" in response:
        response = extract_boxed_content(response)
    elif "<answer>" in response:
        response = extract(response,"answer")
    answer_patterns = [
        "**answer**:",
        "**answer**",
        "*answer*:",
        "**answer:**",
        "answer is",
        "answer:",
        "答案:",
        "final answer",
        "final answer is"
    ]
    for answer_pattern in answer_patterns:
        if answer_pattern in response:
            response = response.split(answer_pattern)[-1]
    
    return response


def judge_close_end_vqa(answer,response):
    answer = answer.lower().replace("\n", "").replace(".", "").replace("-", " ")
    response = parse_response(response)
    response = response.replace("\n","").replace(".","").replace("-"," ")
    if response == answer:
        return True
    else:
        return False

def judge_judgement(answer,response):
    answer = answer.lower().replace("\n","").replace(".","").replace("-"," ")
    response = parse_response(response)
    response = response.replace("\n","").replace(".","").replace("-"," ")
    if ('yes' in response) ^ ('no' in response):
        if answer in response:
            return True
    return False

def judge_judgement_close_options(answer,response):
    response = response.replace("\n", "").replace(".", "").replace("-", " ")
    answer = answer.replace("\n", "").replace(".", "").replace("-", " ")
    pred_text = response.strip()
    predicted_answer = pred_text
    if ":" in pred_text and len(pred_text.split(":", 1)[0].strip()) == 1:
        predicted_answer = pred_text.split(":", 1)[1].strip()

    predicted_answer = predicted_answer.rstrip(' .').lower()
    answer = answer.strip().lower()

    print(f'predicted_answer: {predicted_answer} | Ground Truth: {answer}')
    if predicted_answer == answer:
        print("Correct")
        return True
    else:
        print("Incorrect")
        return False



def judge_open_end_vqa(answer,response):
    answer = answer.lower()
    response = parse_response(response)
    bleu1 = bleu(response,answer,1)
    bleu2 = bleu(response,answer,2)
    bleu3 = bleu(response,answer,3)
    bleu4 = bleu(response,answer,4)

    # bert_score=calculate_bertscore(response,answer)
    meteor_score=calculate_meteor(response,answer)
    bert_score=calculate_bertscore(response,answer)
    # bert_precision = bert_score["bert_precision"]

    em = response == answer
    rouge_scores = rouge(response,answer)
    rouge_1 = rouge_scores[0]["rouge-1"]["f"]
    rouge_2 = rouge_scores[0]["rouge-2"]["f"]
    rouge_l = rouge_scores[0]["rouge-l"]["f"]

    precision,recall,f1 = calculate_f1(response,answer)


    return {
        "em" : em,
        "bleu1" : bleu1,
        "bleu2" : bleu2,
        "bleu3" : bleu3,
        "bleu4" : bleu4,
        "rouge1" : rouge_1,
        "rouge2" : rouge_2,
        "rougel" :  rouge_l,
        "Meteor" :meteor_score,
        "BertScore" :bert_score,
        "precision": precision,
        "recall": recall,
        "f1" :f1         
    }


def calculate_f1(prediction, ground_truth):
    prediction_tokens = set(prediction.lower().split())
    ground_truth_tokens = set(ground_truth.lower().split())
    
    common = prediction_tokens & ground_truth_tokens
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0
    
    precision = len(common) / len(prediction_tokens)
    recall = len(common) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        return 0,0,0
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1,precision,recall





def get_content_between_a_b(start_tag, end_tag, text):
    extracted_text = ""
    start_index = text.find(start_tag)
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text += text[start_index + len(start_tag) : end_index] + " "
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break

    return extracted_text.strip()


def extract(text, type,hard = True):
    if text:
        target_str = get_content_between_a_b(f"<{type}>", f"</{type}>", text)
        if target_str:
            return target_str
        elif hard:
            return text
        else:
            return ""
    else:
        return ""

# DATA SAVING
def save_json(filename, ds):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(ds, f, indent=4, ensure_ascii=False)

class fake_response:
    def __init__(self,usage):
        self.usage = usage

def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")

async def deal_tasks(tasks, max_concurrent_tasks=500):
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    results = []

    async def sem_task(task):
        async with semaphore:
            return await task  

    sem_tasks = [sem_task(task) for task in tasks]

    for coro in tqdm_asyncio.as_completed(sem_tasks, total=len(sem_tasks)):
        result = await coro
        results.append(result)

    return results

def load_and_maybe_compress(img_path, max_side=512):
    image = Image.open(img_path).convert("RGB")
    width, height = image.size

    if max(width, height) > max_side:
        resample = getattr(Image, "Resampling", Image).LANCZOS
        image.thumbnail((max_side, max_side), resample)
    return image

class openai_llm:
    def __init__(self,model = None):
        if model is None:
            model = os.environ.get("judge_gpt_model","gpt-4.1-2025-04-14")

        self.model = model

        # api_key = os.environ["openai_api_key"]
        api_key = "rsadfadsdd"

        self.client = OpenAI(
            api_key=api_key
            )
        self.async_client = AsyncOpenAI(
            api_key=api_key
            )
    
    @retry(wait=wait_fixed(10), stop=stop_after_attempt(1000), before=before_retry_fn)
    def response(self,messages,**kwargs):
        response = self.client.chat.completions.create(
            # gpt4o-0513  gpt4-turbo-2024-04-29 gpt-4o-2
            model=kwargs.get("model", self.model),
            messages=messages,
            n = kwargs.get("n", 1),
            temperature= kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", 4000),
            timeout=kwargs.get("timeout", 180)
        )
        return response.choices[0].message.content
    

    
    @retry(wait=wait_fixed(10), stop=stop_after_attempt(1000), before=before_retry_fn)
    async def response_async(self,messages,**kwargs):
        response = await self.async_client.chat.completions.create(
            # gpt4o-0513  gpt4-turbo-2024-04-29 gpt-4o-2
            model=kwargs.get("model", self.model),
            messages=messages,
            n = kwargs.get("n", 1),
            temperature= kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", 4096),
            timeout=kwargs.get("timeout", 180)
        )      
        return response.choices[0].message.content
    
    def generate_output(self,messages,**kwargs):
        try:
            response = self.response(messages,**kwargs)
        except Exception as e:
            response = None
            print(f"get {kwargs.get('model', self.model)} response failed: {e}")
        return response
    
    async def generate_output_async(self,idx, messages,**kwargs):
        try:
            response = await self.response_async(messages,**kwargs)
        except Exception as e:
            response = None
            print(f"get {kwargs.get('model', self.model)} response failed: {e}")
        return idx,response
    
    def generate_outputs(self,messages,**kwargs):
        tasks = [self.generate_output_async(i,messages[i],**kwargs) for i in range(len(messages))]
        results = asyncio.run(deal_tasks(tasks))
        results = sorted(results, key=lambda x: x[0])
        results = [x[1] for x in results]
        return results

judger = openai_llm()

def sent_encode(tokenizer, sent):
    "Encoding as sentence based on the tokenizer"
    sent = sent.strip()
    if sent == "":
        return tokenizer.build_inputs_with_special_tokens([])
    elif isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, RobertaTokenizer):
        # for RoBERTa and GPT-2
        if version.parse(trans_version) >= version.parse("4.0.0"):
            return tokenizer.encode(
                sent,
                add_special_tokens=True,
                add_prefix_space=True,
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
        elif version.parse(trans_version) >= version.parse("3.0.0"):
            return tokenizer.encode(
                sent,
                add_special_tokens=True,
                add_prefix_space=True,
                max_length=tokenizer.max_len,
                truncation=True,
            )
        elif version.parse(trans_version) >= version.parse("2.0.0"):
            return tokenizer.encode(
                sent,
                add_special_tokens=True,
                add_prefix_space=True,
                max_length=tokenizer.max_len,
            )
        else:
            raise NotImplementedError(
                f"transformers version {trans_version} is not supported"
            )
    else:
        if version.parse(trans_version) >= version.parse("4.0.0"):
            return tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
        elif version.parse(trans_version) >= version.parse("3.0.0"):
            return tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length=tokenizer.max_len,
                truncation=True,
            )
        elif version.parse(trans_version) >= version.parse("2.0.0"):
            return tokenizer.encode(
                sent, add_special_tokens=True, max_length=tokenizer.max_len
            )
        else:
            raise NotImplementedError(
                f"transformers version {trans_version} is not supported"
            )
def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, : lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, : lens[i]] = 1
    return padded, lens, mask

def collate_idf(arr, tokenizer, idf_dict, device="cuda:0"):
    """
    Helper function that pads a list of sentences to hvae the same length and
    loads idf score for words in the sentences.

    Args:
        - :param: `arr` (list of str): sentences to process.
        - :param: `tokenize` : a function that takes a string and return list
                  of tokens.
        - :param: `numericalize` : a function that takes a list of tokens and
                  return list of token indexes.
        - :param: `idf_dict` (dict): mapping a word piece index to its
                               inverse document frequency
        - :param: `pad` (str): the padding token.
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """
    arr = [sent_encode(tokenizer, a) for a in arr]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]

    pad_token = tokenizer.pad_token_id

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, 0, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask

def bert_encode(model, x, attention_mask, all_layers=False):
    model.eval()
    with torch.no_grad():
        out = model(x, attention_mask=attention_mask, output_hidden_states=all_layers)
    if all_layers:
        emb = torch.stack(out[-1], dim=2)
    else:
        emb = out[0]
    return emb

def get_bert_embedding(
    all_sens,
    model,
    tokenizer,
    idf_dict,
    batch_size=-1,
    device="cuda:0",
    all_layers=False,
):
    """
    Compute BERT embedding in batches.

    Args:
        - :param: `all_sens` (list of str) : sentences to encode.
        - :param: `model` : a BERT model from `pytorch_pretrained_bert`.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `idf_dict` (dict) : mapping a word piece index to its
                               inverse document frequency
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """

    padded_sens, padded_idf, lens, mask = collate_idf(
        all_sens, tokenizer, idf_dict, device=device
    )

    if batch_size == -1:
        batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(
                model,
                padded_sens[i : i + batch_size],
                attention_mask=mask[i : i + batch_size],
                all_layers=all_layers,
            )
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=0)

    return total_embedding, mask, padded_idf


def greedy_cos_idf(
    ref_embedding,
    ref_masks,
    ref_idf,
    hyp_embedding,
    hyp_masks,
    hyp_idf,
    all_layers=False,
):
    """
    Compute greedy matching based on cosine similarity.

    Args:
        - :param: `ref_embedding` (torch.Tensor):
                   embeddings of reference sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison
        - :param: `ref_lens` (list of int): list of reference sentence length.
        - :param: `ref_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   reference sentences.
        - :param: `ref_idf` (torch.Tensor): BxK, idf score of each word
                   piece in the reference setence
        - :param: `hyp_embedding` (torch.Tensor):
                   embeddings of candidate sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison
        - :param: `hyp_lens` (list of int): list of candidate sentence length.
        - :param: `hyp_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   candidate sentences.
        - :param: `hyp_idf` (torch.Tensor): BxK, idf score of each word
                   piece in the candidate setence
    """
    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    if all_layers:
        B, _, L, D = hyp_embedding.size()
        hyp_embedding = (
            hyp_embedding.transpose(1, 2)
            .transpose(0, 1)
            .contiguous()
            .view(L * B, hyp_embedding.size(1), D)
        )
        ref_embedding = (
            ref_embedding.transpose(1, 2)
            .transpose(0, 1)
            .contiguous()
            .view(L * B, ref_embedding.size(1), D)
        )
    batch_size = ref_embedding.size(0)
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    if all_layers:
        masks = masks.unsqueeze(0).expand(L, -1, -1, -1).contiguous().view_as(sim)
    else:
        masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)

    masks = masks.float().to(sim.device)
    sim = sim * masks

    word_precision = sim.max(dim=2)[0]
    word_recall = sim.max(dim=1)[0]

    hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
    ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))
    precision_scale = hyp_idf.to(word_precision.device)
    recall_scale = ref_idf.to(word_recall.device)
    if all_layers:
        precision_scale = (
            precision_scale.unsqueeze(0)
            .expand(L, B, -1)
            .contiguous()
            .view_as(word_precision)
        )
        recall_scale = (
            recall_scale.unsqueeze(0).expand(L, B, -1).contiguous().view_as(word_recall)
        )
    P = (word_precision * precision_scale).sum(dim=1)
    R = (word_recall * recall_scale).sum(dim=1)
    F = 2 * P * R / (P + R)

    hyp_zero_mask = hyp_masks.sum(dim=1).eq(2)
    ref_zero_mask = ref_masks.sum(dim=1).eq(2)

    if all_layers:
        P = P.view(L, B)
        R = R.view(L, B)
        F = F.view(L, B)

    if torch.any(hyp_zero_mask):
        print(
            "Warning: Empty candidate sentence detected; setting raw BERTscores to 0.",
            file=sys.stderr,
        )
        P = P.masked_fill(hyp_zero_mask, 0.0)
        R = R.masked_fill(hyp_zero_mask, 0.0)

    if torch.any(ref_zero_mask):
        print(
            "Warning: Empty reference sentence detected; setting raw BERTScores to 0.",
            file=sys.stderr,
        )
        P = P.masked_fill(ref_zero_mask, 0.0)
        R = R.masked_fill(ref_zero_mask, 0.0)

    F = F.masked_fill(torch.isnan(F), 0.0)

    return P, R, F

def bert_cos_score_idf(
    model,
    refs,
    hyps,
    tokenizer,
    idf_dict,
    verbose=False,
    batch_size=64,
    device="cuda:0",
    all_layers=False,
):
    """
    Compute BERTScore.

    Args:
        - :param: `model` : a BERT model in `pytorch_pretrained_bert`
        - :param: `refs` (list of str): reference sentences
        - :param: `hyps` (list of str): candidate sentences
        - :param: `tokenzier` : a BERT tokenizer corresponds to `model`
        - :param: `idf_dict` : a dictionary mapping a word piece index to its
                               inverse document frequency
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """
    preds = []

    def dedup_and_sort(l):
        return sorted(list(set(l)), key=lambda x: len(x.split(" ")), reverse=True)

    sentences = dedup_and_sort(refs + hyps)
    embs = []
    iter_range = range(0, len(sentences), batch_size)
    if verbose:
        print("computing bert embedding.")
        iter_range = tqdm(iter_range)
    stats_dict = dict()
    for batch_start in iter_range:
        sen_batch = sentences[batch_start : batch_start + batch_size]
        embs, masks, padded_idf = get_bert_embedding(
            sen_batch, model, tokenizer, idf_dict, device=device, all_layers=all_layers
        )
        embs = embs.cpu()
        masks = masks.cpu()
        padded_idf = padded_idf.cpu()
        for i, sen in enumerate(sen_batch):
            sequence_len = masks[i].sum().item()
            emb = embs[i, :sequence_len]
            idf = padded_idf[i, :sequence_len]
            stats_dict[sen] = (emb, idf)

    def pad_batch_stats(sen_batch, stats_dict, device):
        stats = [stats_dict[s] for s in sen_batch]
        emb, idf = zip(*stats)
        emb = [e.to(device) for e in emb]
        idf = [i.to(device) for i in idf]
        lens = [e.size(0) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=2.0)
        idf_pad = pad_sequence(idf, batch_first=True)

        def length_to_mask(lens):
            lens = torch.tensor(lens, dtype=torch.long)
            max_len = max(lens)
            base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
            return base < lens.unsqueeze(1)

        pad_mask = length_to_mask(lens).to(device)
        return emb_pad, pad_mask, idf_pad

    device = next(model.parameters()).device
    iter_range = range(0, len(refs), batch_size)
    if verbose:
        print("computing greedy matching.")
        iter_range = tqdm(iter_range)

    with torch.no_grad():
        for batch_start in iter_range:
            batch_refs = refs[batch_start : batch_start + batch_size]
            batch_hyps = hyps[batch_start : batch_start + batch_size]
            ref_stats = pad_batch_stats(batch_refs, stats_dict, device)
            hyp_stats = pad_batch_stats(batch_hyps, stats_dict, device)

            P, R, F1 = greedy_cos_idf(*ref_stats, *hyp_stats, all_layers)
            preds.append(torch.stack((P, R, F1), dim=-1).cpu())
    preds = torch.cat(preds, dim=1 if all_layers else 0)
    return preds


def get_hash(
    model,
    num_layers,
    idf,
    rescale_with_baseline,
    use_custom_baseline,
    use_fast_tokenizer,
):
    msg = "{}_L{}{}_version={}(hug_trans={})".format(
        model, num_layers, "_idf" if idf else "_no-idf", "0.3.12", trans_version
    )
    if rescale_with_baseline:
        if use_custom_baseline:
            msg += "-custom-rescaled"
        else:
            msg += "-rescaled"
    if use_fast_tokenizer:
        msg += "_fast-tokenizer"
    return msg


def cache_scibert(model_type, cache_folder="~/.cache/torch/transformers"):
    if not model_type.startswith("scibert"):
        return model_type

    underscore_model_type = model_type.replace("-", "_")
    cache_folder = os.path.abspath(os.path.expanduser(cache_folder))
    filename = os.path.join(cache_folder, underscore_model_type)

    # download SciBERT models
    if not os.path.exists(filename):
        cmd = f"mkdir -p {cache_folder}; cd {cache_folder};"
        cmd += f"wget {SCIBERT_URL_DICT[model_type]}; tar -xvf {underscore_model_type}.tar;"
        cmd += f"rm -f {underscore_model_type}.tar ; cd {underscore_model_type}; tar -zxvf weights.tar.gz; mv weights/* .;"
        cmd += f"rm -f weights.tar.gz; rmdir weights; mv bert_config.json config.json;"
        print(cmd)
        print(f"downloading {model_type} model")
        os.system(cmd)

    # fix the missing files in scibert
    json_file = os.path.join(filename, "special_tokens_map.json")
    if not os.path.exists(json_file):
        with open(json_file, "w") as f:
            print(
                '{"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]", "mask_token": "[MASK]"}',
                file=f,
            )

    json_file = os.path.join(filename, "added_tokens.json")
    if not os.path.exists(json_file):
        with open(json_file, "w") as f:
            print("{}", file=f)

    if "uncased" in model_type:
        json_file = os.path.join(filename, "tokenizer_config.json")
        if not os.path.exists(json_file):
            with open(json_file, "w") as f:
                print(
                    '{"do_lower_case": true, "max_len": 512, "init_inputs": []}', file=f
                )

    return filename

def process(a, tokenizer=None):
    if tokenizer is not None:
        a = sent_encode(tokenizer, a)
    return set(a)

def get_idf_dict(arr, tokenizer, nthreads=4):
    """
    Returns mapping from word piece index to its inverse document frequency.


    Args:
        - :param: `arr` (list of str) : sentences to process.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `nthreads` (int) : number of CPU threads to use
    """
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process, tokenizer=tokenizer)

    if nthreads > 0:
        with Pool(nthreads) as p:
            idf_count.update(chain.from_iterable(p.map(process_partial, arr)))
    else:
        idf_count.update(chain.from_iterable(map(process_partial, arr)))

    idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
    idf_dict.update(
        {idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()}
    )
    return idf_dict

def get_model(model_type, num_layers, all_layers=None):
    if model_type.startswith("scibert"):
        model = AutoModel.from_pretrained(cache_scibert(model_type))
    elif "t5" in model_type:
        from transformers import T5EncoderModel

        model = T5EncoderModel.from_pretrained(model_type)
    else:
        model = AutoModel.from_pretrained(model_type)
    model.eval()

    if hasattr(model, "decoder") and hasattr(model, "encoder"):
        model = model.encoder

    # drop unused layers
    if not all_layers:
        if hasattr(model, "n_layers"):  # xlm
            assert (
                0 <= num_layers <= model.n_layers
            ), f"Invalid num_layers: num_layers should be between 0 and {model.n_layers} for {model_type}"
            model.n_layers = num_layers
        elif hasattr(model, "layer"):  # xlnet
            assert (
                0 <= num_layers <= len(model.layer)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.layer)} for {model_type}"
            model.layer = torch.nn.ModuleList(
                [layer for layer in model.layer[:num_layers]]
            )
        elif hasattr(model, "encoder"):  # albert
            if hasattr(model.encoder, "albert_layer_groups"):
                assert (
                    0 <= num_layers <= model.encoder.config.num_hidden_layers
                ), f"Invalid num_layers: num_layers should be between 0 and {model.encoder.config.num_hidden_layers} for {model_type}"
                model.encoder.config.num_hidden_layers = num_layers
            elif hasattr(model.encoder, "block"):  # t5
                assert (
                    0 <= num_layers <= len(model.encoder.block)
                ), f"Invalid num_layers: num_layers should be between 0 and {len(model.encoder.block)} for {model_type}"
                model.encoder.block = torch.nn.ModuleList(
                    [layer for layer in model.encoder.block[:num_layers]]
                )
            else:  # bert, roberta
                assert (
                    0 <= num_layers <= len(model.encoder.layer)
                ), f"Invalid num_layers: num_layers should be between 0 and {len(model.encoder.layer)} for {model_type}"
                model.encoder.layer = torch.nn.ModuleList(
                    [layer for layer in model.encoder.layer[:num_layers]]
                )
        elif hasattr(model, "transformer"):  # bert, roberta
            assert (
                0 <= num_layers <= len(model.transformer.layer)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.transformer.layer)} for {model_type}"
            model.transformer.layer = torch.nn.ModuleList(
                [layer for layer in model.transformer.layer[:num_layers]]
            )
        elif hasattr(model, "layers"):  # bart
            assert (
                0 <= num_layers <= len(model.layers)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.layers)} for {model_type}"
            model.layers = torch.nn.ModuleList(
                [layer for layer in model.layers[:num_layers]]
            )
        else:
            raise ValueError("Not supported")
    else:
        if hasattr(model, "output_hidden_states"):
            model.output_hidden_states = True
        elif hasattr(model, "encoder"):
            model.encoder.output_hidden_states = True
        elif hasattr(model, "transformer"):
            model.transformer.output_hidden_states = True
        # else:
        #     raise ValueError(f"Not supported model architecture: {model_type}")

    return model


def get_tokenizer(model_type, use_fast=False):
    if model_type.startswith("scibert"):
        model_type = cache_scibert(model_type)

    if version.parse(trans_version) >= version.parse("4.0.0"):
        tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=use_fast)
    else:
        assert not use_fast, "Fast tokenizer is not available for version < 4.0.0"
        tokenizer = AutoTokenizer.from_pretrained(model_type)

    return tokenizer



def _norm(s: str) -> str:
    """Lightweight medical-term normalization."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    # British ↔ American variants
    s = s.replace("naevus", "nevus").replace("naevi", "nevi").replace("haemangioma", "hemangioma")
    # Common abbreviations
    s = s.replace(" pih", " post-inflammatory hyperpigmentation").replace("pih", "post-inflammatory hyperpigmentation")
    s = s.replace("bcc", "basal cell carcinoma").replace("scc", "squamous cell carcinoma").replace(" sk", " seborrheic keratosis")
    s = s.replace(" ak", " actinic keratosis")
    # Hyphens → spaces, collapse non-alnum
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _alias_map():
    """Alias → canonical string map."""
    groups = [
        # exact pairs requested
        {"scar", "scarring", "cicatrix"},
        {"melanocytic nevus", "melanocytic nevi", "nevus", "nevi", "mole", "moles", "naevus", "naevi"},
        {"epidermal cyst", "epidermoid cyst", "sebaceous cyst", "infundibular cyst"},
        {"angioma", "hemangioma", "cherry angioma", "senile hemangioma"},
        {"acrochordon", "skin tag", "skin tags"},
        # useful dermatology aliases
        {"basal cell carcinoma", "bcc"},
        {"squamous cell carcinoma", "scc"},
        {"seborrheic keratosis", "sk"},
        {"actinic keratosis", "ak"},
        {"post inflammatory hyperpigmentation", "post-inflammatory hyperpigmentation", "pih"},
        {"melasma", "chloasma"},
        {"tinea", "dermatophytosis", "ringworm"},
        {"molluscum contagiosum", "molluscum"},
        {"urticaria", "hives"},
        {"acne vulgaris", "acne"},
        {"psoriasis", "psoriasis vulgaris"},
        {"vitiligo"},
        {"lichen planus"},
        {"impetigo"},
        {"folliculitis"},
        {"paronychia"},
        {"onychomycosis"},
        {"cellulitis"},
        {"abscess"},
        {"rosacea"},
        {"viral wart", "wart", "verruca", "hpv wart"},
        {"Behçet's syndrome", "behcets disease", "behcets","Behçet syndrome"},
        {"hemangioma", "haemangioma"},  # safety duplicate
    ]
    alias2canon = {}
    for g in groups:
        # choose a stable canonical: first sorted normalized term
        norm_group = sorted(_norm(x) for x in g if _norm(x))
        if not norm_group:
            continue
        canon = norm_group[0]
        for x in norm_group:
            alias2canon[x] = canon
    return alias2canon

_ALIAS2CANON = _alias_map()

def _canonical(term: str) -> str:
    n = _norm(term)
    return _ALIAS2CANON.get(n, n)

def judge_close_end_vqa_json(answer: str, response: str) -> bool:
    """
    Return True if `answer` matches response['answer'] exactly after normalization,
    or if both map to the same canonical term via alias groups; otherwise False.
    """
    try:
        pred = json.loads(response).get("answer", None)
    except Exception:
        # try to recover if response has leading/trailing noise
        try:
            start = response.find("{")
            end = response.rfind("}")
            pred = json.loads(response[start:end+1]).get("answer", None) if start != -1 and end != -1 else None
        except Exception:
            pred = None

    if pred is None:
        return False

    a_norm = _norm(answer)
    p_norm = _norm(pred)
    if a_norm == p_norm:
        return True

    a_can = _canonical(a_norm)
    p_can = _canonical(p_norm)
    return a_can == p_can


def parse_json_response(text: str):
    if text is None:
        raise ValueError("response is None")

    s = text.strip()
    if not s:
        raise ValueError("empty response")

    # ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return json.loads(m.group(1))

    # First {...}
    m = re.search(r"(\{.*\})", s, flags=re.DOTALL)
    if m:
        return json.loads(m.group(1))

    return json.loads(s)