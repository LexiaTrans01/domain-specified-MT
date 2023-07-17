import os
import re
import huggingface_hub
import nltk
import numpy as np
import pandas as pd
import sentencepiece
import torch
import transformers
from datasets import Dataset, load_dataset, load_metric
from tqdm import notebook
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, MarianMTModel,
                          MarianTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)
from translate.storage.tmx import tmxfile

TMX_FILE_NAME = 'Letters.tmx'
FILE_NAME_CSV = 'letter.csv'
model_checkpoint = "Helsinki-NLP/opus-mt-ru-en"
metric = load_metric("sacrebleu")

huggingface_hub.login()

with open(TMX_FILE_NAME, 'rb') as fin:
    tmx_file = tmxfile(fin, 'ru', 'en')
    source =[]
    translation=[]
    for node in tmx_file.unit_iter():
        source.append(node.source)
        translation.append(node.target)

dataset = pd.DataFrame(data=(source,translation))
dataset = dataset.transpose()
dataset.columns =['ru','en']
#dataset.to_csv(FILE_NAME_CSV,sep=';',index=None)

#data_dict = dataset.to_dict(orient='list')
#ds = Dataset.from_dict(data_dict)
ds = load_dataset('csv', data_files=FILE_NAME_CSV,delimiter=';',split='train')
ds = ds.train_test_split(test_size=0.2, shuffle=True)

max_input_length = 128
max_target_length = 128
source_lang = "ru"
target_lang = "en"

def preprocess_function(examples):
    inputs = examples["ru"]
    targets = examples["en"]
    model_inputs = tokenizer(inputs, max_length=max_input_length)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

tokenized_datasets = ds.map(preprocess_function,batched=True)

batch_size = 16
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-{source_lang}-to-{target_lang}-lett",
    evaluation_strategy = "epoch",
    learning_rate=1e-3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20,
    predict_with_generate=True,
    report_to='wandb',
    push_to_hub=True
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    if eval_preds is None:
        raise ValueError("No evaluation predictions provided.")
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

old_collator = trainer.data_collator
trainer.data_collator = lambda data: dict(old_collator(data))

trainer.train()
trainer.save_model('letter')