import numpy as np
import pandas as pd
import torch
import transformers
import huggingface_hub
from datasets import Dataset, load_metric
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)
from translate.storage.tmx import tmxfile

TMX_FILE_NAME = 'Letters.tmx'
MODEL_CHEKPOINT = "Helsinki-NLP/opus-mt-ru-en"
BATCH_SIZE = 16
TRAIN_EPOCHS = 5
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
SOURCE_LANG = "ru"
TARGET_LANG = "en"
MODEL_NAME = MODEL_CHEKPOINT.split("/")[-1]


huggingface_hub.login()
metric = load_metric("sacrebleu")

with open(TMX_FILE_NAME, 'rb') as fin:
    tmx_file = tmxfile(fin, 'ru', 'en')
    source =[]
    translation=[]
    for node in tmx_file.unit_iter():
        source.append(node.source)
        translation.append(node.target)

dataset = pd.DataFrame(data=(source,translation))
dataset = dataset.transpose()
dataset.columns = [SOURCE_LANG,TARGET_LANG]
data_dict = dataset.to_dict(orient='list')
ds = Dataset.from_dict(data_dict)
ds = ds.train_test_split(test_size=0.2, shuffle=True)

def preprocess_function(examples):
    inputs = examples["ru"]
    targets = examples["en"]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHEKPOINT)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHEKPOINT)

tokenized_datasets = ds.map(preprocess_function,batched=True)

args = Seq2SeqTrainingArguments(
    f"{MODEL_NAME}-finetuned-{SOURCE_LANG}-to-{TARGET_LANG}-letter",
    evaluation_strategy = "epoch",
    learning_rate=4e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=10,
    num_train_epochs=TRAIN_EPOCHS,
    predict_with_generate=True,
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
