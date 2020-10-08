from tqdm.auto import tqdm
import numpy as np
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
)
import torch
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score
import click

flatten = lambda l: [item for sublist in l for item in sublist]


class MGBDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, length):
        self.encodings = encodings
        self.labels = labels
        self.length = length

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        target = torch.zeros(self.length)
        source = torch.tensor(self.labels[idx])
        target[1:len(source)+1] = source
        item["labels"] = target.long()
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    labels_l = []
    preds_l = []
    for i, label in enumerate(labels):
        # remove padding
        labels_l.append([l for l in label if l != 0])
    for i, pred_item in enumerate(pred.predictions):
        preds_l.append(pred_item.argmax(axis=1)[: len(labels_l[i])])
    filter_labels = ["<none>", "<dots>", "<pad>"]
    used_labels = [l for l in id_map.keys() if l not in filter_labels]
    filter_labels = [id_map[l] for l in filter_labels]
    labels, preds = flatten(labels_l), flatten(preds_l)
    f1s = f1_score(labels, preds, average=None)
    f1_dict = {}
    f1_filtered = []
    for i, f in enumerate(f1s):
        if label_map[i] in used_labels:
            f1_dict[f"f1_{label_map[i]}"] = f
            f1_filtered.append(f)
    f1_dict["f1"] = np.mean(f1_filtered)
    return f1_dict

def read_mgb(split_file, start_index=1, max_lines=float('inf')):
    r_text = []
    r_labels = []
    with open(split_file, "r") as f:
        for i, line in tqdm(enumerate(f)):
            if i >= max_lines:
                break
            tokens = [t.lower() for t in line.split()[start_index:]]
            text = [t for t in tokens if "<" not in t]
            labels = ["<none>"] * len(text)
            i_off = 0
            for i, token in enumerate(tokens):
                if "<" in token:
                    labels[i - 1 - i_off] = token
                    i_off += 1
            r_text.append(text)
            r_labels.append(labels)

    return r_text, r_labels

id_map, label_map = None, None

@click.command()
@click.option("--name", prompt="Run Name", help="The name to log to wandb")
@click.option(
    "--train-set",
    help="train set",
    default="train.txt",
)
@click.option("--lm-set", help="the language model", default=None)
@click.option("--lm-length", help="the language model length", default=162_260) # = length of train
@click.option("--lm-epochs", help="epochs to train the lm for, -1 to combine with train", default=-1)
@click.option("--test-set", help="test set", default="dev.txt")
@click.option("--model", help="transformers model", default="distilbert-base-uncased")
@click.option("--pad-length", help="length to pad to", type=int, default=256)
@click.option("--epochs", help="number of epochs", type=int, default=5)
@click.option("--warmup-steps", help="number of warmup steps", type=int, default=500)
@click.option("--batch-size", help="batch size", type=int, default=64)
@click.option(
    "--gradient-accumulation-steps",
    help="gradient accumulation steps",
    type=int,
    default=1,
)
@click.option("--learning-rate", type=float, default=5e-5)
@click.option("--weight-decay", type=float, default=0.1)
def train(**kwargs):
    global id_map, label_map

    wandb.init(project="APAuT", name=kwargs["name"])

    if kwargs["lm_set"] is not None:
        lm_texts, lm_labels = read_mgb(kwargs["train_set"], 0, kwargs["lm_length"])
    train_texts, train_labels = read_mgb(kwargs["train_set"])
    if kwargs["lm_set"] is not None and kwargs["lm_epochs"] == -1:
        train_texts += lm_texts
        train_labels += lm_labels
    test_texts, test_labels = read_mgb(kwargs["test_set"])

    label_types = np.unique(np.array(flatten(train_labels)))
    encode = lambda l: [np.where(label_types == item)[0][0] + 1 for item in l]
    encode_all = lambda l: [encode(item) for item in l]
    join_all = lambda l: [" ".join(item) for item in l]

    train_labels = encode_all(train_labels)
    test_labels = encode_all(test_labels)
    train_texts = join_all(train_texts)
    test_texts = join_all(test_texts)

    tokenizer = AutoTokenizer.from_pretrained(kwargs["model"], use_fast=True)                       

    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, pad_to_multiple_of=kwargs['pad_length']
    )
    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True, pad_to_multiple_of=kwargs['pad_length']
    )

    if len(train_encodings["input_ids"][0]) != len(test_encodings["input_ids"][0]):
        raise ValueError(
            f"""
                          train length with padding is {len(train_encodings['input_ids'][0])} 
                          while test length is {len(test_encodings['input_ids'][0])}
                          """
        )
    else:
        length = len(train_encodings["input_ids"][0])

    train_dataset = MGBDataset(train_encodings, train_labels, length)
    test_dataset = MGBDataset(test_encodings, test_labels, length)

    kwargs["effective_batch_size"] = kwargs["gradient_accumulation_steps"] * kwargs["batch_size"]

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=kwargs["epochs"],
        per_device_train_batch_size=kwargs["batch_size"],
        per_device_eval_batch_size=kwargs["batch_size"],
        logging_dir="./logs",
        logging_steps=10,
        eval_steps=50_000 // kwargs["effective_batch_size"],
        evaluate_during_training=True,
        gradient_accumulation_steps=kwargs["gradient_accumulation_steps"],
    )

    label_map = {i + 1: label for i, label in enumerate(label_types)}
    label_map[0] = "<pad>"
    id_map = {label: i + 1 for i, label in enumerate(label_types)}
    id_map["<pad>"] = 0

    config = AutoConfig.from_pretrained(
        kwargs['model'],
        num_labels=len(label_types),
        id2label=label_map,
        label2id=id_map,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        kwargs["model"], config=config
    )

    optimizer = AdamW(
        [
            {"params": model.base_model.parameters()},
            {"params": model.classifier.parameters()},  #'lr': 1e-3}
        ],
        lr=kwargs["learning_rate"],
        weight_decay=kwargs["weight_decay"],
    )

    total_steps = len(train_dataset) // kwargs["effective_batch_size"]
    total_steps = total_steps * kwargs["epochs"]
    schedule = get_cosine_schedule_with_warmup(
        optimizer, kwargs["warmup_steps"], total_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, schedule),
    )

    wandb.config.update(kwargs)
    trainer.train()

if __name__ == "__main__":
    train()