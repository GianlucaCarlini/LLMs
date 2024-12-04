# %% import
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from tqdm import tqdm
from models.lightning_models import BERT
from pytorch_lightning import Trainer

# %% load data

file = "./Data/jigsaw-toxic-comment-classification-challenge/train.csv/train.csv"

df = pd.read_csv(file)

# convert the last columns to a list
df["list"] = df[df.columns[2:]].values.tolist()
new_df = df[["comment_text", "list"]].copy()

# %% dataset


class BertDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        super().__init__()

        self.dataframe = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.dataframe.list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.targets[index], dtype=torch.float),
        }


# %% split data

train_ds = new_df.sample(frac=0.8, random_state=42)
val_ds = new_df.drop(train_ds.index).reset_index(drop=True)
train_ds = train_ds.reset_index(drop=True)

# %% datasets

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = BertDataset(train_ds, tokenizer, 200)
val_dataset = BertDataset(val_ds, tokenizer, 200)

# %% params

train_params = {"batch_size": 32, "shuffle": True, "num_workers": 0}

val_params = {"batch_size": 32, "shuffle": True, "num_workers": 0}

train_dataloader = DataLoader(train_dataset, **train_params)
val_dataloader = DataLoader(val_dataset, **val_params)

# %% model

model = BERT("bert-base-uncased", num_classes=6, lr=1e-5)

# %% training

trainer = Trainer(max_epochs=1)

trainer.fit(model, train_dataloader, val_dataloader)
