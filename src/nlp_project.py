from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset
import torch
import random
import os
from multiprocessing import Pool

# To check if we are using GPU

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Download PG-19 dataset

import gdown

url = "https://drive.google.com/uc?id=1-3KMd7n6NiT2m0GMFlbQ_aw70IWzZwTT"
output = "pg19.zip"
if output not in os.listdir("."):
    gdown.download(url, output, quiet=False)

from zipfile import ZipFile

if "metadata.csv" not in os.listdir("."):
    with ZipFile("./pg19.zip", "r") as zip:
        zip.extractall()

test_dir = "./test"
train_dir = "./train"

# test_text = []
# for file in os.listdir(test_dir)[:1]:  # only us the first 5 books
#     with open(os.path.join(test_dir, file), "r") as f:
#         test_text.append(f.read())

# train_text = []
# for file in os.listdir(train_dir)[:20]:  # only uses first 10 train books
#     with open(os.path.join(train_dir, file), "r") as f:
#         train_text.append(f.read())

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def continue_text(text, model, tokenizer, device):
    model.to(device)
    model.eval()
    return tokenizer.decode(
        model.generate(
            tokenizer.encode(text, return_tensors="pt").to(device),
            do_sample=True,
            temperature=0.7,
            max_length=100,
        )[0]
    )


# def tf_tokenize(text):
#     return tokenizer.encode(text, return_tensors="pt")


# with Pool(os.cpu_count()) as p:
#     test_data = p.map(tf_tokenize, test_text)
#     train_data = p.map(tf_tokenize, train_text)


# class Pg19_Dataset(Dataset):
#     def __init__(self, data):  # path=None):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         length = 1024

#         tokenized = torch.squeeze(self.data[idx])
#         label_idx = random.randrange(4, len(tokenized))

#         item = {}
#         tokens = tokenized[max(label_idx - length, 0) : label_idx]

#         item["input_ids"] = torch.zeros((length,))
#         item["input_ids"][: len(tokens)] = tokens
#         item["input_ids"] = item["input_ids"].long()
#         item["labels"] = item["input_ids"]
#         item["attention_mask"] = torch.zeros((length,))
#         item["attention_mask"][: len(tokens)] = 1
#         item["attention_mask"] = item["attention_mask"].half()

#         return item


def size(a):
    return a.element_size() * a.nelement()


class Pg19_Dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.file_names = os.listdir(path)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        length = 1024

        with open(os.path.join(self.path, self.file_names[idx]), "r") as f:
            text = f.read()

        tokenized = tokenizer.encode(text)
        label_idx = random.randrange(4, len(tokenized))

        item = {}
        tokens = torch.tensor(tokenized[max(label_idx - length, 0) : label_idx])
        del tokenized

        item["input_ids"] = torch.zeros((length,))
        item["input_ids"][: len(tokens)] = tokens
        item["input_ids"] = item["input_ids"].long()
        item["labels"] = item["input_ids"]
        item["attention_mask"] = torch.zeros((length,))
        item["attention_mask"][: len(tokens)] = 1
        item["attention_mask"] = item["attention_mask"].half()

        return item


test_dataset = Pg19_Dataset(test_dir)
train_dataset = Pg19_Dataset(train_dir)

print(test_dataset[0])


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    dataloader_num_workers=os.cpu_count(),
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

text = "He was a very sad man in"
print(continue_text(text, model, tokenizer, device))
trainer.train()
print(continue_text(text, model, tokenizer, device))
