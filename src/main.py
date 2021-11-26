from datasets import get_dataset_config_names
from datasets import get_dataset_split_names
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import LEDTokenizer, LEDForConditionalGeneration
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import random

INPUT_MAX = 16384

dataset_name = "pg19"

# val_data = load_dataset(dataset_name, split="train")
# print(val_data.column_names)
# post_1900 = val_data.filter(lambda x: x["publication_date"] >= 1900)
# first_ten = post_1900[:10]
# print(post_1900["publication_date"])
# print(first_ten["publication_date"])


class Pg19_Dataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokenized = self.transform.encode(text, return_tensors="pt")
        # print(tokenized.shape)
        label_idx = random.randrange(len(tokenized[0]))
        # print(tokenized)
        # print(label_idx)

        item = {}
        item["input_ids"] = torch.zeros((1, INPUT_MAX))
        left_boundary = (
            0 if INPUT_MAX - label_idx < 0 else INPUT_MAX - label_idx
        )
        # print(INPUT_MAX)
        # print(label_idx)
        # print(left_boundary)

        item["input_ids"][:, left_boundary:] = tokenized[
            :, max(label_idx - INPUT_MAX, 0) : label_idx
        ]
        item["input_ids"] = item["input_ids"].type(torch.IntTensor)
        """

        amount to pad = 
        with this it would be 75 
        min(max - idx, max)

        idx = 25
        max = 100

        75

        idx = 150
        max = 100
        max - 150 = - 50

        we want it to be either less than the number or greater?

        we want it to either be input_max
        or we want it to be if max - idx < 0, then max, else idx - max

        """
        # print(item["input_ids"].shape)
        # print(item["input_ids"])

        item["global_attention_mask"] = torch.zeros_like(item["input_ids"])
        item["global_attention_mask"][:, 0] = 1
        # item["num_beams"] = 3
        # item["max_length"] = 1
        # item["early_stopping"] = False
        item["labels"] = tokenized[:, label_idx]

        # print(item["input_ids"])
        # print(item["input_ids"].shape)
        # print(item["global_attention_mask"].shape)
        # print(item["labels"].shape)
        return item


tokenizer = LEDTokenizer.from_pretrained("allenai/led-large-16384-arxiv")

test_data = load_dataset(dataset_name, split="test")["text"]
test_data = Pg19_Dataset(test_data, tokenizer)

# for i in range(10):
#     test_data[i]
# assert 2 == 3

# train_data = load_dataset(dataset_name, split="train")["text"]
# train_data = Pg19_Dataset(train_data, tokenizer)

# train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=12)
# test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=12)
model = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv")
one = test_data[1]
print(one.keys())
print(model(one))
assert 2 == 3


training_args = TrainingArguments(
    output_dir="./results",  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir="./logs",  # directory for storing logs
    logging_steps=10,
)

# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    # train_dataset=train_data,  # training dataset
    train_dataset=test_data,  # training dataset
    # eval_dataset=test_data,  # evaluation dataset
)

trainer.train()
"""
okay sooo what do I want to doooooooo
bananananananannanana

Do I want to do this with pytorch or tensorflow.
ultimately I'd like to do it with ??????
I guess pyrorch sounds a bit easier in a way, I'll be using prebuild models for all of this...

So how do they build up these models?
they use some strange compressed transformer thing
https://arxiv.org/pdf/1911.05507.pdf

How does it predict the next thing?
- how long should the label be?
one word
- does it predict one word
yes it predicts one word (lame I know...)
- does it predict multiple words?
no
- if it is multiple words, how do we tell the model how long to make the thing?
- is there an end of sentence tag in the bert tokenization?
- if so we could ahve it use the previous x sentences and summary to complete the next sentence?
- what if the next sententence is really long

- what loss funciton do they use?
cross entropy
- do they pass tokenized data into the model and loss and everyting else is done on tokens?
yes, you do cross entropy on the tokens

- can you get perplexity for predicting a sequence rather than one word?
perplexity is single word

can pass pytorch modules into this...
https://huggingface.co/transformers/custom_datasets.html#fine-tuning-with-trainer 
https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer
"""
