# from transformers import pipeline

# # using pipeline API for summarization task
# summarization = pipeline("summarization")
# original_text = """
# Paul Walker is hardly the first actor to die during a production.
# But Walker's death in November 2013 at the age of 40 after a car crash was especially eerie given his rise to fame in the "Fast and Furious" film franchise.
# The release of "Furious 7" on Friday offers the opportunity for fans to remember -- and possibly grieve again -- the man that so many have praised as one of the nicest guys in Hollywood.
# "He was a person of humility, integrity, and compassion," military veteran Kyle Upham said in an email to CNN.
# Walker secretly paid for the engagement ring Upham shopped for with his bride.
# "We didn't know him personally but this was apparent in the short time we spent with him.
# I know that we will never forget him and he will always be someone very special to us," said Upham.
# The actor was on break from filming "Furious 7" at the time of the fiery accident, which also claimed the life of the car's driver, Roger Rodas.
# """
# summary_text = summarization(original_text)[0]['summary_text']
# print("Summary:", summary_text)


# from re import A
# from transformers import GPT2Tokenizer, TFGPT2Model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = TFGPT2Model.from_pretrained('gpt2')
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='tf')
# output = model(encoded_input)
# print(output)

# from transformers import pipeline, set_seed
# generator = pipeline('text-generation', model='gpt2')
# set_seed(42)
# output = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
# print(output)

# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

from datasets import get_dataset_config_names
from datasets import get_dataset_split_names
from datasets import load_dataset

dataset_name = "pg19"
# print(get_dataset_split_names('sent_comp'))
print(get_dataset_config_names(dataset_name))

val_data = load_dataset(dataset_name, split="train")
# print(val_data.info)
# for key in val_data:
#     print(key)
print(val_data.column_names)
post_1900 = val_data.filter(lambda x: x["publication_date"] >= 1900)
first_ten = post_1900[:10]
print(post_1900["publication_date"])
print(first_ten["publication_date"])
# import torch

'''
okay sooo what do I want to doooooooo
bananananananannanana
'''

# from transformers import LEDTokenizer, LEDForConditionalGeneration

# model = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv")
# tokenizer = LEDTokenizer.from_pretrained("allenai/led-large-16384-arxiv")

# ARTICLE_TO_SUMMARIZE = """Transformers (Vaswani et al., 2017) have achieved state-of-the-art
# results in a wide range of natural language tasks including generative
# language modeling (Dai et al., 2019; Radford et al., 2019) and discriminative
# language understanding (Devlin et al., 2019). This success is partly due to
# the self-attention component which enables the network to capture contextual
# information from the entire sequence. While powerful, the memory and computational
# requirements of self-attention grow quadratically with sequence length, making
# it infeasible (or very expensive) to process long sequences.
# To address this limitation, we present Longformer, a modified Transformer
# architecture with a self-attention operation that scales linearly with the
# sequence length, making it versatile for processing long documents (Fig 1). This
# is an advantage for natural language tasks such as long document classification,
# question answering (QA), and coreference resolution, where existing approaches
# partition or shorten the long context into smaller sequences that fall within the
# typical 512 token limit of BERT-style pretrained models. Such partitioning could
# potentially result in loss of important cross-partition information, and to
# mitigate this problem, existing methods often rely on complex architectures to
# address such interactions. On the other hand, our proposed Longformer is able to
# build contextual representations of the entire context using multiple layers of
# attention, reducing the need for task-specific architectures."""
# inputs = tokenizer.encode(ARTICLE_TO_SUMMARIZE, return_tensors="pt")

# # Global attention on the first token (cf. Beltagy et al. 2020)
# global_attention_mask = torch.zeros_like(inputs)
# global_attention_mask[:, 0] = 1

# # Generate Summary
# summary_ids = model.generate(
#     inputs,
#     global_attention_mask=global_attention_mask,
#     num_beams=3,
#     max_length=32,
#     early_stopping=True,
# )

# print(
#     tokenizer.decode(
#         summary_ids[0],
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=True,
#     )
# )
