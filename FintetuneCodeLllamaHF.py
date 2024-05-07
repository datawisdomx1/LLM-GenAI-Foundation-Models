#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install --upgrade huggingface_hub


# In[2]:


from huggingface_hub import notebook_login

notebook_login()


# In[ ]:


#huggingface-cli login


# In[ ]:


#!pip install --upgrade transformers datasets peft bitsandbytes flash-attn diffusers scipy numpy pandas


# In[3]:


# Training arguments
MAX_STEPS = 2000  # max_steps
BATCH_SIZE = 4  # batch_size
GR_ACC_STEPS = 1  # gradient_accumulation_steps
LR = 5e-4  # learning_rate
LR_SCHEDULER_TYPE = "cosine"  # lr_scheduler_type
WEIGHT_DECAY = 0.01  # weight_decay
NUM_WARMUP_STEPS = 30  # num_warmup_steps
EVAL_FREQ = 100  # eval_freq
SAVE_FREQ = 100  # save_freq
LOG_FREQ = 25  # log_freq
OUTPUT_DIR = "peft-starcoder-lora-a100"  # output_dir
BF16 = True  # bf16
FP16 = False  # no_fp16

# FIM trasformations arguments
FIM_RATE = 0.5  # fim_rate
FIM_SPM_RATE = 0.5  # fim_spm_rate

# LORA
LORA_R = 8  # lora_r
LORA_ALPHA = 32  # lora_alpha
LORA_DROPOUT = 0.0  # lora_dropout
LORA_TARGET_MODULES = "c_proj,c_attn,q_attn,c_fc,c_proj"  # lora_target_modules

# bitsandbytes config
USE_NESTED_QUANT = True  # use_nested_quant
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"  # bnb_4bit_compute_dtype

SEED = 0


# In[4]:


#MODEL = "bigcode/starcoderbase-1b"  # Model checkpoint on the Hugging Face Hub 
DATASET = "smangrul/hf-stack-v1"  # Dataset on the Hugging Face Hub
DATA_COLUMN = "content"  # Column name containing the code content
MODEL = "bigcode/starcoderbase-1b"
#device = "cuda" # for GPU usage or "cpu" for CPU usage

SEQ_LENGTH = 2048  # Sequence length


# In[5]:


from datasets import load_dataset
import torch
from tqdm import tqdm


dataset = load_dataset(
    DATASET,
    data_dir="data",
    split="train",
    streaming=True,
)

valid_data = dataset.take(4000)
train_data = dataset.skip(4000)
train_data = train_data.shuffle(buffer_size=5000, seed=SEED)


# In[6]:


torch.set_default_device('cuda')


# In[7]:


torch.cuda.is_available()
torch.cuda.device_count()
print(torch.cuda.get_device_name(0))
print('Memory Usage:')
print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


# In[8]:


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
    set_seed,
    BitsAndBytesConfig,
)

set_seed(SEED)


# In[9]:


tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """

    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens


chars_per_token = chars_token_ratio(train_data, tokenizer, DATA_COLUMN)
print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")


# In[10]:


train_data


# In[11]:


import functools
import numpy as np


# Helper function to get token ids of the special tokens for prefix, suffix and middle for FIM transformations.
@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer):
    try:
        FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD = tokenizer.special_tokens_map["additional_special_tokens"][1:5]
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
            tokenizer.vocab[tok] for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD]
        )
    except KeyError:
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = None, None, None, None
    return suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id


## Adapted from https://github.com/bigcode-project/Megatron-LM/blob/6c4bf908df8fd86b4977f54bf5b8bd4b521003d1/megatron/data/gpt_dataset.py
def permute(
    sample,
    np_rng,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    pad_tok_id,
    fim_rate=0.5,
    fim_spm_rate=0.5,
    truncate_or_pad=False,
):
    """
    Take in a sample (list of tokens) and perform a FIM transformation on it with a probability of fim_rate, using two FIM modes:
    PSM and SPM (with a probability of fim_spm_rate).
    """

    # The if condition will trigger with the probability of fim_rate
    # This means FIM transformations will apply to samples with a probability of fim_rate
    if np_rng.binomial(1, fim_rate):

        # Split the sample into prefix, middle, and suffix, based on randomly generated indices stored in the boundaries list.
        boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
        boundaries.sort()

        prefix = np.array(sample[: boundaries[0]], dtype=np.int64)
        middle = np.array(sample[boundaries[0] : boundaries[1]], dtype=np.int64)
        suffix = np.array(sample[boundaries[1] :], dtype=np.int64)

        if truncate_or_pad:
            # calculate the new total length of the sample, taking into account tokens indicating prefix, middle, and suffix
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - len(sample)

            # trancate or pad if there's a difference in length between the new length and the original
            if diff > 0:
                if suffix.shape[0] <= diff:
                    return sample, np_rng
                suffix = suffix[: suffix.shape[0] - diff]
            elif diff < 0:
                suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])

        # With the probability of fim_spm_rateapply SPM variant of FIM transformations
        # SPM: suffix, prefix, middle
        if np_rng.binomial(1, fim_spm_rate):
            new_sample = np.concatenate(
                [
                    [prefix_tok_id, suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    prefix,
                    middle,
                ]
            )
        # Otherwise, apply the PSM variant of FIM transformations
        # PSM: prefix, suffix, middle
        else:

            new_sample = np.concatenate(
                [
                    [prefix_tok_id],
                    prefix,
                    [suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    middle,
                ]
            )
    else:
        # don't apply FIM transformations
        new_sample = sample

    return list(new_sample), np_rng


# In[12]:


from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
import random

# Create an Iterable dataset that returns constant-length chunks of tokens from a stream of text files.


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            fim_rate (float): Rate (0.0 to 1.0) that sample will be permuted with FIM.
            fim_spm_rate (float): Rate (0.0 to 1.0) of FIM permuations that will use SPM.
            seed (int): Seed for random number generator.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        fim_rate=0.5,
        fim_spm_rate=0.5,
        seed=0,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seed = seed

        (
            self.suffix_tok_id,
            self.prefix_tok_id,
            self.middle_tok_id,
            self.pad_tok_id,
        ) = get_fim_token_ids(self.tokenizer)
        if not self.suffix_tok_id and self.fim_rate > 0:
            print("FIM is not supported by tokenizer, disabling FIM")
            self.fim_rate = 0

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        np_rng = np.random.RandomState(seed=self.seed)
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []

            for tokenized_input in tokenized_inputs:
                # optionally do FIM permutations
                if self.fim_rate > 0:
                    tokenized_input, np_rng = permute(
                        tokenized_input,
                        np_rng,
                        self.suffix_tok_id,
                        self.prefix_tok_id,
                        self.middle_tok_id,
                        self.pad_tok_id,
                        fim_rate=self.fim_rate,
                        fim_spm_rate=self.fim_spm_rate,
                        truncate_or_pad=False,
                    )

                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


train_dataset = ConstantLengthDataset(
    tokenizer,
    train_data,
    infinite=True,
    seq_length=SEQ_LENGTH,
    chars_per_token=chars_per_token,
    content_field=DATA_COLUMN,
    fim_rate=FIM_RATE,
    fim_spm_rate=FIM_SPM_RATE,
    seed=SEED,
)
eval_dataset = ConstantLengthDataset(
    tokenizer,
    valid_data,
    infinite=False,
    seq_length=SEQ_LENGTH,
    chars_per_token=chars_per_token,
    content_field=DATA_COLUMN,
    fim_rate=FIM_RATE,
    fim_spm_rate=FIM_SPM_RATE,
    seed=SEED,
)


# In[13]:


from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer

load_in_8bit = False

# 4-bit quantization
compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=USE_NESTED_QUANT,
)

device_map = {"": 0}

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    load_in_8bit=load_in_8bit,
    quantization_config=bnb_config,
    device_map=device_map,
    use_cache=False,  # We will be using gradient checkpointing
    trust_remote_code=True,
    use_flash_attention_2=True,
)


# In[14]:


model = prepare_model_for_kbit_training(model)


# In[15]:


# Set up lora
peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET_MODULES.split(","),
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# In[16]:


train_data.start_iteration = 0


training_args = TrainingArguments(
    output_dir=f"Your_HF_username/{OUTPUT_DIR}",
    dataloader_drop_last=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    max_steps=MAX_STEPS,
    eval_steps=EVAL_FREQ,
    save_steps=SAVE_FREQ,
    logging_steps=LOG_FREQ,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_steps=NUM_WARMUP_STEPS,
    gradient_accumulation_steps=GR_ACC_STEPS,
    gradient_checkpointing=True,
    fp16=FP16,
    bf16=BF16,
    weight_decay=WEIGHT_DECAY,
    push_to_hub=True,
    include_tokens_per_second=True,
)


# In[17]:


trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)

print("Training...")
trainer.train()


# In[ ]:


trainer.push_to_hub()


# In[ ]:


from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/starcoderbase"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)

inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))


# In[ ]:





# In[ ]:


from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
dataset["train"][100]


# In[ ]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


# In[ ]:


small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


# In[ ]:


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)


# In[ ]:


from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")


# In[ ]:


#!pip install evaluate
#!pip install scikit-learn


# In[ ]:


import numpy as np
import evaluate

metric = evaluate.load("accuracy")


# In[ ]:


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# In[ ]:


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")


# In[ ]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)


# In[ ]:


trainer.train()


# In[ ]:





# In[ ]:





# In[ ]:




