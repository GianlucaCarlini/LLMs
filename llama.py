# %% import

from transformers import pipeline
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    LlamaTokenizerFast,
)

# %% load llama

tokenizer = AutoTokenizer.from_pretrained("./llama3.2-1B")
model = LlamaForCausalLM.from_pretrained("./llama3.2-1B")

# %%

llama = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device="cuda",
    max_length=1000,
)

# %% generate text

message = [
    {
        "role": "user",
        "content": "Hi LLAMA, can you do a tarot cart reading for me? I got the chariot, the magician, and the strength card. What does this mean?",
    }
]

tokenized_chat = tokenizer.apply_chat_template(
    message, tokenize=True, add_generation_prompt=True, return_tensors="pt"
)


# out = model.generate(tokenized_chat, max_new_tokens=1000)
# terminators = [
#     llama.tokenizer.eos_token_id,
#     llama.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
# ]

# out = llama(
#     message,
#     max_length=1000,
#     do_sample=True,
#     temperature=0.6,
#     top_p=0.9,
# )


# %%

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
tokenized_chat = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
)
