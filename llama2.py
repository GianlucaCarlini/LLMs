# %% import libraries
from transformers import pipeline
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    LlamaTokenizerFast,
)

# %% load llama

model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
# %%

llama = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
# %%

message = [
    {
        "role": "user",
        "content": ("Hi LLama, are you good at summarizing text?"),
    }
]

out = llama(
    message,
    max_length=1000,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

reply = out[0]["generated_text"][-1]["content"]

print(reply)

# %%
