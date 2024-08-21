import torch
from transformers import pipeline
import os
os.environ["HF_TOKEN"] = "Your Token"
def prompt(debt_amount,due_days,last_contact_date,collection):
    text = "Write a mail to borrower whose debt amount is Rs {} and its due from last {} days, last contact date was {} with the intention of {}".format(debt_amount,due_days,last_contact_date,collection)
    pipe = pipeline(
        "text-generation",
        model="google/gemma-2-2b-it",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device = "cuda",
    )

    messages = [
        {"role": "user", "content": text},
    ]

    # outputs = pipe(messages, max_new_tokens=256)
    outputs = pipe(messages, max_new_tokens=400)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    return assistant_response

# Ahoy, matey! I be Gemma, a digital scallywag, a language-slingin' parrot of the digital seas. I be here to help ye with yer wordy woes, answer yer questions, and spin ye yarns of the digital world.  So, what be yer pleasure, eh? 🦜
#bhaiii itna kya heavy hai ye
if __name__  == "__main__":
    pass