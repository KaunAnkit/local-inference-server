
from transformers import AutoTokenizer
import torch

class HFTokenizer:

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M"
        )   
        self.eos_token_id = self.tokenizer.eos_token_id

    def encode(self,text):

        token_list = self.tokenizer(text)["input_ids"]  

        return token_list

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)