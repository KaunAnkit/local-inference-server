
from transformers import AutoModelForCausalLM
import torch

class HFModel:

    def __init__(self):

        self.model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M"
        )

    def forward(self,encoded):

        encoded = torch.tensor([encoded])

        with torch.no_grad():
            output = self.model(encoded)

        return output.logits[0, -1, :].tolist()