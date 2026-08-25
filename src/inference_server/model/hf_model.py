
from transformers import AutoModelForCausalLM
import torch

class HFModel:

    def __init__(self,model_name = "HuggingFaceTB/SmolLM2-135M"):

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        )

    def forward(self,encoded,past_key_values=None):

        encoded = torch.tensor([encoded])

        with torch.inference_mode():
            output = self.model(encoded,
                                past_key_values=past_key_values,
                                use_cache=True)

        return (output.logits[0, -1, :].tolist(),
                output.past_key_values)