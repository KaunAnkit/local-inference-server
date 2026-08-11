from inference_server.generation.generator import Generator
from inference_server.model.hf_model import HFModel
from inference_server.sampler.sampler import Sampler
from inference_server.tokenizer.hf_tokenizer import HFTokenizer
import random


def test_generator():

        tokenizer = HFTokenizer()
        model = HFModel()
        sampler = Sampler()

        generator = Generator(tokenizer,model,sampler)

        result = generator.generate("Hello",max_new_tokens=3,temperature=0.5)

        print(result)


