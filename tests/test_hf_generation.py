from inference_server.generation.generator import Generator
from inference_server.model.hf_model import HFModel
from inference_server.sampler.sampler import Sampler
from inference_server.tokenizer.hf_tokenizer import HFTokenizer
import random
import time


def test_generator():

        start = time.perf_counter()

        tokenizer = HFTokenizer()
        model = HFModel()
        sampler = Sampler()

        generator = Generator(tokenizer,model,sampler)

        for token in generator.generate(
                "Explain what Python is",
                max_new_tokens=50,
                temperature=0.7):
                print(token, end="",flush=True)

        elapsed = time.perf_counter() - start

        

        print(f"Time: {elapsed:.2f}s")
        print(f"Tokens/sec: {50 / elapsed:.2f}")
