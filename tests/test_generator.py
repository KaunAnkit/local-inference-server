from inference_server.generation.generator import Generator
from inference_server.model.model import Model
from inference_server.sampler.sampler import Sampler
from inference_server.tokenizer.tokenizer import Tokenizer
import random


def test_generator(monkeypatch):

        vocab = {"Hello":0,
        "World":1,
        "Hi":2,
        "Cat":3,
        "<EOS>":4}

        random_values = iter([0.5,0.9])

        monkeypatch.setattr(
                random,
                "random",
                lambda: next(random_values)

        )


        tokenizer = Tokenizer(vocab)
        model = Model(5)
        sampler = Sampler()

        generator = Generator(tokenizer,model,sampler)

        result = generator.generate("Hello",max_new_tokens=3,temperature=0.5)

        assert result == "Cat"

