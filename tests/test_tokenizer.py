from inference_server.tokenizer.tokenizer import Tokenizer
from inference_server.tokenizer.hf_tokenizer import HFTokenizer


vocab = {"Hello":0,
            "World":1,
            "Hi":2,
            "Cat":3}

def test_encode():
    
    model = Tokenizer(vocab)

    result = model.encode("Hello World")

    assert result == [0,1]


def test_decode():

    model = Tokenizer(vocab)

    result = model.decode([0,1])

    assert result == "Hello World"

def test_round_trip():

    model = Tokenizer(vocab)
    
    result = model.encode("Hello World")

    result = model.decode(result)
    
    assert result == "Hello World"

def test_eos_token():

    model = Tokenizer(vocab)

    result = model.eos_token_id

    assert result == 4

def test_tokenizer_loads():
    tokenizer = HFTokenizer()

    assert tokenizer.tokenizer is not None

def test_hf_encode():

    tokenizer = HFTokenizer()

    result = tokenizer.encode("Hello world")

    print(result)

    assert isinstance(result, list)
    assert all(isinstance(token_id, int) for token_id in result)
    assert len(result) > 0

def test_hf_round_trip():
    tokenizer = HFTokenizer()

    text = "Hello world"

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text