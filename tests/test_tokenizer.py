from inference_server.tokenizer.tokenizer import Tokenizer

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