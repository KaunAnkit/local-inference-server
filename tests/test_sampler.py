from inference_server.sampler.sampler import Sampler
import pytest
import random


def test_sampling(monkeypatch):

    sampler = Sampler()

    monkeypatch.setattr(
        random,
        "random",
        lambda: 0.05
    )

    result = sampler.sample([1.0, 2.0, 3.0],temperature=1)

    assert result == 0


def test_softmax():

    logits = [1.0, 2.0, 3.0]

    sampler = Sampler()

    result_1 = sampler.softmax(logits)

    assert result_1 == pytest.approx(
        [0.0900, 0.2447, 0.6652],
        rel=1e-3
    )

def test_temperature():
    logits = [1.0, 2.0, 3.0]

    sampler = Sampler()

    result = sampler.apply_temperature(logits, temperature=0.5)

    assert result == [2.0, 4.0, 6.0]

    with pytest.raises(ValueError):
        sampler.apply_temperature(logits, temperature=0)

    with pytest.raises(ValueError):
        sampler.apply_temperature(logits,temperature=-0.3)


def test_top_k():

    logits = [1.0, 2.0, 3.0]
    
    sampler = Sampler()

    result = sampler.top_k(logits, 2)[1]
    result2 = sampler.top_k(logits, 2)[0]

    assert result == [3.0, 2.0]
    assert result == [2,1]


    with pytest.raises(ValueError):
        sampler.top_k(logits, 0)

    with pytest.raises(ValueError):
            sampler.top_k(logits, -1)


