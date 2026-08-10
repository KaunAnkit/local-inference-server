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

    result = sampler.sample([1.0, 2.0, 3.0])

    assert result == 0


def test_softmax():

    logits = [1.0, 2.0, 3.0]

    sampler = Sampler()

    result_1 = sampler.softmax(logits)

    assert result_1 == pytest.approx(
        [0.0900, 0.2447, 0.6652],
        rel=1e-3
    )

