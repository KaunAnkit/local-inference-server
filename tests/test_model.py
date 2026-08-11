from inference_server.model.model import Model
from inference_server.model.hf_model import HFModel

def test_model_cat():

    model = Model(5)

    result = model.forward([0])

    assert result[3] == 0.9

def test_model_EOS():

    model = Model(5)

    result = model.forward([0, 3])

    assert result[4] == 0.9
    assert result[3] == 0.1

def test_model_loads():
    model = HFModel()

    assert model.model is not None

def test_model_output():

    model = HFModel()

    result = model.forward([1,2,3])


    assert isinstance(result, list)
    assert len(result) == 49152