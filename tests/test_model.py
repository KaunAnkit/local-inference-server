from inference_server.model.model import Model


def test_model_cat():

    model = Model(5)

    result = model.forward([0])

    assert result[3] == 0.9

def test_model_EOS():

    model = Model(5)

    result = model.forward([0, 3])

    assert result[4] == 0.9
    assert result[3] == 0.1

