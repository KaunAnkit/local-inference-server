from inference_server.model.model import Model


def test_model():

    model = Model(5)

    result = model.forward([0,1])

    assert result == [0.0, 0.2, 0.4, 0.6, 0.8]