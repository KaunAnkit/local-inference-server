from inference_server.sampler.sampler import Sampler


def test_sampler():

    a = [0.0, 0.2, 0.4, 0.6, 0.8] 
    b = [0.9, 0.2, 0.8] 

    sampler = Sampler()

    result_1 = sampler.sample(a)
    result_2 = sampler.sample(b)

    assert result_1 == 4
    assert result_2 == 0