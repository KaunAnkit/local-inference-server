
class Sampler:

    def sample(self,logits):

        value = logits[0]
        value_index = 0

        for index,logit in enumerate(logits):

            if logit >= value:

                value = logit
                value_index = index


        return value_index

        