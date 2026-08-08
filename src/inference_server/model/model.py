

class Model:

    def __init__(self,vocab_size):

        self.vocab_size = vocab_size


    def forward(self,encoded):

        logits = []

        for token_id in range(self.vocab_size):

            a = token_id/self.vocab_size

            logits.append(a)

        return logits   
            