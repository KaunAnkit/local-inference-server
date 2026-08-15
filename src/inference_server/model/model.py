

class Model:

    def __init__(self,vocab_size):

        self.vocab_size = vocab_size


    def forward(self,encoded):

        logits = [0.1] * self.vocab_size

        if encoded[-1] == 0:

            logits[3] = 0.9

        if encoded[-1] == 3:

            logits[4] = 0.9

        return logits , None

        #[0.0,0.2,0.3,0.4]  
        