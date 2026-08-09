
class Tokenizer:

    def __init__(self,vocab: dict):

        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        

    def encode(self,token):

        token_list = token.split(" ")

        encoded_token = []

        for tokens in token_list:

            a = self.vocab[tokens]
            encoded_token.append(a)

        return encoded_token
        

    def decode(self,token_list):

        decoded_token = []

        
        for tokens in token_list:

            a = self.reverse_vocab[tokens]
            decoded_token.append(a)

        return " ".join(decoded_token)


    

        