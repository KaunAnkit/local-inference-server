
class Generator:

    def __init__(self,tokenizer,model,sampler):

        self.tokenizer = tokenizer
        self.model = model
        self.sampler = sampler

    def generate(self,prompt,max_new_tokens,temperature=0.5):

        encoded_input = self.tokenizer.encode(prompt)

        generated_id = []

        for x in range(max_new_tokens):

            logits = self.model.forward(encoded_input)

            next_token = self.sampler.sample(logits,temperature)

            if next_token == self.tokenizer.eos_token_id:
                break

            encoded_input.append(next_token)
            generated_id.append(next_token)

            


        return self.tokenizer.decode(generated_id)



    