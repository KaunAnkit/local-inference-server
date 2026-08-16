
class Generator:

    def __init__(self,tokenizer,model,sampler):

        self.tokenizer = tokenizer
        self.model = model
        self.sampler = sampler

    def generate(self,prompt,max_new_tokens,temperature=0.5,top_k=50,top_p=0.9,penalty=1.0):

        encoded_input = self.tokenizer.encode(prompt)

        generated_id = []

        past_key_values = None

        for x in range(max_new_tokens):

            if past_key_values is None:

                logits,past_key_values= self.model.forward(encoded_input)
            else:

                logits,past_key_values = self.model.forward([next_token],past_key_values)

            next_token = self.sampler.sample(
                    logits,
                    generated_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    penalty=penalty,
                )

            if next_token == self.tokenizer.eos_token_id:
                break

            
            encoded_input.append(next_token)
            generated_id.append(next_token)

            


            yield self.tokenizer.decode([next_token])

            


    