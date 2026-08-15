import math
import random

class Sampler:

    def sample(self,logits,temperature=0.5,top_k = 50):

        logits = self.apply_temperature(logits,temperature)


        top_logits, top_indices = self.top_k(logits, top_k)



        probabilities  = self.softmax(top_logits)

        cumulative = 0

        for x in range(len(probabilities)):

            cumulative += probabilities[x]
            probabilities[x] = cumulative

        random_value = random.random()

        for x in range(len(probabilities)):

            if probabilities[x] >= random_value:

                return top_indices[x]
        

    def softmax(self,logits):

        probabilites = logits.copy()

        sum_of_logits = 0
        data = max(probabilites)

        for x in range(len(probabilites)):

            probabilites[x] -= data

            probabilites[x] = math.exp(probabilites[x])

            sum_of_logits += probabilites[x]


        for x in range(len(probabilites)):

            probabilites[x] = probabilites[x]/sum_of_logits

        return probabilites


    def apply_temperature(self,logits,temperature):

        if temperature <= 0:

            raise ValueError("temperature must be greater than 0")

        return [logit/temperature for logit in logits]
        

    def top_k(self, logits, k):

        top = sorted(
            enumerate(logits),
            key=lambda item: item[1],
            reverse=True
        )[:k]

        top_indices = [idx for idx, _ in top]
        top_logits = [logit for _, logit in top]

        return top_logits, top_indices



        