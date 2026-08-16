import math
import random

class Sampler:

    def sample(self,logits,generated_tokens,temperature=0.5,top_k = 50,top_p = 0.9,penalty = 1):

        if penalty >1 :

            logits = self.apply_repetition_penalty(logits,generated_tokens,penalty)



        logits = self.apply_temperature(logits,temperature)

        if top_k is not None:
            top_logits, top_indices = self.top_k(logits, top_k)
        else:
            top_logits = logits
            top_indices = list(range(len(logits)))

        probabilities  = self.softmax(top_logits)

        if top_p is not None:
            top_p_probs, top_p_prob_indices = self.top_p(probabilities, top_p)
        else:
            top_p_probs = probabilities
            top_p_prob_indices = list(range(len(probabilities)))
        

        cumulative = 0

        total = sum(top_p_probs)
        top_p_probs = [p/total for p in top_p_probs]


        for x in range(len(top_p_probs)):

            cumulative += top_p_probs[x]
            top_p_probs[x] = cumulative

        random_value = random.random()

        for x in range(len(top_p_probs)):

            if top_p_probs[x] >= random_value:

                return top_indices[top_p_prob_indices[x]]

        return top_indices[top_p_prob_indices[-1]]

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

        if k is not None and k <= 0:
            raise ValueError("top_k must be > 0")

        top = sorted(
            enumerate(logits),
            key=lambda item: item[1],
            reverse=True
        )[:k]

        top_indices = [idx for idx, _ in top]
        top_logits = [logit for _, logit in top]

        return top_logits, top_indices


    def top_p(self,probabilites,p):

        if p <= 0 or p > 1:
            raise ValueError("top_p must be in (0, 1]")

        top = sorted(
                    enumerate(probabilites),
                    key=lambda item: item[1],
                    reverse=True
                )

        top_logits = []
        top_indices = []

        cnt = 0
        cnt2 = 0

        while cnt <= p and cnt2 < len(top):
            
            top_logits.append(top[cnt2][1])
            cnt+= top[cnt2][1]
            top_indices.append(top[cnt2][0])
            cnt2+=1

        return top_logits,top_indices


    def apply_repetition_penalty(
            self,
            logits,
            generated_tokens,
            penalty
    ):
        if penalty < 1:
            raise ValueError("The value should be greater >= 1")

        logits= logits.copy()
        
        for token in generated_tokens:

            if 0 <= token < len(logits) :

                logits[token] /= penalty

        return logits