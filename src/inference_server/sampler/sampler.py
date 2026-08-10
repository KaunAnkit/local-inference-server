import math
import random
class Sampler:

    def sample(self,logits):


        probalility  = self.softmax(logits)

        cumulative = 0

        for x in range(len(probalility)):

            cumulative += probalility[x]
            probalility[x] = cumulative

        random_value = random.random()

        for x in range(len(probalility)):

            if probalility[x] >= random_value:

                return x
        

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


        


        