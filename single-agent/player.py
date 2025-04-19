import random
class Player():
    def __init__(self, method):
        self.method = method
        self.actions = ['buy', 'sell', 'hold']
    def move(self):
        if self.method == 'random':
            return random.choice(self.actions), random.randint(1, 10)
        else:
            raise ValueError("Unknown method: {}".format(self.method))