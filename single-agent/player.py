import random
from DDPG import DDPGAgent 
from single_agent_mdp import buy, sell
import math
import numpy as np
def convert_state_to_vector(state):
    price, portfolio, cash = state
    vector = [price]
    for stock in portfolio.keys():
        vector.append(portfolio[stock])
    vector.append(cash)
    return vector
class Player():
    def __init__(self, method, init_state, data, stock_name='RELIANCE', threshold=0.5):
        self.method = method
        self.actions = ['buy', 'sell', 'hold']
        self.threshold = threshold
        self.state = {}
        self.init_state = init_state
        self.df = data
        self.stock_name = stock_name
        if method == 'RL':
            self.RL_Train()
    def RL_Train(self):
        """
        Initialize the RL agent.
        """
        def get_current_reward(next_state,current_state):
            pv1 = current_state[0]*current_state[1][self.stock_name] + current_state[2]
            pv2 = next_state[0]*next_state[1][self.stock_name] + next_state[2]
            return pv2 - pv1
        def get_current_price(episode, time):
            return self.df[(self.df['episode'] == episode+1) & (self.df['time'] == time+1)]['Close'].values[0]
        state_dim = 3
        action_dim = 1
        max_action = 5 #MAX NUMBER OF STOCKS TO BE SOLD OR BOUGHT
        self.agent = DDPGAgent(state_dim, action_dim, max_action, threshold=self.threshold)
        for iter in range(10):
            self.state = tuple(self.init_state) 
            self.state[1][self.stock_name] = 0
            for episode in range(240):
                total_reward = 0
                self.agent.noise.reset()
                for t in range(7):
                    action = self.agent.act(convert_state_to_vector(self.state))
                    next_state = self.state
                    current_price = get_current_price(episode, t)
                    if(action[0] >= self.threshold):
                        portfolio, cash = buy(abs(action[0]), current_price, self.state[1], self.state[2],self.stock_name)
                        next_state = (current_price, portfolio, cash)
                    elif(action[0] <= -self.threshold):
                        portfolio, cash = sell(abs(action[0]), current_price, self.state[1], self.state[2],self.stock_name)
                        next_state = (current_price, portfolio, cash)
                    reward = get_current_reward(next_state, self.state)
                    self.agent.replay_buffer.push(convert_state_to_vector(self.state), [action[0]], reward, convert_state_to_vector(next_state))
                    self.agent.update()
                    self.state = next_state
                    total_reward += reward
    def move(self, state=None):
        if self.method == 'random':
            return random.choice(self.actions), random.randint(1, 10)
        if self.method == 'RL':
            if state is None:
                raise ValueError("State must be provided for RL method.")
            action = self.agent.act(convert_state_to_vector(state), add_noise=False)
            if(action >= self.threshold):
                return 'buy', abs(action)
            elif(action <= -self.threshold):
                return 'sell', abs(action)
            else:
                return 'hold', 0
        else:
            raise ValueError("Unknown method: {}".format(self.method))