import torch.nn as nn
import torch
import numpy as np
import random
from collections import deque
import torch.optim as optim
import torch.nn.functional as F
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x * self.max_action
# Define the Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        state, action, reward, next_state = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward, dtype=np.float32), np.array(next_state)

    def __len__(self):
        return len(self.buffer)
    
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action,  learning_rate = 1e-4, threshold=0.5):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.threshold = threshold
        #self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        #self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(100000)
        self.noise = OUNoise(action_dim)
        self.max_action = max_action
        self.tau = 0.05
        self.gamma = 0.99
        self.batch_size = 64
        self.scale = 1
        self.iteration = 1

    def act(self, state, add_noise=True):
        price, num_stocks, cash = state
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        self.iteration += 1
        if self.iteration % 100 == 0:
            self.scale *= 0.9
        if add_noise:
            noise = self.noise.sample()*2*self.max_action * self.scale
            action = action + noise
        max_value = 10000
        if action >= self.threshold:
            max_value = cash/price
        if action <= -self.threshold:
            max_value = num_stocks
        max_value = min(max_value, self.max_action)
        action = np.clip((action), -1*max_value, max_value)
        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample from replay buffer
        states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)

        # Update Critic
        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states, next_actions.detach())
        target_Q = rewards + (self.gamma * target_Q)
        current_Q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q, target_Q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
