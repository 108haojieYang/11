import gym
import numpy as np
import torch
from torch.nn import Module, Linear
import matplotlib.pyplot as plt
def exp_moving_avg(arr, beta=0.9):
     n = arr.shape[0]
     mov_avg = np.zeros(n)
     mov_avg[0] = (1-beta) * arr[0]
     for i in range(1, n):
         mov_avg[i] = beta * mov_avg[i-1] + (1-beta) * arr[i]
     return mov_avg
class Policy_net(Module):
      def __init__(self, state_dim, n_actions, n_hidden):
          super(Policy_net, self).__init__()
          self.state_dim = state_dim
          self.n_actions = n_actions
          self.fc1 = Linear(state_dim, n_hidden)
          self.fc2 = Linear(n_hidden, n_actions)
      def forward(self, X):
          X = self.fc1(X)
          X = torch.nn.LeakyReLU()(X)
          X = self.fc2(X)
          X = torch.nn.Softmax(dim=0)(X)
          return X
      
      def select_action(self, curr_state):
         actions_prob_dist = self.forward(torch.Tensor(curr_state))
         selected_act = np.random.choice(self.n_actions, p=actions_prob_dist.
data.numpy())
         return selected_act
def loss_fn(probs, r):
    return -1 * torch.sum(r * torch.log(probs))
state_dim = 4
n_actions = 2
n_hidden = 150
n_episodes = 700
max_episode_len = 500
lr = 0.009
discount = 0.99
history_episode_len = np.zeros(n_episodes)
agent = Policy_net(state_dim, n_actions, n_hidden)
optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
for episode in range(n_episodes):
    env = gym.make('CartPole-v1', new_step_api=True, render_mode=None)
    state = env.reset()
    transitions = []
    for t in range(max_episode_len):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        transitions.append((state, action, t+1))
        finished = terminated or truncated
        if finished:
            break
        state = next_state
    episode_len = len(transitions)
    history_episode_len[episode] = episode_len
    state_batch = torch.Tensor(np.array([s for (s,a,r) in transitions]))
    action_batch = torch.Tensor([a for (s,a,r) in transitions])
    reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=
(0,))
    pred_batch = agent(state_batch)
    prob_batch = pred_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze()
    disc_return = torch.pow(discount, torch.arange(episode_len).float())*reward_batch
    disc_return /= disc_return.max()
    loss = loss_fn(prob_batch, disc_return)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Episode: {episode+1}, Episode Length: {episode_len}")
    env.close()
plt.figure(figsize=(14,7))
plt.plot(exp_moving_avg(history_episode_len, 0.8));
plt.xlabel("Episode", fontsize=25);
plt.ylabel("Length", fontsize=25);
plt.grid(True);
n_episodes = 10
for episode in range(n_episodes):
    env = gym.make('CartPole-v1', new_step_api=True, render_mode="human")
    state = env.reset()
    episode_len = 0
    for t in range(max_episode_len):
        episode_len += 1
        action = agent.select_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        finished = terminated or truncated
        if finished:
            break
    print(f"Episode: {episode+1}, Episode Length: {episode_len}")
    env.close()
