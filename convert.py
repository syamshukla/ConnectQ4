import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
class Connect4Env:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1  # 1 for Player 1, -1 for Player 2

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        return self.board.copy()

    def step(self, action):
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break

        done, winner = self.check_winner()
        if done:
            reward = 1 if winner == 1 else -1
        else:
            reward = 0

        self.current_player *= -1

        return self.board.copy(), reward, done

    def check_winner(self):
        for player in [1, -1]:
            # Check for a win horizontally
            for row in range(self.rows):
                for col in range(self.cols - 3):
                    if np.all(self.board[row, col:col+4] == player):
                        return True, player

            # Check for a win vertically
            for row in range(self.rows - 3):
                for col in range(self.cols):
                    if np.all(self.board[row:row+4, col] == player):
                        return True, player

            # Check for a win diagonally (from bottom-left to top-right)
            for row in range(3, self.rows):
                for col in range(self.cols - 3):
                    if np.all(self.board[row-3:row+1, col:col+4] == player):
                        return True, player

            # Check for a win diagonally (from top-left to bottom-right)
            for row in range(self.rows - 3):
                for col in range(self.cols - 3):
                    if np.all(self.board[row:row+4, col:col+4].diagonal() == player):
                        return True, player

        # Check for a draw
        if np.all(self.board != 0):
            return True, 0

        return False, 0

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(6 * 7, 128)
        self.dense2 = nn.Linear(128, 7)

    def forward(self, state):
        x = self.flatten(state)
        x = torch.relu(self.dense1(x))
        return self.dense2(x)


class DQNAgent:
    def __init__(self, dqn, target_dqn, replay_memory, gamma=0.99, epsilon=0.1, learning_rate=0.001, batch_size=32):
        self.dqn = dqn
        self.target_dqn = target_dqn
        self.replay_memory = replay_memory
        self.gamma = gamma
        self.epsilon = epsilon
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.batch_size = batch_size

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(7)
        else:
            q_values = self.dqn(torch.FloatTensor(state).view(1, -1))
            return torch.argmax(q_values).item()

    def update_q_network(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))

        q_values = self.dqn(states)
        target_q_values = self.target_dqn(next_states)

        for i in range(self.batch_size):
            if dones[i]:
                q_values[i, actions[i]] = rewards[i]
            else:
                q_values[i, actions[i]] = rewards[i] + self.gamma * torch.max(target_q_values[i])

        loss = nn.MSELoss()(q_values, self.dqn(states))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def play_one(env, agent, tagent, eps, gamma, copy_period, learn=True):
    env.reset()

    observation = env.board.copy().flatten()
    prev_observation = observation.copy()

    done = False
    total_reward = 0

    while not done:
        if learn:
            action = agent.select_action(observation)
        else:
            action = int(input('Player 1 input: '))

        _, reward, done = env.step(action)
        total_reward += reward

        agent.replay_memory.append((prev_observation, action, reward, observation, done))

        if len(agent.replay_memory) > agent.batch_size:
            batch = agent.replay_memory.sample(agent.batch_size)
            agent.update_q_network(batch)

        prev_observation = observation.copy()
        observation = env.board.copy().flatten()

        if done:
            break

        if learn:
            action = agent.select_action(rev(observation))
        else:
            action = int(input('Player 2 input: '))

        _, reward, done = env.step(action)
        total_reward += reward

        agent.replay_memory.append((prev_observation, action, reward, observation, done))

        if len(agent.replay_memory) > agent.batch_size:
            batch = agent.replay_memory.sample(agent.batch_size)
            agent.update_q_network(batch)

        prev_observation = observation.copy()
        observation = env.board.copy().flatten()

        if done:
            break

    return total_reward


def rev(a):
    b = a.copy()
    b[a == 1] = 2
    b[a == 2] = 1
    return b.copy()

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def append(self, experience):
        self.memory.append(experience)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[i] for i in indices]

def update_board():
    print("Current Game Board:")
    for row in env.board:
        print(" ".join(map(str, row)))
    print()

def hello_callback(value):
    global observation, prev_observation

    if env.check_winner()[0] > 0:
        return

    prev_observation = observation.copy().flatten()

    placed = env.step(1, value)
    if not placed:
        return

    prev_observation = observation.copy().flatten()
    observation = env.board.copy().flatten()

    if env.check_winner()[0] > 0:
        print("Player 1 wins!")
        update_board()
        return

    action = agent.select_action(rev(observation))
    placed = env.step(2, action)

    prev_observation = observation.copy().flatten()
    observation = env.board.copy().flatten()

    if env.check_winner()[0] > 0:
        print("Player 2 wins!")
        update_board()
        return

    update_board()
    print(value)

env = Connect4Env()
dqn = DQN()
target_dqn = DQN()
replay_memory = ReplayMemory(capacity=10000)
agent = DQNAgent(dqn, target_dqn, replay_memory)

gamma = 0.99
copy_period = 50

N = 15
total_rewards = np.empty(N)
avg_rewards = []

for n in range(N):
    eps = 0.1
    total_reward = play_one(env, agent, target_dqn, eps, gamma, copy_period)
    total_rewards[n] = total_reward

    if n % 10 == 0:
        avg_reward = total_rewards[max(0, n - 10):(n + 1)].mean()
        avg_rewards.append(avg_reward)
        print("Episode:", n, "Total Reward:", total_reward, "Average Reward (last 10):", avg_reward)

# Console-based gameplay
# Console-based gameplay against the trained agent
while True:
    env.reset()
    update_board()

    user_player = int(input("Choose your player (1 or 2): "))
    if user_player not in [1, 2]:
        print("Invalid player choice. Please choose 1 or 2.")
        continue

    while not env.check_winner()[0] and not np.all(env.board != 0):
        if env.current_player == user_player:
            action = int(input("Your move: "))
        else:
            # Agent's move
            action = agent.select_action(rev(env.board.flatten()))

        _, _, done = env.step(action)
        update_board()

        if done or env.check_winner()[0] > 0:
            break

    if env.check_winner()[0] > 0:
        print(f"Player {env.check_winner()[0]} wins!")
    else:
        print("It's a draw!")

    play_again = input("Do you want to play again? (yes/no): ")
    if play_again.lower() != 'yes':
        print("Game over!")
        break