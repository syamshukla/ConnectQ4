import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Connect4Env:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1  # 1 for Player 1, -1 for Player 2

    # resets the game to the starting state, board with all 0s
    def reset(self): 
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        return self.board.copy()
    # executes a move for player
    def step(self, action):
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break
        # checks for win or loss
        done, winner = self.check_winner()
        if done:
            reward = 1 if winner == 1 else -1
        else:
            # else draw
            reward = 0
        # toggle player turn
        self.current_player *= -1

        return self.board.copy(), reward, done

    # checking for all the win conditions
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
            
            # Check for a win diagonally (from top-right to bottom-left)
            for row in range(self.rows - 3):
                for col in range(3, self.cols):
                    if self.board[row, col] == player and \
                    self.board[row + 1, col - 1] == player and \
                    self.board[row + 2, col - 2] == player and \
                    self.board[row + 3, col - 3] == player:
                        return True, player


        # Check for a draw
        if np.all(self.board != 0):
            return True, 0

        return False, 0

# Deep Q Network
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.flatten = nn.Flatten()
        # first layer with 128 units
        self.dense1 = nn.Linear(6 * 7, 128)
        # second layer with 7 units
        self.dense2 = nn.Linear(128, 7)

    def forward(self, state):
        x = self.flatten(state)
        x = torch.relu(self.dense1(x))
        return self.dense2(x)

#  Deep Q Network Agent
# want to implement episilon decay method
# inspo from ->  https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class DQNAgent:
    def __init__(self, dqn, target_dqn, replay_memory, gamma=0.99, epsilon=0.1, learning_rate=0.001, batch_size=128):
        self.dqn = dqn
        self.target_dqn = target_dqn
        self.replay_memory = replay_memory
        self.gamma = gamma
        self.epsilon = epsilon
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.batch_size = batch_size

    # epsilon greedy policy
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(7)
        else:
            q_values = self.dqn(torch.FloatTensor(state).view(1, -1))
            return torch.argmax(q_values).item()

    # updating the network based on a batch of experiences
    # we have batch size of 32, can try 128 too
    def update_q_network(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))

        q_values = self.dqn(states)
        # calculating the target q values
        target_q_values = self.target_dqn(next_states)

        # calculates the loss
        # updating the network using backpropogation
        for i in range(self.batch_size):
            if dones[i]:
                q_values[i, actions[i]] = rewards[i]
            else:
                q_values[i, actions[i]] = rewards[i] + self.gamma * torch.max(target_q_values[i])

        loss = nn.MSELoss()(q_values, self.dqn(states))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# single game episode
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

    # random sampling of batches for training
    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[i] for i in indices]
    

# reversing the sides
# allows the AI to play as player2 by swapping the roles of the players
def rev(a):
    b = a.copy()
    b[a == 1] = 2
    b[a == 2] = 1
    return b.copy()

# display game board and let player choose their move
def update_board():
    print("Current Game Board:")
    for row in env.board:
        print("|", end=" ")
        for cell in row:
            if cell == 1:  # Assuming 1 represents player 1's token
                print("X", end=" ")
            elif cell == -1:  # Assuming 2 represents player 2's token
                print("O", end=" ")
            else:
                print(" ", end=" ")
        print("|")
    print("| 1 2 3 4 5 6 7 |")
    print()


# ----------------------------------------------------------------------
# Console based game and DQN initialization
# ----------------------------------------------------------------------
    
env = Connect4Env()
dqn = DQN()
target_dqn = DQN()
replay_memory = ReplayMemory(capacity=10000)
agent = DQNAgent(dqn, target_dqn, replay_memory)

gamma = 0.99
copy_period = 50

N = 1000
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
            action -= 1
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
