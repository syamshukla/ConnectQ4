import random
from matplotlib import pyplot as plt
import numpy as np
import torch

#need to create a deep q network to train an agent to play connect 4 via command line reinforcement learning. 
#The game is played on a 6x7 grid, and the agent is trained to play against a clone of itself.
#both agents are trained using the same deep q network, but with different experiences.
# both expirences get used to train the network, and the network is updated after every game.

class Connect4Env:

    def __init__(self):
        self.board = np.zeros((6, 7))
        self.player = 1
        self.done = False
        self.winner = None
        self.episode_reward = 0
        self.number_of_steps_in_episode = 0  # Track
    def reset(self):
        self.board = np.zeros((6, 7))
        self.player = 1
        self.done = False
        self.winner = None
        return self.board

    def step(self, action):
        if self.done:
            return self.board, 0, self.done, self.winner

        for i in range(5, -1, -1):
            if self.board[i, action] == 0:
                self.board[i, action] = self.player
                break

        if self._is_winner():
            self.done = True
            self.winner = self.player
            reward = 1
        elif self._is_draw():
            self.done = True
            reward = 0
        else:
            self.player = -1 if self.player == 1 else 1
            reward = 0

        return self.board, reward, self.done, self.winner

    def _is_winner(self):
        # Check horizontal
        for i in range(6):
            for j in range(4):
                if self.board[i, j] == self.player and self.board[i, j + 1] == self.player and self.board[i, j + 2] == self.player and self.board[i, j + 3] == self.player:
                    return True

        # Check vertical
        for i in range(3):
            for j in range(7):
                if self.board[i, j] == self.player and self.board[i + 1, j] == self.player and self.board[i + 2, j] == self.player and self.board[i + 3, j] == self.player:
                    return True

        # Check positive diagonal
        for i in range(3):
            for j in range(4):
                if self.board[i, j] == self.player and self.board[i + 1, j + 1] == self.player and self.board[i + 2, j + 2] == self.player and self.board[i + 3, j + 3] == self.player:
                    return True

        # Check negative diagonal
        for i in range(3, 6):
            for j in range(4):
                if self.board[i, j] == self.player and self.board[i - 1, j + 1] == self.player and self.board[i - 2, j + 2] == self.player and self.board[i - 3, j + 3] == self.player:
                    return True
                
        return False
    
    def _is_draw(self):
        return np.all(self.board != 0)

    def render(self):
        for row in self.board:
            print("|", end=" ")
            for cell in row:
                if cell == 1:
                    print("\033[91mX\033[0m", end=" ")  # Use ANSI escape codes for red
                elif cell == -1:
                    print("\033[93mO\033[0m", end=" ")  # Use ANSI escape codes for yellow
                else:
                    print(" ", end=" ")
            print("|")
        print("| 1 2 3 4 5 6 7 |")
    
    def play(self, agent1, agent2):
        self.reset()
        self.episode_reward = 0
        while not self.done:
            if self.player == 1:
                action = agent1.act(self.board)
            else:
                action = agent2.act(self.board)
            _, reward, done, _ = self.step(action)
            if hasattr(agent1, 'update_memory'):
                agent1.update_memory(self.board, action, reward, self.board, done)
            if hasattr(agent2, 'update_memory'):
                agent2.update_memory(self.board, action, -reward, self.board, done)
            
            # agent1.update_memory(self.board, action, reward, self.board, done)
            # agent2.update_memory(self.board, action, -reward, self.board, done)
            self.episode_reward += reward
            self.number_of_steps_in_episode += 1
        average_episode_reward = self.episode_reward / self.number_of_steps_in_episode
        return self.winner, average_episode_reward
    
    def get_board_dimensions(self):
        return self.board.shape

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def act(self, state):
        valid_actions = self.get_valid_actions(state)
        if not valid_actions:
            return -1  # No valid actions available
        return np.random.choice(valid_actions)

    def get_valid_actions(self, state):
        return [col for col in range(self.env.get_board_dimensions()[1]) if state[0, col] == 0]

class DQN:
    def __init__(self, input_dim, output_dim, lr, gamma):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.gamma = gamma
        self.memory = []
        self.replay_buffer = ReplayBuffer(max_size=1000)  # Experience Replay Buffer
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()
        
    def act(self, state):
        if np.random.rand() < 0.1:
            return np.random.choice(7)

        # Get the number of rows and columns from the environment (assuming env provides these)
        num_rows, num_cols = env.get_board_dimensions()  # Replace with your method to get dimensions

        # Flatten the state represents the board
        flat_state = state.flatten()

        # Calculate the index offset for each column
        col_offsets = [i * num_rows for i in range(num_cols)]

        # Get valid actions (non-full columns)
        valid_actions = [col for col in range(num_cols) if flat_state[col_offsets[col]] == 0]

        if not valid_actions:
            return -1  # No valid actions, return a special value

        action = np.random.choice(valid_actions)
        return action

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state.flatten(), action, reward, next_state.flatten(), done))

    def train(self):
        batch_size = 32  # Hyperparameter: batch_size
        if len(self.replay_buffer) < batch_size:  # Hyperparameter: batch_size
            return

        experiences = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(states)
        
        # Use target network for next state Q-values
        with torch.no_grad():  # Detach gradients for target network
            next_q_values = self.target_model(next_states)
        next_q_values[dones == 1] = 0

        q_value = q_values.gather(1, actions.unsqueeze(1).long()).squeeze(1)
        target_q_value = rewards + self.gamma * torch.max(next_q_values, dim=1).values

        loss = self.loss_fn(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network weights slowly
        self.update_target_weights()  # Hyperparameter: update frequency
class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if self.pos >= len(self.buffer):
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.max_size

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)
    

env = Connect4Env()
agent1 = DQN(42, 7, 0.001, 0.99)
agent2 = DQN(42, 7, 0.001, 0.99)
rand_agent = RandomAgent(env)
episode_rewards = []
training_losses = []
for i in range(5000):
    winner, average_episode_reward = env.play(agent1, agent2)
    agent1.train()
    agent2.train()

    # Calculate average episode reward (using data from env.play())
    episode_rewards.append(average_episode_reward)

    # Calculate loss if applicable (assuming loss is a tensor)
    loss = agent1.train()  # Assuming training returns the loss
    if loss:  # Check if loss exists (optional for loss plotting)
        training_losses.append(loss.item())

    if i % 100 == 0:
        print(f'Game {i}, winner: {winner}')

for i in range(5000):
    winner, average_episode_reward = env.play(agent1, rand_agent)
    agent1.train()

    episode_rewards.append(average_episode_reward)

    # Calculate loss if applicable (assuming loss is a tensor)
    loss = agent1.train()  # Assuming training returns the loss
    if loss:  # Check if loss exists (optional for loss plotting)
        training_losses.append(loss.item())

    if i % 100 == 0:
        print(f'Game {i}, winner: {winner}')

# Play against the trained model
while True:
    env.reset()
    env.render()

    user_player = int(input("Choose your player (1 or 2): "))
    if user_player not in [1, 2]:
        print("Invalid player choice. Please choose 1 or 2.")
        continue

    while not env.done:
        if env.player == user_player:
            action = int(input("Your move: ")) - 1
        else:
            # Agent's move
            action = agent1.act(env.board.flatten())

        _, _, env.done, _ = env.step(action)
        env.render()

        if env.done or env._is_winner():
            break

    if env._is_winner():
        print(f"Player {env.winner} wins!")
    else:
        print("It's a draw!")

    play_again = input("Do you want to play again? (yes/no): ")
    if play_again.lower() != 'yes':
        print("Game over!")
        break

# After training, plot the results (using matplotlib)
plt.figure(figsize=(10, 6))

# Plot Average Episode Reward
plt.plot(episode_rewards, label='Average Episode Reward')

# Plot Loss (optional)
if training_losses:
    plt.plot(training_losses, label='Training Loss')

plt.xlabel('Training Episodes')
plt.ylabel('Reward/Loss') 
plt.title('DQN Training Results (Connect4)')
plt.legend()
plt.grid(True)
plt.show()