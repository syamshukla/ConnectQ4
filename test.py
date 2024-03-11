import random
from matplotlib import pyplot as plt
import numpy as np
import torch

# Deep Q Network (DQN) with Experience Replay for Connect4
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
    
    def get_board_dimensions(self):
        return self.board.shape

    def step(self, action):
        if self.done:
            return self.board, 0, self.done, self.winner

        for i in range(5, -1, -1):
            if self.board[i, action] == 0:
                self.board[i, action] = self.player
                break

        intermediate_reward = 0

        # Check for three in a row for current player (after placing the piece)
        for i in range(6):
            for j in range(4):
                if (self.board[i, j] == self.player and
                        self.board[i, j + 1] == self.player and
                        self.board[i, j + 2] == self.player):
                    intermediate_reward += 0.1  # Reward for three in a row

        # Check for blocking opponent's three in a row (after placing the piece)
        for i in range(6):
            for j in range(4):
                if (self.board[i, j] == -self.player and
                        self.board[i, j + 1] == -self.player and
                        self.board[i, j + 2] == -self.player):
                    intermediate_reward += 0.1  # Reward for blocking opponent's three

        # Check for center placement (after placing the piece)
        if action in [3, 4]:
            intermediate_reward += 0.05  # Reward for placing in center column

        # Check for opponent's two in a row (after placing the piece)
        for i in range(6):
            for j in range(3):
                if (self.board[i, j] == -self.player and
                        self.board[i, j + 1] == -self.player):
                    intermediate_reward += 0.02  # Reward for blocking opponent's two

        # Update total reward with intermediate reward
        reward = intermediate_reward

        if self._is_winner():
            self.done = True
            self.winner = self.player
            reward += 1  # Final win reward (add to intermediate reward)
        elif self._is_draw():
            self.done = True
            reward += 0  # No additional reward for draw

        self.player = -1 if self.player == 1 else 1

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
        while not self.done:
            flattened_board = self.board.flatten()
            if self.player == 1:
                action = agent1.act(flattened_board)
            else:
                action = agent2.act(flattened_board)
            if self._is_valid_action(action):
                _, reward, done, _ = self.step(action)
            else:
                # Only apply penalties for invalid actions during training
                if hasattr(agent1, 'update_memory'):
                    reward = -0.1  # Penalty for invalid action
                    done = True  # End agent's turn due to invalid move
                else:
                    print(f"Invalid action: {action}. Player {self.player} loses turn.")
            self.player = -1 if self.player == 1 else 1
            if self.done:
                break  # Exit loop if game is over
            if hasattr(agent1, 'update_memory'):
                agent1.update_memory(self.board.flatten(), action, reward, self.board.flatten(), done)
                agent2.update_memory(self.board.flatten(), action, -reward, self.board.flatten(), done)
        self.episode_reward = reward
        self.number_of_steps_in_episode = 1  # Count only valid moves
        average_episode_reward = self.episode_reward  # No need for averaging
        return self.winner, average_episode_reward

    def get_board_dimensions(self):
        return self.board.shape

    def _is_valid_action(self, action):
        return 0 <= action < self.board.shape[1] and self.board[0, action] == 0  # Check valid column

class RandomAgent:
    def __init__(self, env):
        self.env = env  # Store the environment

    def act(self, state):  # Removed `env` as an argument
        # Use self.env instead of env to access the environment methods
        state = state.reshape([self.env.get_board_dimensions()[0], self.env.get_board_dimensions()[1]])  # Reshape to 2D
        valid_actions = self.get_valid_actions(state)
        if not valid_actions:
            return -1  # No valid actions available
        return np.random.choice(valid_actions)

    def get_valid_actions(self, state):
        return [col for col in range(self.env.get_board_dimensions()[1]) if state[0, col] == 0]
    def update_memory(self, state, action, reward, next_state, done):
        pass


class DQN:
    def __init__(self, input_dim, output_dim, lr, gamma, epsilon=0.9, epsilon_decay=0.995):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon  # Initialize 
        self.epsilon_decay = epsilon_decay
        self.model = torch.nn.Sequential(
        torch.nn.Linear(self.input_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, self.output_dim)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.target_model = torch.nn.Sequential(  # Identical structure for target network
        torch.nn.Linear(self.input_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, self.output_dim)
        )
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize with model weights
        self.replay_buffer = ReplayBuffer(10000)  # Experience replay buffer

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)  # Convert to PyTorch tensor
        self.epsilon *= self.epsilon_decay  # Decay epsilon for exploration vs exploitation
        self.epsilon = max(self.epsilon, 0.1)  # Ensure epsilon has a minimum value
        if np.random.rand() < self.epsilon:
            with torch.no_grad():  # No gradient calculation during exploration
                return np.random.choice(range(self.output_dim))
        q_values = self.model(state)
        return torch.argmax(q_values, dim=1).item()  # Return action with highest Q-value

    def update_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.append(state, action, reward, next_state, done)

    def train(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).long().unsqueeze(1)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1)
        next_states = torch.from_numpy(next_states).float()
        dones = torch.from_numpy(dones).bool().unsqueeze(1)

        q_values = self.model(states)  # Q-values for current states
        q_values_next = self.target_model(next_states).detach()  # Detached Q-values for next states (for stability)
        mask = ~dones  # Invert boolean mask for non-terminal states
        q_values_target = rewards + self.gamma * torch.max(q_values_next * mask, dim=1)[0].unsqueeze(1)

        # Double DQN loss calculation
        selected_q_values = torch.gather(q_values, dim=1, index=actions)  # Selected Q-values for actions
        dqn_loss = torch.nn.functional.mse_loss(selected_q_values, q_values_target)

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        # Soft update of target network (optional, for improved stability)
        self.soft_update(self.target_model, self.model, 0.1)

    def soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.store = []
        self.ptr = 0
    
    def append(self, state, action, reward, next_state, done):
        if len(self.store) == self.max_size:
            self.store[self.ptr] = (state, action, reward, next_state, done)
        else:
            self.store.append((state, action, reward, next_state, done))
        self.ptr = (self.ptr + 1) % self.max_size  # Update pointer for circular buffer

    def sample(self, batch_size):
        max_idx = len(self.store) if len(self.store) < self.max_size else self.max_size
        batch_indices = np.random.choice(max_idx, size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in batch_indices:
            states.append(self.store[i][0])
            actions.append(self.store[i][1])
            rewards.append(self.store[i][2])
            next_states.append(self.store[i][3])
            dones.append(self.store[i][4])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
env = Connect4Env()
input_dim = env.get_board_dimensions()[0] * env.get_board_dimensions()[1]  # Flatten board for input
output_dim = env.get_board_dimensions()[1]  # Number of possible actions (columns)
learning_rate = 0.001
gamma = 0.99  # Discount factor
num_episodes = 1000  # Number of training episodes
input_dim = env.get_board_dimensions()[0] * env.get_board_dimensions()[1]  # Flatten board for input
agent1 = DQN(input_dim, output_dim, learning_rate, gamma)
agent2 = RandomAgent(env)  # Initially play against a random agent
episode_rewards = []
average_episode_rewards = []

for i in range(num_episodes):
    winner, episode_reward = env.play(agent1, RandomAgent(env))
    episode_rewards.append(episode_reward)
    average_episode_rewards.append(np.mean(episode_rewards[-100:]))  # Track average of last 100 episodes

    # Train agent1 after each episode
    if len(agent1.replay_buffer.store) > 1000:  # Minimum experience for training
        agent1.train(64)  # Train with a batch size of 64

    # Print progress and plot (optional)
    if i % 100 == 0:
        print(f"Episode: {i}, Winner: {winner}, Average Reward: {average_episode_rewards[-1]:.2f}")
        # Plot learning curve using matplotlib (optional)

while True:
    env.reset()  # Reset the game board

    while not env.done:
        # Player 1's move
        env.render()
        valid_move = False
        while not valid_move:
            player_action = int(input("Enter your move (1-7): ")) - 1
            if 0 <= player_action < env.get_board_dimensions()[1] and env.board[0, player_action] == 0:
                env.step(player_action)
                valid_move = True
            else:
                print("Invalid move. Please choose an empty column.")

        # Check for game over after player's move
        if env.done:
            break

        # Agent's move
        env.render()
        agent_action = agent1.act(env.board.flatten())
        env.step(agent_action)

    # Display winner and prompt to play again
    env.render()
    print(f"Winner: {env.winner}")

    play_again = input("Play again? (y/n): ")
    if play_again.lower() != 'y':
        break

        