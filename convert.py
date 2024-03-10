from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from termcolor import colored
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
    def __init__(self, dqn, target_dqn, replay_memory, gamma=0.99, epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.1, batch_size=32):
        self.dqn = dqn
        self.target_dqn = target_dqn
        self.replay_memory = replay_memory
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.batch_size = batch_size

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(7)
        else:
            q_values = self.dqn(torch.FloatTensor(state).view(1, -1))
            return torch.argmax(q_values).item()
    
    # Call this method at the end of each episode
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
    
    def update_target_network(self):
        """Copies the weights from the policy network to the target network."""
        self.target_dqn.load_state_dict(self.dqn.state_dict())



class MCTSAgent:
    def __init__(self, env):
        self.env = env
        self.C = 1  # Exploration parameter for UCT

    class Node:
        def __init__(self, parent, action, state):
            self.parent = parent
            self.action = action
            self.state = state
            self.wins = 0
            self.visits = 0
            self.children = []

    def select_action(self, state):
        root = self.create_root_node(state)
        for _ in range(self.num_simulations):
            self.simulate(root)
        return self.get_most_visited_action(root)

    def create_root_node(self, state):
        return self.Node(None, None, state)

    def simulate(self, node):
        state = node.state.copy()
        done = False
        while not done:
            # Select action based on UCT
            action = self.select_uct_action(node)
            next_state, reward, done = self.env.step(action)
            # Create new node if not visited before
            if not self.is_fully_expanded(node, action):
                self.expand_node(node, action, next_state)
            # Update visit counts and propagate reward
            self.backpropagate(node, reward)

    def select_uct_action(self, node):
        best_uct = float('-inf')
        best_action = None
        for child in node.children:
            if child.visits > 0:
                uct = child.wins / child.visits + self.C * np.sqrt(np.log(node.visits) / child.visits)
                if uct > best_uct:
                    best_uct = uct
                    best_action = child.action
        return best_action

    def expand_node(self, node, action, state):
        new_node = self.Node(node, action, state)
        node.children.append(new_node)

    def backpropagate(self, node, reward):
        while node is not None:
            node.wins += reward
            node.visits += 1
            node = node.parent

    def get_most_visited_action(self, node):
        visit_counts = [child.visits for child in node.children]
        return np.argmax(visit_counts)

    def is_fully_expanded(self, node, action):
        for child in node.children:
            if child.action == action:
                return True
        return False

    # Adjust this parameter for number of simulations per training episode
    num_simulations = 100


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
    b = np.where(a == 1, -1, a)  # Swap 1 with -1
    b = np.where(a == -1, 1, b)  # Swap -1 with 1
    return b


# display game board and let player choose their move
def update_board():
    print("Current Game Board:")
    for row in env.board:
        print("|", end=" ")
        for cell in row:
            if cell == 1:
                print(colored("X", "red"), end=" ")
            elif cell == -1:
                print(colored("O", "yellow"), end=" ")
            else:
                print(" ", end=" ")
        print("|")
    print("| " + " ".join([str(i + 1) for i in range(env.cols)]) + " |")
    print()
# single game episode
def play_one(env, agent, opponent, eps, gamma, copy_period, learn=True):
        env.reset()

        observation = env.board.copy().flatten()
        prev_observation = observation.copy()

        done = False
        total_reward = 0

        while not done:
            if learn:
                action = agent.select_action(observation)
            else:
                if opponent == "self":
                    action = agent.select_action(observation)
                elif opponent == "random":
                    action = np.random.randint(7)
                else:
                    raise ValueError("Invalid opponent type")

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
                if opponent == "self":
                    action = agent.select_action(rev(observation))
                elif opponent == "random":
                    action = np.random.randint(7)
                else:
                    raise ValueError("Invalid opponent type")

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



# ----------------------------------------------------------------------
# Console based game and DQN initialization
# ----------------------------------------------------------------------
    
env = Connect4Env()
dqn = DQN()
target_dqn = DQN()
replay_memory = ReplayMemory(capacity=100000)
agent = DQNAgent(dqn, target_dqn, replay_memory)

gamma = 0.92
eps = 0.2
copy_period = 50
N = 500
total_rewards = np.empty(N)
avg_rewards = []

opponent = "random"
switch = 200

mcts_agent = MCTSAgent(env)  # Move initialization outside the loop

for n in range(N):
    total_reward = play_one(env, agent, opponent, eps, gamma, copy_period)
    total_rewards[n] = total_reward

    if n % copy_period == 0:
        # Update epsilon
        agent.update_epsilon()
        # Update target network
        agent.update_target_network()
        
        avg_reward = total_rewards[max(0, n - 50):(n + 1)].mean()
        avg_rewards.append(avg_reward)
        print("Episode:", n, "Total Reward:", total_reward, "Average Reward (last 50):", avg_reward)
    
        # Switch opponent to MCTS agent after a certain number of episodes
    if n == switch:
        opponent = "mcts_agent"
        print("Switching opponent to MCTS agent")


# show the total wins and losses:
print("Total wins:", (total_rewards == 1).sum())
print("Total losses:", (total_rewards == -1).sum())
print("Total draws:", (total_rewards == 0).sum())


def plot_rewards(total_rewards, avg_rewards):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(total_rewards)
    plt.title('Total Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(1, 2, 2)
    plt.plot(avg_rewards)
    plt.title('Average Rewards (last 10 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    plt.show()
    
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
plot_rewards(total_rewards, avg_rewards)