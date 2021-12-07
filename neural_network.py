import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
import chess_ai_helper
import constants as c
from game_state import GameState


class DeepQNetwork(nn.Module):
    """
    The neural network for a Deep Q learning agent.
    """
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        """
        Constructor for this neural network.

        :param lr: learning rate
        :param input_dims: the input parameters dimension to the neural network
        :param fc1_dims: the number of first layer nodes
        :param fc2_dims: the number of second layer nodes
        :param n_actions: the number of outputs from the neural network (possible actions)
        """
        # Call super constructor
        super(DeepQNetwork, self).__init__()

        # Store local variables
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # Define the layers of the neural network
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)  # Layer 1
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)     # Layer 2
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)    # Output layer

        # Define the optimiser, loss and device
        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Sends the inputs through the neural network to get the raw action values as outputs that can be used to find
        the best action.

        :param state: the input state
        :return: the raw actions data found from the given state
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class DRLAgent:
    """
    The class that represents the deep reinforcement learning agent for chess.
    """
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=c.STATE_SIZE, eps_end=0.01, eps_dec=1e-5):
        """
        The constructor for this deep reinforcement learning agent.

        :param gamma: the parameter that determines the decreasing value of future rewards
        :param epsilon: the parameter that determines how often the agent performs a random action
        :param lr: the learning rate
        :param input_dims: the number of inputs from the state
        :param batch_size: the number of agents in each episode being ran
        :param n_actions: the number of actions the agent can perform
        :param max_mem_size: the maximum information from previous episodes we should store
        :param eps_end: the minimum size of epsilon
        :param eps_dec: how much we should decrease epsilon by on each episode
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_counter = 0

        self.Q_eval = DeepQNetwork(self.lr, input_dims, fc1_dims=384, fc2_dims=384, n_actions=n_actions)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool8)

        # Load the already learnt information
        print("Loading data...")
        self.load()
        print("Data Loaded!")

    def store_transition(self, state, action, state_new, terminal):
        """
        Stores the information from the last action in memory so it can be learnt from.

        :param state: the state that the action was taken in
        :param action: the action taken
        :param state_new: the new state that the action took the agent to
        :param terminal: whether the new state is terminal
        """
        # Reward
        reward = chess_ai_helper.AIChessHelper.reward(state, action, state_new)

        # Convert state, action, state_new to appropriate forms
        converted_state = self.convert_state(state)
        converted_new_state = self.convert_state(state_new)
        converted_action = self.action_to_index(action)

        # Store into the appropriate memory index. If it exceeds the max memory size, reset to 0 index
        index = self.mem_counter % self.mem_size

        # Store the appropriate values in their vectors
        self.state_memory[index] = converted_state
        self.new_state_memory[index] = converted_new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = converted_action
        self.terminal_memory[index] = terminal

        # Increment the memory counter
        self.mem_counter += 1

    def choose_action(self, observation, possible_actions):
        """
        Chooses the action using epsilon greedy method.

        :param observation: the state we need to take the action in
        :param possible_actions: the possible actions for this state (in chess, not all actions are possible)
        :return: the action chosen by the agent
        """
        # If there are no actions, we're terminal. Terminal is flagged with None in chosen action
        if possible_actions is None:
            return None

        # epsilon greedy
        if np.random.random() > self.epsilon:
            # Set the action as the best possible action
            action = self.get_best_action(observation, possible_actions)
        else:
            # Otherwise, take a random action
            action = np.random.choice(possible_actions)

        return action

    def get_best_action(self, observation, possible_actions):
        """
        Finds the best action given the current information in the neural network.

        :param observation: the state we need to take the action in
        :param possible_actions: the possible actions for this state (in chess, not all actions are possible)
        :return: the best action
        """
        if possible_actions is None:
            return None
        # Convert actions into an easier format of icon, location, offset
        converted_actions = chess_ai_helper.AIChessHelper.convert_actions(possible_actions)
        converted_state = self.convert_state(observation)

        # Reformat state so the neural network can take it as input
        state = torch.tensor(np.float32(converted_state)).to(self.Q_eval.device)

        # Use the neural network to get the raw actions outputs
        actions = self.Q_eval.forward(state)

        # Basic actions are just possible actions without the promotion or piece type
        basic_actions = []
        if isinstance(converted_actions, list):
            for elem in converted_actions:
                if len(elem) == 3:
                    basic_actions.append(elem[1:])
                else:
                    basic_actions.append(elem[1:-1])
        else:
            if len(converted_actions) == 3:
                basic_actions.append(converted_actions[1:])
            else:
                basic_actions.append(converted_actions[1:-1])

        # Get indexes of sorted actions
        indexes = torch.argsort(actions, descending=True)

        for index in indexes:
            index = index.item()

            # Convert index to an action
            action = self.index_to_action(index)

            # Break loop when it's a possible action (ignoring the piece type being moved)
            if action in basic_actions:
                # Get index for possible actions so we can update the action with the appropriate piece
                i = basic_actions.index(action)
                action = possible_actions[i]
                break

        return action

    @staticmethod
    def index_to_action(index):
        """
        Converts the index of an action chosen by the agent into an actual action the chess program can use.

        :param index: the index of the action chosen by the agent
        :return: the actual action
        """
        # Find the starting square and final square (0-63) from the index. First 64 index corresponds to moving from
        # square 0 to squares 0-63. This pattern is repeated
        start_square = np.floor_divide(index, 64)
        final_square = index % 64

        # Now convert the numbers (0-63) to a tuple (x, y), start bottom left and go right
        y_start = np.floor_divide(start_square, 8) + 1
        y_final = np.floor_divide(final_square, 8) + 1
        x_start = start_square % 8 + 1
        x_final = final_square % 8 + 1

        # Calculate offset
        offset = x_final - x_start, y_final - y_start

        return (x_start, y_start), offset

    @staticmethod
    def action_to_index(action):
        """
        Converts an action from the form the chess program can read to its index in the neural network.

        :param action: the action in the form readable by the chess program
        :return: the index of the given action in the neural network
        """
        # The only part of action that matters is location and offset, so remove the unnecessary parts
        if len(action) == 4:
            action = action[1].location, action[2:-1]
        else:
            action = action[0].location, action[1]

        # Starting square and final square
        start_square = action[0][0] - 1, action[0][1] - 1
        final_square = action[0][0] + action[1][0] - 1, action[0][1] + action[1][1] - 1

        # Convert start and final square to index
        index = (start_square[1] * 8 + start_square[0]) * 64 + (final_square[1] * 8 + final_square[0])

        if 0 > index or index > 4095:
            print("Error in calculating index of", index, "for action", action)
        return index

    @staticmethod
    def convert_state(state: GameState):
        """
        Converts the state from a chess board state into a form readable by the neural network.

        :param state: the state of the chess board
        :return: a readable state input for the neural network
        """
        # Board representation. Val given by piece value on that square (negative for enemy). Example list below:
        # [val at (1, 1), val at (2, 1), ..., val at (8, 8)]
        board_repr = []
        for y in range(1, 9):
            for x in range(1, 9):
                piece = state.board_state.get((x, y), None)
                if piece is None:
                    # 0 for no piece
                    board_repr.append(0)
                else:
                    # Piece value if there's a piece here
                    direction = 1 if piece.colour == state.turn else -1
                    board_repr.append(direction * piece.value)

        return board_repr

    def learn(self):
        """
        Learns from whatever actions were just taken.

        """
        # Don't learn until the memory has filled up sufficiently - no point in learning from zeros
        if self.mem_counter < self.batch_size:
            return

        # Zero the gradient so it doesn't continue summing the previous gradient on this iteration
        self.Q_eval.optim.zero_grad()

        # Find up to what index we've filled the memory to
        max_mem = min(self.mem_counter, self.mem_size)

        # Select random memories within the above index without replacement
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Choose the appropriate elements from the batch and convert to tensors where necessary
        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        # Get the q evaluations for the state and next state
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        # Get the target q values with the q update formula
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        # Get the loss function and optimize
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optim.step()

        # Update value of epsilon
        self.epsilon -= self.eps_dec if self.epsilon > self.eps_min else 0

    def save(self):
        """
        Saves the current neural network into a JSON file.

        """
        # Print to screen when saving
        print("Saving...")

        # Name of file
        name = c.SAVE_NAME + ".json"
        name_model = c.SAVE_NAME + "_model.pt"

        # Save model
        torch.save(self.Q_eval, name_model)

        # Save agent
        # File we're saving to
        file = open(name, 'w')

        # Put data into list
        data = [self.state_memory, self.new_state_memory, self.reward_memory,
                self.action_memory, self.terminal_memory, self.mem_counter]

        # Convert data from np array to normal list so it can be saved
        converted_data = [elem.tolist() for elem in data[:-1]]
        converted_data.append(data[-1])

        # Dump data into file and close it
        json.dump(converted_data, file)
        file.close()

        # Let user know we're done
        print("Saved")

    def load(self):
        """
        Loads the neural network data if it exists.

        """
        # Name of file
        name = c.SAVE_NAME + ".json"
        name_model = c.SAVE_NAME + "_model.pt"

        # Check if file exists
        path = Path(name)
        if path.exists():
            # Extract model information
            self.Q_eval = torch.load(name_model)

            # Extract agent for information
            # Open file
            file = open(name, 'r')

            # Load data
            data = json.load(file)

            # Convert each data element to the correct type
            data[0] = np.array(data[0], dtype='float32')
            data[1] = np.array(data[1], dtype='float32')
            data[2] = np.array(data[2], dtype='int32')
            data[3] = np.array(data[3], dtype='float32')
            data[4] = np.array(data[4], dtype='bool8')

            # Load information into the agent
            self.state_memory, self.new_state_memory, self.reward_memory,\
            self.action_memory, self.terminal_memory, self.mem_counter = data

            # Close the file
            file.close()
