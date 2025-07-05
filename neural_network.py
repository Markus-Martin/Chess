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
import time
import matplotlib.pyplot as plt


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


class DRLMainAction:
    """
    The class that represents the deep reinforcement learning agent for chess. It decides the main action to take with
    the DRLPieceChooser helping when duplicate actions appear (i.e. two or more pieces of the same type can perform the
    given move). In particular the actions are chosen as:
    Index                     Actions
    000-055        Queen offset moves (Length is 8x7 = 56)
    056-063        Knight offset moves (Length is 8)
    064-091        Rook offset moves (Length is 4x7 = 28)
    092-119        Bishop offset moves (Length is 4x7 = 28)
    120-127        King offset moves (Length is 8)
    128-140        Pawn offset moves (Length is 1 (2 squares foward) + 4 (promotions) x 3 (possible moves when promoting)  = 13)
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

        # Track the loss throughout training for visualisation
        self.loss_history = []

        # Set epsilon (random action chance) to 0 if the AI is in play mode (as opposed to learn mode)
        if c.AI_MODE == 2:
            self.epsilon = 0

        self.Q_eval = DeepQNetwork(self.lr, input_dims, fc1_dims=384, fc2_dims=384, n_actions=n_actions)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

        # Create the agent that chooses the piece when duplicate options appear. 64 possible location of pieces
        print("Loading data...")
        self.piece_chooser = DRLPieceChooser(gamma, self.epsilon, lr, input_dims, batch_size, 64, self.mem_size, eps_end, eps_dec)

        # Load the already learnt information
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

        # Call store transition for piece chooser
        self.piece_chooser.store_transition(state, action, state_new, terminal)

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
            action_index = np.random.choice(range(len(possible_actions)))
            action = possible_actions[action_index]

        return action

    def get_best_action(self, observation, possible_actions):
        """
        Finds the best action given the current information in the neural network.

        :param observation: the state we need to take the action in
        :param possible_actions: the possible actions for this state (in chess, not all actions are possible)
        :return: the best action
        """
        action_chosen = None

        if possible_actions is None:
            return None

        # Convert actions and state into an easier format of icon, location, offset
        converted_actions = chess_ai_helper.AIChessHelper.convert_actions(possible_actions)
        converted_state = self.convert_state(observation)

        # Reformat state so the neural network can take it as input
        state = torch.tensor(np.float32(converted_state)).to(self.Q_eval.device)

        # Use the neural network to get the raw actions outputs
        actions = self.Q_eval.forward(state)

        # Basic actions are just possible actions without the location
        # Icon, Offset
        basic_actions = []
        if isinstance(converted_actions, list):
            for elem in converted_actions:
                basic_actions.append((elem[0], elem[-1]))
        else:
            basic_actions.append((converted_actions[0], converted_actions[-1]))

        # Get indexes of sorted actions
        indexes = torch.argsort(actions, descending=True)
        for index in indexes:
            index = index.item()

            # Convert index to an action
            action = self.index_to_action(index)

            # Adjust pawn direction. Pawn promotion will be dealt with later
            if action[0] == c.PAWN:
                # Invert y direction of pawn if it's black's turn because everything is written in terms of white
                if not observation.turn:
                    if len(action[1]) == 3:
                        action = action[0], (action[1][0], -action[1][1], action[1][2])
                    else:
                        action = action[0], (action[1][0], -action[1][1])

            # Break loop when it's a possible action (ignoring the piece type being moved)
            if action in basic_actions or (action[0] == c.PAWN and (action[0], action[1][:-1]) in basic_actions):
                # Get every index of duplicate actions and store the actions in a list
                start_at = -1
                duplicate_actions = []
                while True:
                    try:
                        i = basic_actions.index(action, start_at + 1)
                    except ValueError:
                        break
                    else:
                        duplicate_actions.append(possible_actions[i])
                        start_at = i

                # If it's a pawn, we must also check without promotions
                if action[0] == c.PAWN:
                    start_at = -1
                    while True:
                        try:
                            i = basic_actions.index((action[0], action[1][:-1]), start_at + 1)
                        except ValueError:
                            break
                        else:
                            duplicate_actions.append(possible_actions[i])
                            start_at = i

                # If there are duplicates, send the board information to the piece chooser DRL agent
                if len(duplicate_actions) == 1:
                    action_chosen = duplicate_actions[0]
                    break
                else:
                    location = self.piece_chooser.choose_location(observation, duplicate_actions)
                    for duplicate_action in duplicate_actions:
                        if duplicate_action[0].location == location:
                            action_chosen = duplicate_action
                            break
                    break
        if action_chosen is None:
            print("Error in choosing action from possible options:", possible_actions)
            return possible_actions[np.random.choice(range(len(possible_actions)))]
        return action_chosen

    @staticmethod
    def index_to_action(index):
        """
        Converts the index of an action chosen by the agent into an actual action the chess program can use. Full list
        given below:

        Index                     Actions
        000-055        Queen offset moves (Length is 8x7 = 56)
        056-063        Knight offset moves (Length is 8)
        064-091        Rook offset moves (Length is 4x7 = 28)
        092-119        Bishop offset moves (Length is 4x7 = 28)
        120-129        King offset moves (Length is 8 + 2 (castling))
        130-142        Pawn offset moves (Length is 1 (2 squares foward) + 4 (promotions) x 3 (possible moves when promoting)  = 13)

        :param index: the index of the action chosen by the agent
        :return: the actual action in the form: icon, offset
        """
        if index < 0:
            # Error. Index out of bounds.
            raise IndexError("Neural Network index out of bounds")
        elif index < 56:
            # Moving a queen
            icon = c.QUEEN

            # Begin in up direction then rotate clockwise (index assigned like a clock)
            directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

            # Index of the directions should increase by 1 every 8 indices
            dir_index = np.floor_divide(index, 7)

            # How far to go in a particular direction
            strength = index % 7 + 1

            # Offset found by multiplying the direction by the strength (element wise)
            offset = directions[dir_index][0] * strength, directions[dir_index][1] * strength
        elif index < 64:
            # Moving a knight
            icon = c.KNIGHT

            # Adjust index so it starts at 0 again for the knight
            index -= 56

            # For offset, we can just direction substitute the index into the known knight actions
            offset = c.KNIGHT_OFFSETS[index]
        elif index < 92:
            # Moving a rook
            icon = c.ROOK

            # Adjust index so it starts at 0 again for the rook
            index -= 64

            # Begin in up direction then rotate clockwise (index assigned like a clock)
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

            # Index of the directions should increase by 1 every 8 indices
            dir_index = np.floor_divide(index, 7)

            # How far to go in a particular direction
            strength = index % 7 + 1

            # Offset found by multiplying the direction by the strength (element wise)
            offset = directions[dir_index][0] * strength, directions[dir_index][1] * strength
        elif index < 120:
            # Moving a bishop
            icon = c.BISHOP

            # Adjust index so it starts at 0 again for the bishop
            index -= 92

            # Begin in up direction then rotate clockwise (index assigned like a clock)
            directions = [(1, 1), (1, -1), (-1, -1), (-1, 1)]

            # Index of the directions should increase by 1 every 8 indices
            dir_index = np.floor_divide(index, 7)

            # How far to go in a particular direction
            strength = index % 7 + 1

            # Offset found by multiplying the direction by the strength (element wise)
            offset = directions[dir_index][0] * strength, directions[dir_index][1] * strength
        elif index < 130:
            # Moving a king
            icon = c.KING

            # Adjust index so it starts at 0 again for the king
            index -= 120

            # Possible king directions is same as queen but the strength is always 1
            directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (2, 0), (-2, 0)]

            # Offset can be directly accessed from directions similar to with the knight
            offset = directions[index]
        elif index < 143:
            # Moving a pawn
            icon = c.PAWN

            # Adjust index so it starts at 0 again for the pawn
            index -= 130

            # First action is assigned to an offset of (0, 2) or (0, -2). For simplicity, assume positive
            if index == 0:
                offset = (0, 2)
            else:
                index -= 1

                # For all other indices, the direction is in clockwise order
                directions = [(0, 1), (1, 1), (-1, 1)]

                # Each promotion type
                promotion_list = ["Q", "N", "R", "B"]

                # Index of the directions should increase by 1 every 4 indices (each promotion type)
                dir_index = np.floor_divide(index, 4)

                # Promotion chosen
                promotion = promotion_list[index % 4]

                # Combine everything into an offset. Promotion should be dealt with later at the same time as direction.
                # This is because this function has no access to piece location nor colour so it cannot be dealt with
                offset = *directions[dir_index], promotion
        else:
            # Error. Index out of bounds.
            raise IndexError("Neural Network index out of bounds")

        return icon, offset

        # # Find the starting square and final square (0-63) from the index. First 64 index corresponds to moving from
        # # square 0 to squares 0-63. This pattern is repeated
        # start_square = np.floor_divide(index, 64)
        # final_square = index % 64
        #
        # # Now convert the numbers (0-63) to a tuple (x, y), start bottom left and go right
        # y_start = np.floor_divide(start_square, 8) + 1
        # y_final = np.floor_divide(final_square, 8) + 1
        # x_start = start_square % 8 + 1
        # x_final = final_square % 8 + 1
        #
        # # Calculate offset
        # offset = x_final - x_start, y_final - y_start
        #
        # return (x_start, y_start), offset

    @staticmethod
    def action_to_index(action):
        """
        Converts an action from the form the chess program can read to its index in the neural network.

        :param action: the action in the form readable by the chess program
        :return: the index of the given action in the neural network
        """
        # The only part of action that matters is icon, offset and promotion (if it's there)
        if not isinstance(action[0], str):
            action = action[0].icon, action[1]

        # Initialise index
        index = 0

        # Cycle through each possibility for icon
        if action[0] == "Q":
            # Begin in up direction then rotate clockwise (index assigned like a clock)
            directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

            # Index of the directions from the given offset
            dir_index = directions.index(tuple(map(np.sign, action[1])))

            # How far to go in a particular direction
            strength = max(map(abs, action[1]))

            # Add to the index the appropriate amount
            index += dir_index * 7 + (strength - 1)
        elif action[0] == "N":
            # Adjust index due to the type icon
            index += 56

            # Index of the possible knight moves
            dir_index = c.KNIGHT_OFFSETS.index(action[1])

            # Adjust index based on which offset was taken
            index += dir_index
        elif action[0] == "R":
            # Adjust index due to the type icon
            index += 64

            # Begin in up direction then rotate clockwise (index assigned like a clock)
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

            # Index of the directions from the given offset
            dir_index = directions.index(tuple(map(np.sign, action[1])))

            # How far to go in a particular direction
            strength = max(map(abs, action[1]))

            # Add to the index the appropriate amount
            index += dir_index * 7 + (strength - 1)
        elif action[0] == "B":
            # Adjust index due to the type icon
            index += 92

            # Begin in up direction then rotate clockwise (index assigned like a clock)
            directions = [(1, 1), (1, -1), (-1, -1), (-1, 1)]

            # Index of the directions from the given offset
            dir_index = directions.index(tuple(map(np.sign, action[1])))

            # How far to go in a particular direction
            strength = max(map(abs, action[1]))

            # Add to the index the appropriate amount
            index += dir_index * 7 + (strength - 1)
        elif action[0] == "K":
            # Adjust index due to the type icon
            index += 120

            # Possible king movements
            directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (2, 0), (-2, 0)]

            # Index of the possible knight moves
            dir_index = directions.index(action[1])

            # Adjust index based on which offset was taken
            index += dir_index
        else:
            # Final option is moving a pawn, so we first adjust the index for this type
            index += 130

            # Adjust the y direction to always be positive when converting to index
            if len(action[1]) == 3:
                action = action[0], (action[1][0], abs(action[1][1]), action[1][2])
            else:
                action = action[0], (action[1][0], abs(action[1][1]), "Q")

            # We don't need to add to index any further if the action is a 2 offset so consider the other case
            if action[1][:-1] != (0, 2):
                # Possible pawn movements
                directions = [(0, 1), (1, 1), (-1, 1)]

                # Possible pawn promotions
                promotion_list = ["Q", "N", "R", "B"]

                # Index of the directions and promotion
                dir_index = directions.index((action[1][:-1]))
                prom_index = promotion_list.index(action[1][2])

                index += dir_index * 4 + prom_index + 1

        return index

        # # The only part of action that matters is icon, offset and promotion (if it's there)
        # if len(action[1]) == 3:
        #     action = action[0].icon, (action[1][0], action[1][1])
        # else:
        #     action = action[0].location, action[1]
        #
        # # Starting square and final square
        # start_square = action[0][0] - 1, action[0][1] - 1
        # final_square = action[0][0] + action[1][0] - 1, action[0][1] + action[1][1] - 1
        #
        # # Convert start and final square to index
        # index = (start_square[1] * 8 + start_square[0]) * 64 + (final_square[1] * 8 + final_square[0])
        #
        # if 0 > index or index > 4095:
        #     print("Error in calculating index of", index, "for action", action)
        #
        # return index

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

        # Store the loss so training progress can be analysed
        self.loss_history.append(loss.item())

        # Update value of epsilon
        self.epsilon -= self.eps_dec if self.epsilon > self.eps_min else 0

        # Make the piece chooser learn as well
        self.piece_chooser.learn()

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

        # Save agent only if the options say to
        if c.SAVE_FOR_TRAINING:
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

        # Save the piece chooser too
        self.piece_chooser.save()

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

            # Extract agent for information only if the options say to
            if c.SAVE_FOR_TRAINING:
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

    def plot_loss(self, filename="training_loss.png"):
        """Saves a plot of the training loss."""
        if len(self.loss_history) == 0:
            return
        plt.figure()
        plt.plot(self.loss_history)
        plt.title("Training Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.savefig(filename)
        plt.close()


class DRLPieceChooser:
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

        # Track the loss throughout training for visualisation
        self.loss_history = []

        self.Q_eval = DeepQNetwork(self.lr, input_dims, fc1_dims=64, fc2_dims=64, n_actions=n_actions)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.location_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

        # Load the already learnt information
        self.load()

    def store_transition(self, state, action, state_new, terminal):
        """
        Stores the information from the last action in memory so it can be learnt from.

        :param state: the state that the action was taken in
        :param action: the action taken
        :param state_new: the new state that the action took the agent to
        :param terminal: whether the new state is terminal
        """
        location = action[0].location

        # Reward
        reward = chess_ai_helper.AIChessHelper.reward(state, action, state_new)

        # Convert state, action, state_new to appropriate forms
        converted_state = self.convert_state(state)
        converted_new_state = self.convert_state(state_new)
        converted_location = self.location_to_index(location)

        # Store into the appropriate memory index. If it exceeds the max memory size, reset to 0 index
        index = self.mem_counter % self.mem_size

        # Store the appropriate values in their vectors
        self.state_memory[index] = converted_state
        self.new_state_memory[index] = converted_new_state
        self.reward_memory[index] = reward
        self.location_memory[index] = converted_location
        self.terminal_memory[index] = terminal

        # Increment the memory counter
        self.mem_counter += 1

    def choose_location(self, observation, possible_actions):
        """
        Chooses the piece using epsilon greedy method.

        :param observation: the state we need to take the action in
        :param possible_actions: the possible actions for this state (piece, offset)
        :return: the action chosen by the agent
        """
        # If there are no actions, we're terminal. Terminal is flagged with None in chosen action
        if possible_actions is None:
            return None

        # epsilon greedy
        if np.random.random() > self.epsilon:
            # Set the piece as the best possible piece
            location = self.get_best_location(observation, possible_actions)
        else:
            # Otherwise, take a random action
            piece_index = np.random.choice(range(len(possible_actions)))
            location = possible_actions[piece_index][0].location

        return location

    def get_best_location(self, observation, possible_actions):
        """
        Finds the best piece given the current information in the neural network.

        :param observation: the state we need to take the action in
        :param possible_actions: the possible actions for this state (in chess, not all actions are possible)
        :return: the best action
        """
        # Initialise the best location
        loc = None

        # Ensure there are actually possible actions
        if possible_actions is None:
            print("Possible actions is None so we cannot get best location")
            return None

        # Convert actions and state into an easier format of icon, location, offset
        converted_state = self.convert_state(observation)

        # Reformat state so the neural network can take it as input
        state = torch.tensor(np.float32(converted_state)).to(self.Q_eval.device)

        # Use the neural network to get the raw actions outputs
        locations = self.Q_eval.forward(state)

        # Convert the pieces to a location
        possible_locations = []
        if isinstance(possible_actions, list):
            for elem in possible_actions:
                possible_locations.append(elem[0].location)
        else:
            possible_locations.append(possible_actions[0].location)

        # Get indexes of sorted actions
        indexes = torch.argsort(locations, descending=True)

        for index in indexes:
            index = index.item()

            # Convert index to an action
            loc = self.index_to_location(index)

            # Break loop when it's a possible action (ignoring the piece type being moved)
            if loc in possible_locations:
                break

        if loc is None:
            print("Error in finding best location with possible actions:", possible_actions)

        return loc

    @staticmethod
    def index_to_location(index):
        """
        Converts the index of a location chosen by the agent into an actual location the chess program can use.

        :param index: the index of the location chosen by the agent
        :return: the actual location
        """
        # Index labelled left to right, bottom to top:
        # .......................
        # 8  9  10 11 12 13 14 15
        # 0  1  2  3  4  5  6  7
        y_start = np.floor_divide(index, 8) + 1
        x_start = index % 8 + 1

        return x_start, y_start

    @staticmethod
    def location_to_index(location):
        """
        Converts an location from the form the chess program can read to its index in the neural network.

        :param location: the location in the form readable by the chess program
        :return: the index of the given location in the neural network
        """
        # Starting square and final square
        start_square = location[0] - 1, location[1] - 1

        # Convert start and final square to index
        index = start_square[1] * 8 + start_square[0]

        if 0 > index or index > 63:
            print("Error in calculating index of", index, "for location", location)

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
        action_batch = self.location_memory[batch]

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

        # Store the loss so training progress can be analysed
        self.loss_history.append(loss.item())

        # Update value of epsilon
        self.epsilon -= self.eps_dec if self.epsilon > self.eps_min else 0

    def save(self):
        """
        Saves the current neural network into a JSON file.

        """
        # Name of file
        name = c.SAVE_NAME + "_2.json"
        name_model = c.SAVE_NAME + "_model_2.pt"

        # Save model
        torch.save(self.Q_eval, name_model)

        # Save agent only if the options say to
        if c.SAVE_FOR_TRAINING:
            # File we're saving to
            file = open(name, 'w')

            # Put data into list
            data = [self.state_memory, self.new_state_memory, self.reward_memory,
                    self.location_memory, self.terminal_memory, self.mem_counter]

            # Convert data from np array to normal list so it can be saved
            converted_data = [elem.tolist() for elem in data[:-1]]
            converted_data.append(data[-1])

            # Dump data into file and close it
            json.dump(converted_data, file)
            file.close()

    def load(self):
        """
        Loads the neural network data if it exists.

        """
        # Name of file
        name = c.SAVE_NAME + "_2.json"
        name_model = c.SAVE_NAME + "_model_2.pt"

        # Check if file exists
        path = Path(name)
        if path.exists():
            # Extract model information
            self.Q_eval = torch.load(name_model)

            # Extract agent for information only if the options say to
            if c.SAVE_FOR_TRAINING:
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
                self.state_memory, self.new_state_memory, self.reward_memory, \
                self.location_memory, self.terminal_memory, self.mem_counter = data

                # Close the file
                file.close()

    def plot_loss(self, filename="training_loss_piece_chooser.png"):
        """Saves a plot of the training loss for the piece chooser."""
        if len(self.loss_history) == 0:
            return
        plt.figure()
        plt.plot(self.loss_history)
        plt.title("Piece Chooser Training Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.savefig(filename)
        plt.close()
