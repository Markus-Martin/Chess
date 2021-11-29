from pathlib import Path
import random
import math


class QLearn:
    """
    Creates a Q learning agent that either accesses an existing file to update Q values or creates a new one. Note that
    the provided helper_class must implement the following methods:

    text_to_state(string) - Converts the given string into a state that
    state_to_text(state) - Converts the given state into a string that can be stored in a text file
    text_to_action(string) - Converts the given string into an action that
    action_to_text(action) - Converts the given action into a string that can be stored in a text file
    convert_state(GameState) - Converts the given game state into the preferred state form (identity for exact states)
    convert_actions(GameState) - Converts the game actions into the preferred action

    To use this class, create the above helper class and increment ucb values. Also, call Q_update.

    """
    def __init__(self, file_name: str, helper_class):
        # Choose the exploration strategy: 0 - epsilon greedy, 1 - UCB
        self.exploration_strategy = 1
        self.sarsa_mode = False

        # Set parameters for each method
        if self.exploration_strategy == 0:
            self.epsilon = 0.2
        elif self.exploration_strategy == 1:
            self.c = 1.4
            self.num_moves = {}
            self.num_actions = {}

        # Gamma and learning rate alpha
        self.gamma = 0.99
        self.alpha = 0.2

        # Initialise Q table
        self.q_table = {}
        self.file_name = file_name
        self.helper_class = helper_class

        # If a file already exists, extract the information
        self.extract_q_table()
        print("Q Table Extracted!")

        self.next_action = None

    def extract_q_table(self):
        """
        Extracts the q table from the file_name stored in this class if it exists. Then stores it in self.q_table.

        """
        # Add extension for file name
        name = self.file_name + ".txt"

        # Check whether file exists
        path = Path(name)
        if path.exists():
            # If the path exists then access the file and read its contents to the Q table
            read_file = open(name, "r")
            for row in read_file:
                # Ignore empty lines
                if row == "\n":
                    continue

                state_txt, action_txt, q_value = row.split("\t")
                state = self.helper_class.text_to_state(state_txt)
                action = self.helper_class.text_to_action(action_txt)

                self.q_table.update({(state, action): float(q_value)})

            # Close the file because we will edit it later
            read_file.close()

            # Also, if we're using UCB, this needs to be extracted
            if self.exploration_strategy == 1:
                name = self.file_name + "_UCB.txt"

                path = Path(name)
                if path.exists():
                    read_file = open(name, "r")

                    for row in read_file:
                        # Ignore empty lines
                        if row == "\n":
                            continue

                        # Extract numbers
                        state_ucb_txt, action_ucb_txt = row.split("\t")

                        # Access all except the last number (which holds the num_moves, num_actions)
                        state_ucb_split = state_ucb_txt.split(",")
                        state_txt = ",".join(state_ucb_split[0:-1])
                        num_moves = int(state_ucb_split[-1])
                        action_ucb_split = action_ucb_txt.split(",")
                        action_txt = ",".join(action_ucb_split[0:-1])
                        num_actions = int(action_ucb_split[-1])

                        # Get state, action from helper class
                        state = self.helper_class.text_to_state(state_txt)
                        action = self.helper_class.text_to_action(action_txt)

                        # Update num moves and num actions
                        self.num_moves.update({state: num_moves})
                        self.num_actions.update({(state, action): num_actions})

                    # Close the reader
                    read_file.close()

    def save_q_table(self):
        """
        Saves the q table in the file_name stored in this class. To do so, the file is deleted initially so this
        process should not be stopped halfway through otherwise data loss could occur.

        """
        print("Saving...")
        # Add extension for file name
        name = self.file_name + ".txt"

        # Open file
        write_file = open(name, "w")

        # Iterate through each state, action pair
        for state, action in self.q_table.keys():
            # Convert each component to text
            state_text = self.helper_class.state_to_text(state)
            action_text = self.helper_class.action_to_text(action)
            q_value = str(self.q_table[(state, action)])

            # Combine the parts into one string
            full_string = state_text + "\t" + action_text + "\t" + q_value + "\n"
            write_file.write(full_string)

        # Close q table writer
        write_file.close()

        # Also, if we're using UCB, this needs to be saved
        if self.exploration_strategy == 1:
            name = self.file_name + "_UCB.txt"
            write_file = open(name, "w")

            # Iterate through each state action pair
            for state, action in self.num_actions.keys():
                state_text = self.helper_class.state_to_text(state)
                action_text = self.helper_class.action_to_text(action)

                # Write num_moves for given state
                write_file.write("\n" + state_text + "," + str(self.num_moves.get(state, 0)))

                # Write the num actions to the same row
                write_file.write("\t" + action_text + "," + str(self.num_actions.get((state, action), 0)))

            # Close writer
            write_file.close()

        print("Saved!")

    def get_best_action(self, game_state, actions):
        """
        Finds the best action for a given state from the current estimate of the Q table.

        :param game_state: the state for which we're finding the best action in
        :param actions: a list of all possible actions in this state
        :return: the best action according to the Q table
        """
        # Transform the state into the q-table readable format
        state = self.helper_class.convert_state(game_state)

        # Get actions
        actions_q = self.helper_class.convert_actions(actions)

        # Initialise q value as -inf
        q_val = float('-inf')
        best_action = None

        if isinstance(actions, list):
            # In the case of many options
            # Cycle through each action and check Q values. Extract the one with the best Q value
            i = -1
            for action in actions_q:
                i += 1
                # Default q value for unexplored state action pair is 0
                new_q = self.q_table.get((state, action), 0)

                if new_q > q_val:
                    # Extract new q value and action. Action taken from the game action list rather than q table version
                    q_val = new_q
                    best_action = actions[i]
        else:
            # In the case of one option
            best_action = actions

        return best_action

    def eps_greedy(self, game_state, actions):
        """
        Calculates the action to take while being epsilon greedy.

        :param game_state: the current game state
        :param actions: the available actions for this state
        :return: action for this method given the current Q table estimate
        """
        # Check if random number falls within epsilon, if it does, let the action be random
        if random.random() < self.epsilon:
            return random.choice(actions)

        # Otherwise, we pick the best action
        return self.get_best_action(game_state, actions)

    def ucb(self, game_state, actions):
        """
        Chooses an action using the upper confidence bound method.

        :param game_state: state we're taking an action in
        :param actions: the list of actions in this state
        :return: action given the UCB method and current Q table estimates
        """
        # argmax(v(s, a) + C sqrt(log(N) / n)

        # Initialise UCB value to -inf so any action will beat it
        ucb_val = float('-inf')

        zero_ucb = []
        zero_ucb_exists = False

        # Get the actions and state as q table version
        actions_q = self.helper_class.convert_actions(actions)
        state = self.helper_class.convert_state(game_state)

        if isinstance(actions, list):
            # Loop through each action
            i = -1
            for action in actions_q:
                i += 1
                # Calculate the new ucb value if the action has never been taken, the default value should be 0
                if self.num_actions.get((state, action), 0) <= 0:
                    # Store in list, as long as this list isn't empty, the next action will come from here
                    zero_ucb.append(actions[i])
                    zero_ucb_exists = True
                elif not zero_ucb_exists:
                    # usb_val   =            Q(s, a)            +     C  *   sqrt   (          ln(N(s))          / n(s, a))
                    ucb_val_new = self.q_table.get((state, action), 0) + self.c * math.sqrt(math.log(self.num_moves[state])
                                                                                     / self.num_actions[(state, action)])
                    # Compare to the old ucb value and keep the bigger one and it's action
                    if ucb_val_new > ucb_val:
                        ucb_val = ucb_val_new
                        best_action = actions[i]
        else:
            # In the case of one option
            return actions

        # In case of ucb values of zero, we must pick a random action
        if zero_ucb_exists:
            return random.choice(zero_ucb)
        # Otherwise return the best action
        return best_action

    def select_action_training(self, game_state, actions):
        """
        Selects the action during training mode.

        :param game_state: the state we're performing the action in
        :param actions: the list of actions
        :return: the most appropriate action
        """
        # If SARSA, we need to check if next_action is already chosen
        if self.sarsa_mode and self.next_action is not None:
            return self.next_action

        # Pick the algorithm
        if self.exploration_strategy == 0:
            # Epsilon-greedy
            self.next_action = self.eps_greedy(game_state, actions)
            return self.next_action
        elif self.exploration_strategy == 1:
            # UCB
            self.next_action = self.ucb(game_state, actions)
            return self.next_action

    def q_update(self, prev_game_state, action, game_state, actions):
        """
        Updates the q table with the new information obtained from the move.

        :param prev_game_state: the previous game state
        :param action: the action we performed to get to the current game state
        :param game_state: the current game state
        :param actions: the list of actions that can be performed in the current game state
        """
        # Check reward
        reward = self.helper_class.reward(prev_game_state, action, game_state)

        # Convert format to q table readable
        state_q = self.helper_class.convert_state(game_state)
        action_q = self.helper_class.convert_actions(action)
        prev_state_q = self.helper_class.convert_state(prev_game_state)

        old_q = self.q_table.get((prev_state_q, action_q), 0)

        if self.sarsa_mode:
            # SARSA
            # Get next action
            self.next_action = None
            self.select_action_training(game_state, actions)

            # If terminal, update Q assuming Q(s', a') = 0
            if self.next_action is None:
                # Update Q(s, a)
                #       Q(s, a) + alpha    * (r - Q(s, a))
                new_q = old_q + self.alpha * (reward - old_q)
            else:
                # Update Q(s, a)
                #       Q(s, a) + alpha * (     r    +      gamma *    Q(s', a')    - Q(s, a))
                new_q = old_q + self.alpha * (reward + self.gamma * self.q_table.get((state_q, self.next_action), 0) - old_q)
        else:
            # Q Learning
            # Update Q(s, a)
            best_new_action = self.get_best_action(game_state, actions)

            # If terminal, update Q assuming Q(s', a') = 0
            if best_new_action is None:
                #       Q(s, a) + alpha    * (r - Q(s, a))
                new_q = old_q + self.alpha * (reward - old_q)
            else:
                action_best = self.helper_class.convert_actions(best_new_action)
                #       Q(s, a) + alpha * (     r    +      gamma *    max(Q(s', a'))    - Q(s, a))
                new_q = old_q + self.alpha * (reward + self.gamma * self.q_table.get((state_q, action_best), 0) - old_q)
        # Update this element
        self.q_table.update({(prev_state_q, action_q): new_q})
