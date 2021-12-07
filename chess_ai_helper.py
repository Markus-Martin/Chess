from game_state import GameState
import random
import constants as c


class AIChessHelper:
    """
    A helper class used by the q_learn file to interpret the chosen game. In this case, the class interprets chess.
    States and actions are represented as:
    State: (value, num_squares, king_safety)
    Action: (type, location, offset)
    """
    @staticmethod
    def convert_state(state: GameState):
        """
        Converts the given exact game state into the AI approximated form.

        :param state: the exact game state
        """
        if c.STATE_TYPE != 0:
            # Cycle through each piece and extract the parameter values from them.
            king = None

            # Calculate control values
            control_squares = {}
            for x in range(1, 9):
                for y in range(1, 9):
                    square = (x, y)  # Target square we're calculating the control value for
                    target_value = 0  # Number of friendly pieces targeting this square minus enemy pieces
                    for piece in state.pieces:
                        # Because we're iterating through each piece anyway, we may as well find king now for later use
                        if king is None and piece.colour == state.turn:
                            if piece.icon == c.KING:
                                king = piece

                        # +1 if friendly, -1 if enemy
                        direction = 1 if piece.colour == state.turn else -1

                        # Add this piece to the counter if it has sight on the location
                        if piece.has_sight(state, square):
                            target_value += direction

                    # Store the final result in control squares. If > 0, set it to 1, < 0 set to -1, = 0 set to 0
                    if target_value > 0:
                        control_squares.update({square: 1})
                    elif target_value < 0:
                        control_squares.update({square: -1})
                    else:
                        control_squares.update({square: 0})

            # Now use the control squares to find out who controls each 2x4 area on the board starting bottom left
            control_blocks = []
            for squares in c.CONTROL_SQUARES:
                cumulative_value = 0
                for square in squares:
                    cumulative_value += control_squares.get(square, 0)

                # Set control block values appropriately depending on which side dominates more
                if cumulative_value < 0:
                    control_blocks.append(-1)
                elif cumulative_value > 0:
                    control_blocks.append(1)
                else:
                    control_blocks.append(0)

            # King safety calculations
            # Use king to update parameter 3
            if king.allies == 0:
                king_safety = 0
            elif king.allies <= 2:
                king_safety = 1
            else:
                king_safety = 2

            # Time value calculation
            num_pieces = len(state.pieces)
            if num_pieces >= 24:
                time_value = 0
            elif num_pieces >= 14:
                time_value = 1
            else:
                time_value = 2

        # Now output state depending on which state type we're using
        if c.STATE_TYPE == 0:
            # For this state representation, we will brute force it without trying to capture any features
            # Get FEN representation of state
            fen = state.state_to_fen()

            # Send the FEN string to a number between 0 and 1 seeded by fen, then scale it to the state_size
            val = round(random.Random(fen).random() * c.STATE_SIZE)

            return val
        elif c.STATE_TYPE == 1:
            return tuple(control_blocks)
        else:
            # State represented by:
            # - Control values over 2x4 areas (-1 neither controls, 0 enemy controls, 1 agent controls)
            # - King safety (0 unsafe: not allies, 1 ok: 2 allies, safe: 3 allies)
            # - Time (0 early game, 1 mid game, 2 late game)
            # State space: 78,732 (ok size; preferably smaller)
            return tuple(control_blocks), king_safety, time_value

    @staticmethod
    def state_to_text(state_tuple):
        """
        Converts and returns the approximate state into a string that can be saved in a text file. The format is:
        (value, num_squares, king_safety) -> "value,num_squares,king_safety"

        :param state_tuple: the tuple that represents the state
        :return: the string that represents this state
        """
        # Join the above parameters by commas to get the desired format
        if c.STATE_TYPE != 0:
            state_text = ",".join(str(elem) for elem in state_tuple)
        else:
            # Accounting for single element state spaces
            state_text = str(state_tuple)

        return state_text

    @staticmethod
    def text_to_state(text: str):
        """
        Converts the given text into an AI approximate state and returns this state. The format is:
        "value,num_squares,king_safety" -> (value, num_squares, king_safety)

        :param text: the text we're converting to a state
        :return: the state that the text represents
        """
        # Split the text into its components
        if c.STATE_TYPE != 0:
            state_text = text.split(",")
            new_text = state_text
            # Remove non-digit characters
            for i in range(0, len(state_text)):
                if len(state_text[i]) > 1:
                    j = -1
                    for elem in state_text[i]:
                        j += 1
                        if not elem.isdigit():
                            new_text[i] = state_text[i].replace(elem, "")

            state = (int(x) for x in new_text)
        else:
            # Accounting for single element state spaces
            state = int(text)
        return state

    @staticmethod
    def text_to_action(text: str):
        """
        Converts the given text into an action which is of the form:
        "type,locationx,locationy,offsetx,offsety,promotion" -> (type, location, offset)

        :param text: the text we're converting to an action
        :return: the action that the text represents
        """
        # Split the text into its components
        text_split = text.split(",")
        if len(text_split) == 6:
            icon, loc_x, loc_y, offset_x, offset_y, promotion = text_split
            offset = (int(offset_x), int(offset_y), promotion)
        elif len(text_split) == 5:
            icon, loc_x, loc_y, offset_x, offset_y = text_split
            offset = (int(offset_x), int(offset_y))
        else:
            print(text)

        # Format location and offset appropriately
        location = (int(loc_x), int(loc_y))

        # Return the action
        return icon, location, offset

    @staticmethod
    def action_to_text(action_tuple) -> str:
        """
        Converts the given text into an action which is of the form:
        (type, location, offset) -> "type,locationx,locationy,offsetx,offsety,promotion"

        :param action_tuple: the action we're converting to text
        :return: the text that represents the given action
        """
        # Split the text into its components
        icon, location, offset = action_tuple

        # Format into list
        actions = [icon, *location, *offset]

        # Return the action
        actions_text = ",".join(str(elem) for elem in actions)
        return actions_text

    @staticmethod
    def convert_actions(actions):
        """
        Returns the list of actions into actions that are readable for the q table.

        :param actions: the list of actions
        :return: a q table readable list of the possible actions in the game state
        """
        # Action space list
        actions_q = []

        if isinstance(actions, list):
            # Cycle through all actions
            for action in actions:
                piece = action[0]
                offset = action[1]

                # By the time the actions come through, we've already updated piece positions so we revert the change
                # location = (piece.location[0] - offset[0], piece.location[1] - offset[1])
                actions_q.append((piece.icon, piece.location, offset))
        else:
            piece = actions[0]
            offset = actions[1]

            # By the time the actions come through, we've already updated piece positions so we revert the change
            # location = (piece.location[0] - offset[0], piece.location[1] - offset[1])
            return piece.icon, piece.location, offset
        return actions_q

    @staticmethod
    def reward(prev_game_state: GameState, action, game_state: GameState):
        """
        Returns R(s, a, s'). For chess, the action performed doesn't matter - just how your position improved/worsened.

        :param prev_game_state: the game state before performing the move
        :param action: the action that was taken to get to the current game state
        :param game_state: the current game state
        :return: the reward for this transition
        """
        # Get the colour of the position we're evaluating
        turn = prev_game_state.turn

        # On AI Mode 3, if we're on the last move, we need to set the reward to whether win or loss
        if c.AI_MODE == 3 and game_state.final_turn:
            if c.GAME_OUTCOME == turn:
                return c.WIN_REWARD
            elif c.GAME_OUTCOME == -1:
                return c.STALEMATE_PENALTY
            elif c.GAME_OUTCOME != turn:
                return c.LOSE_PENALTY

        # Return the difference between the board evaluations from the current to previous board
        return c.MOVE_COST + game_state.evaluate_position(turn) - prev_game_state.evaluate_position(turn)