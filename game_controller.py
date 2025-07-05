import pygame
import constants as c
from chess_piece import Pawn, Knight, Bishop, Rook, Queen, King, new_location
from game_state import GameState
from renderer import Renderer
from q_learn import QLearn
from chess_ai_helper import AIChessHelper
from chess_game_interpreter import ChessGameInterpreter
from neural_network import DRLMainAction
import random
import math
import time
import copy
import numpy as np
from scipy.special import expit
from PIL import Image

"""
Start initial game state and get player inputs for moves. Perform moves and update renderer.
"""


class GameController:
    """
    The class that controls the game.

    """

    def __init__(self):
        # Variable to hold whether the agent tracks learning progress (when learning from actual games)
        self.track_progress = False

        # Make Q learning agent if that mode is selected
        if c.AI_MODE == 1 or c.AI_MODE == 2 or c.AI_MODE == 3:
            # We must also ensure there's actually an AI playing before creating the agent
            if c.PLAY_MODE == 0 or c.PLAY_MODE == 2:
                if c.USE_NEURAL_NETWORK:
                    self.agent = DRLMainAction(gamma=0.99, epsilon=1.0, lr=0.03, input_dims=[64], batch_size=1, n_actions=141, eps_end=0.01)
                else:
                    self.agent = QLearn(c.SAVE_NAME, AIChessHelper)

        # If we're learning from games create the game interpreter
        if c.AI_MODE == 3:
            self.interpreter = ChessGameInterpreter(c.LEARN_FROM, c.ELO_RANGE)

            # Track learning progress when we're showing stats
            if c.SHOW_STATS:
                self.track_progress = True

        # Choose the function each AI will pick it's moves from
        if c.PLAY_MODE == 0 or c.PLAY_MODE == 2:
            if c.USE_NEURAL_NETWORK and c.AI_MODE == 1:
                self.choose_action = {True: self.agent.choose_action, False: self.agent.choose_action}
            elif c.USE_NEURAL_NETWORK and c.AI_MODE == 2:
                self.choose_action = {True: self.agent.get_best_action, False: self.agent.get_best_action}
            elif c.AI_MODE == 1:
                #                     White action selection function          Black action selection function
                self.choose_action = {True: self.agent.select_action_training, False: self.agent.select_action_training}
            elif c.AI_MODE == 2:
                #                     White action selection function   Black action selection function
                self.choose_action = {True: self.agent.get_best_action, False: self.agent.get_best_action}
            elif c.AI_MODE == 3:
                #                     White action selection function          Black action selection function
                self.choose_action = {True: self.interpreter.get_next_move, False: self.interpreter.get_next_move}

        # Start game in initial state. Set up chess board according to rules and add pieces to arrays above
        self.state = self.get_init_state()
        self.render = None
        self.has_updated = False
        self.total_half_moves = 0
        self.prev_state = None
        self.prev_action = None
        self.game_stat = [[], []]

        # Use heat maps to update the dictionaries in constant.py
        self.interpret_heatmaps()

        # Begin game
        self.play_game()

    @staticmethod
    def get_init_state():
        """
        Returns the basic initial state of a chess board with the pawns in a line and the special units behind them.

        """
        return GameState.fen_to_state("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    def play_game(self):
        """
        Runs the game, specifying PLAY_MODE in constants will change whether it's run in PvP, PvAI, or AIvAI mode

        """

        # Only intialise pygame if we're player pve or pvp mode
        if c.PLAY_MODE <= 1 or c.RENDER_FOR_AI:
            # Start pygame
            pygame.init()

            # Create renderer
            self.render = Renderer()

            # Render game
            self.render.draw(self.state)
        elif c.PLAY_MODE == 2:
            print("Episodes\tTime used\tTime Left")

        # Basic stat tracking variables
        t0 = time.time()
        t = time.time()
        episodes = 0

        # Open writer for collecting data if the option is turned on
        if c.SHOW_STATS:
            stat_file = open("data1.txt", "w")

        # ----------------------------------- Loop for episodes ----------------------------------- #
        # Repeat games until allocated time is over for AIvAI, other modes just run once
        while c.PLAY_MODE == 2 and c.ALLOCATED_RUN_TIME > 0 and time.time() - t0 < c.ALLOCATED_RUN_TIME or episodes <= 0:
            # Reset quantities for next episode
            self.state = self.get_init_state()
            self.has_updated = False
            self.total_half_moves = 0
            self.prev_state = None
            self.game_stat = [[], []]

            # Check learning progress by playing itself every 10 episodes. This only matters when learning from games
            if c.USE_NEURAL_NETWORK and self.track_progress and c.PLAY_MODE != 1:
                if episodes % 10 == 0:
                    c.AI_MODE = 1
                    self.choose_action = {True: self.agent.choose_action, False: self.agent.choose_action}
                elif episodes % 10 == 1:
                    c.AI_MODE = 3
                    self.choose_action = {True: self.interpreter.get_next_move, False: self.interpreter.get_next_move}

            # Increment episode
            episodes += 1

            # Pick the next game if we're learning from data
            if c.AI_MODE == 3:
                # Play the next game
                self.interpreter.next_game()

            # Every 100 episodes print stats and update q table
            if episodes % 100 == 0:
                # Print episode number, time used for those past 100 episodes and the time left
                print(episodes, "\t", round((time.time() - t) / 60, 3), "min", "\t", round((c.ALLOCATED_RUN_TIME - (time.time() - t0)) / 60, 3), "min", sep="")

                # Update q table and t (time since starting the last 1000s episode)
                t = time.time()
                if c.AI_MODE == 1 or c.AI_MODE == 2 or c.AI_MODE == 3:
                    if c.USE_NEURAL_NETWORK:
                        self.agent.save()
                    else:
                        self.agent.save_q_table()

            # Initially running
            running = True

            # ----------------------------------- Game Loop ----------------------------------- #
            while running:
                # For AIvAI mode, delay the rendering of the game for ease of watching
                if c.PLAY_MODE == 2 and c.RENDER_FOR_AI:
                    time.sleep(c.AI_MOVE_DELAY)

                # Update possible moves and break loop if game over
                if not self.has_updated and self.update_game():
                    break

                # Break the loop also if the AI exceeds 1000 half steps
                if self.total_half_moves >= 1000:
                    break

                # Computer's turn if it's not player's turn in PvAI or if the mode is AIvAI
                if (c.PLAY_MODE == 0 and c.PLAYER_STARTS is not self.state.turn) or c.PLAY_MODE == 2:
                    self.computer_action()
                    # End game even if check mate/stale mate not reached when using simulated games
                    if c.AI_MODE == 3 and len(self.interpreter.move_list) == self.total_half_moves:
                        break

                # Only allow the player side to run if the play mode is PvAI or PvP
                if c.PLAY_MODE <= 1 or c.RENDER_FOR_AI:
                    # Go through each GUI event
                    for event in pygame.event.get():
                        # What to do when quitting
                        if event.type == pygame.QUIT:
                            running = False

                        # Left mouse button down event
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            if event.button == 1:
                                # As long as it's the player's turn in PvAI or anyone's turn in PvP, perform player action
                                if (c.PLAY_MODE == 0 and c.PLAYER_STARTS == self.state.turn) or c.PLAY_MODE == 1:
                                    self.player_action()

                # Render the game (when the options say to)
                if c.PLAY_MODE <= 1 or c.RENDER_FOR_AI:
                    self.render.draw(self.state)

            # Store half move stats (when the options say to)
            if c.SHOW_STATS and c.AI_MODE != 3:
                mean_1 = round(np.mean(expit(self.game_stat[0])), 3)  # One step reward
                mean_2 = round(np.mean(expit(self.game_stat[1])), 3)  # Two step reward
                # Percent of the one step rewards that are better than just the move cost
                percent = round(sum(num > c.MOVE_COST for num in self.game_stat[0]) / len(self.game_stat[0]), 3)
                stat_file.write(str(episodes) + "\t" + str(self.total_half_moves)
                                + "\t" + str(mean_1) + "\t" + str(mean_2) + "\t" + str(percent) + "\n")
                stat_file.flush()

        # ----------------------------------- Final updates ----------------------------------- #
        print(episodes, "\t", round((time.time() - t) / 60, 3), "min", "\t", round((c.ALLOCATED_RUN_TIME - (time.time() - t0)) / 60, 3), "min", sep="")

        # Close the writer
        if c.SHOW_STATS:
            stat_file.close()

        # Save Q table
        if (c.AI_MODE == 1 or c.AI_MODE == 2 or c.AI_MODE == 3) and c.PLAY_MODE != 1:
            if c.USE_NEURAL_NETWORK:
                # Save the neural network knowledge
                self.agent.save()
            else:
                self.agent.save_q_table()

        # Close the reader for chess games
        if c.AI_MODE == 3:
            self.interpreter.file.close()

        # If game over, just display the board
        if c.PLAY_MODE <= 1 or c.RENDER_FOR_AI:
            self.board_display()

    def player_action(self):
        """
        Handles the player's turn

        """
        # Get position
        pos = pygame.mouse.get_pos()

        # Convert position to coordinates
        x = math.floor(pos[0] / c.CELL_SIZE)
        y = math.floor((c.MAP_HEIGHT - pos[1]) / c.CELL_SIZE)

        # If within bounds, do something
        if 0 <= x < self.render.size and 0 <= y < self.render.size:
            # Proper position coordinates
            location = (x + 1, y + 1)

            pawn_promotion = None

            # If pawn promotion is underway, we're selecting the type of piece to put in location
            if self.state.performing_promotion:
                if 3 <= location[0] <= 4 and 4 <= location[1] <= 5:
                    pawn_promotion = c.QUEEN
                elif 5 <= location[0] <= 6 and 4 <= location[1] <= 5:
                    pawn_promotion = c.KNIGHT

                piece = None

            else:
                # Get the piece at this position
                piece = self.state.piece_at(location)

            # Currently selected
            selected = self.state.selected_piece

            # Select the piece if it's friendly. Deselect if it's already selected.
            if piece is not None and piece.colour == self.state.turn:
                if selected == piece:
                    self.state.selected_piece = None
                else:
                    self.state.selected_piece = piece
            # If the selected piece is not None and the selected point is a move, allow it
            elif selected is not None:
                # If a promotion is underway, set the offset accordingly
                if self.state.performing_promotion:
                    offset = self.state.promotion_move
                else:
                    offset = (location[0] - selected.location[0], location[1] - selected.location[1])

                # Special case of pawn promotion
                if self.state.selected_piece.icon == c.PAWN:
                    new_loc = new_location(self.state.selected_piece.location, offset)
                    if (new_loc[1] == 8 or new_loc[1] == 1) and (*offset, c.QUEEN) in self.state.selected_piece.move_list:
                        # Force the player to choose knight or queen
                        self.state.performing_promotion = True
                        self.state.promotion_move = offset

                # Check if move is in move list or special pawn promotion case move is in move list
                if offset in selected.move_list or (
                        pawn_promotion is not None and (*offset, pawn_promotion) in selected.move_list):
                    # Perform the move depending on if a pawn is being promoted
                    self.has_updated = False
                    if pawn_promotion is None:
                        self.state = selected.move(self.state, offset)
                    else:
                        self.state = selected.move(self.state, offset, pawn_promotion)
                    self.state.performing_promotion = False
                    self.state.promotion_move = None

    def computer_action(self):
        """
        Handles the computer's turn.

        """
        # Reset highlighted piece
        self.state.selected_piece = None

        # Get all possible actions
        actions = self.state.get_action_space(self.state.turn)

        # Ensure list isn't empty
        if len(actions) == 0:
            print("Computer in check mate or stale mate")
            return

        if c.AI_MODE == 1 or c.AI_MODE == 2 or c.AI_MODE == 3:
            # Q-learning (learning phase) or (training phase)
            chosen_action = self.choose_action[self.state.turn](self.state, actions)
        else:
            # Random mode
            chosen_action = random.choice(actions)

        # Store previous state and state from last turn of the current colour
        last_turn_state = None
        if self.prev_state is not None:
            last_turn_state = self.prev_state
        # Store a deep copy of the current state so future updates do not
        # mutate the previous state used for learning.
        self.prev_state = copy.deepcopy(self.state)

        piece = chosen_action[0]
        offset = chosen_action[1]

        # Set piece to actual piece if we're in AI mode 3
        if c.AI_MODE == 3:
            piece = self.state.board_state[piece.location]

        # Action is pawn promotion
        if len(offset) >= 3:
            # Update state with promotion
            promotion = offset[2]
            offset = (offset[0], offset[1])
            self.state = piece.move(self.state, offset, promotion)
        else:
            # Update state
            self.state = piece.move(self.state, offset)

        self.has_updated = False
        self.total_half_moves += 1

        # Update num_moves and num_actions
        if not c.USE_NEURAL_NETWORK and (c.AI_MODE == 1 or c.AI_MODE == 2 or c.AI_MODE == 3):
            # Increment the states visited and actions performed in that state if we're using ucb
            if self.agent.exploration_strategy == 1:
                action_q = AIChessHelper.convert_actions(chosen_action)
                state_q = AIChessHelper.convert_state(self.state)
                self.agent.num_moves[state_q] = self.agent.num_moves.get(state_q, 0) + 1
                self.agent.num_actions[(state_q, action_q)] = self.agent.num_actions.get((state_q, action_q), 0) + 1

        # If it's the final run then let the program know so it can run special cases in the reward function
        if c.AI_MODE == 3 and len(self.interpreter.move_list) == self.total_half_moves:
            self.state.final_turn = True

        # Update q values for PvAI and AIvAI when it's been a full cycle back to the computer's turn
        if (c.AI_MODE == 1 or c.AI_MODE == 2 or c.AI_MODE == 3) and last_turn_state is not None:
            # Use everything to update q values. AI knows action from 2 half turns ago caused this current state because
            # it has to account for the other colour playing in between. Thus, we use 2 turns ago state and the previous
            # action that got us to the previous state.
            if c.USE_NEURAL_NETWORK:
                # Check for terminal state
                terminal = self.state.in_check_mate(True) or self.state.in_check_mate(False) \
                           or len(self.state.get_action_space(self.state.turn)) == 0 or len(self.state.pieces) == 2
                self.agent.store_transition(last_turn_state, self.prev_action, self.state, terminal)
                self.agent.learn()
            else:
                # Get actions for next move
                for piece in self.state.board_state.values():
                    piece.get_possible_moves(self.state)
                actions = self.state.get_action_space(self.state.turn)

                self.agent.q_update(last_turn_state, self.prev_action, self.state, actions)

        # If we're using AI Mode 3 and this is the final action, q update again but for just this action
        if c.AI_MODE == 3 and self.state.final_turn:
            if c.USE_NEURAL_NETWORK:
                self.agent.store_transition(self.prev_state, chosen_action, self.state, True)
                self.agent.learn()
            else:
                self.agent.q_update(self.prev_state, chosen_action, self.state, None)

        # Update previous action (piece has already been moved so we must revert move to get proper original position)
        old_piece = copy.copy(chosen_action[0])
        if c.AI_MODE != 3:
            old_piece.location = (old_piece.location[0] - chosen_action[1][0], old_piece.location[1] - chosen_action[1][1])
        self.prev_action = old_piece, chosen_action[1]

        # Immediate reward of a move
        if last_turn_state is not None:
            immediate_reward = AIChessHelper.reward(self.prev_state, chosen_action, self.state)
            two_move_reward = AIChessHelper.reward(last_turn_state, chosen_action, self.state)

            # Print so we can track how well the AI is doing
            self.game_stat[0].append(immediate_reward)
            self.game_stat[1].append(two_move_reward)

    def update_game(self) -> bool:
        """
        Updates the move list for each colour and checks for game ending states. Returns true if the game is over or
        false if it's not.

        :return: T/F depending on if the game is over
        """
        # Reset has_updated to true so the program knows not to update again until the next turn
        self.has_updated = True

        # Update move lists
        for piece in self.state.pieces:
            piece.get_possible_moves(self.state)

        # Check win/loss/stalemate conditions
        if self.state.in_check_mate(True):
            self.state.final_turn = True
            if c.PLAY_MODE != 1 and (c.AI_MODE == 1 or c.AI_MODE == 2 or c.AI_MODE == 3):
                if c.USE_NEURAL_NETWORK:
                    self.agent.store_transition(self.prev_state, self.prev_action, self.state, True)
                    self.agent.learn()
                else:
                    self.agent.q_update(self.prev_state, self.prev_action, self.state, None)
            return True
        elif self.state.in_check_mate(False):
            self.state.final_turn = True
            if c.PLAY_MODE != 1 and (c.AI_MODE == 1 or c.AI_MODE == 2 or c.AI_MODE == 3):
                if c.USE_NEURAL_NETWORK:
                    self.agent.store_transition(self.prev_state, self.prev_action, self.state, True)
                    self.agent.learn()
                else:
                    self.agent.q_update(self.prev_state, self.prev_action, self.state, None)
            return True
        elif len(self.state.get_action_space(self.state.turn)) == 0 or len(self.state.pieces) == 2:
            self.state.final_turn = True
            if c.PLAY_MODE != 1 and (c.AI_MODE == 1 or c.AI_MODE == 2 or c.AI_MODE == 3):
                if c.USE_NEURAL_NETWORK:
                    self.agent.store_transition(self.prev_state, self.prev_action, self.state, True)
                    self.agent.learn()
                else:
                    self.agent.q_update(self.prev_state, self.prev_action, self.state, None)
            return True
        return False

    def board_display(self):
        """
        Shows just the board with no possible interactions. This should be used when game over is achieved.

        """
        # Continue drawing until we close window
        running = True
        while running:
            for event in pygame.event.get():
                # What to do when quitting
                if event.type == pygame.QUIT:
                    running = False
            self.render.draw(self.state)

    @staticmethod
    def interpret_heatmaps():
        """
        Interprets the heat maps for the strength of a piece's position and updates them as dictionaries in
        constants.py.

        """
        # Iterate through each type of piece
        for icon in c.PIECE_LIST:
            # Access map rgb values
            map_name = c.HEATMAP_FILE[icon]
            image = Image.open(map_name)
            rgb = image.load()

            # Iterate through each x, y of map
            for x in range(0, 8):
                for y in range(0, 8):
                    # Get the numerical strength of this position
                    strength = sum(rgb[x, y]) / 3 / 255

                    # Place it into dictionary
                    c.HEATMAPS[icon].update({(x + 1, y + 1): strength})
