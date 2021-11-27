import pygame
import constants as c
from chess_piece import Pawn, Knight, Bishop, Rook, Queen, King, new_location
from game_state import GameState
from renderer import Renderer
import random
import math
import time

"""
Start initial game state and get player inputs for moves. Perform moves and update renderer.
"""


class GameController:
    """
    The class that controls the game.

    """

    def __init__(self):
        # Variables for how to play game
        # Player vs Player?
        # Player vs AI?
        # AI vs AI?

        # Start game in initial state. Set up chess board according to rules and add pieces to arrays above
        board = self.get_basic_board()
        self.state = GameState(board, True)
        self.render = None
        self.has_updated = False
        self.total_half_moves = 0
        c.RENDER_FOR_AI = True

        # Begin game
        self.play_game()

    def get_basic_board(self):
        """
        Returns the basic initial state of a chess board with the pawns in a line and the special units behind them.

        """
        board = {}
        # Iterate along x axis
        for x in range(1, 9):
            # Add pawns to board
            board.update({(x, 2): Pawn((x, 2), c.PLAYER_STARTS)})
            board.update({(x, 7): Pawn((x, 7), not c.PLAYER_STARTS)})

            # Add special pieces
            dist_center = abs(x - 4.5)
            if dist_center == 0.5:
                # Special case of King/Queen
                if x == 4:
                    board.update({(x, 1): Queen((x, 1), c.PLAYER_STARTS)})
                    board.update({(x, 8): Queen((x, 8), not c.PLAYER_STARTS)})
                else:
                    board.update({(x, 1): King((x, 1), c.PLAYER_STARTS)})
                    board.update({(x, 8): King((x, 8), not c.PLAYER_STARTS)})
                continue

            # Non King/Queen
            if dist_center == 3.5:
                # Rook
                piece = Rook
            elif dist_center == 2.5:
                # Knight
                piece = Knight
            elif dist_center == 1.5:
                # Bishop
                piece = Bishop
            else:
                piece = Pawn
                print("Error in creating base board.")

            board.update({(x, 1): piece((x, 1), c.PLAYER_STARTS)})
            board.update({(x, 8): piece((x, 8), not c.PLAYER_STARTS)})

        return board

    def play_game(self):
        """
        Runs the game, specifying PLAY_MODE in constants will change whether it's run in PvP, PvAI, or AIvAI mode

        """
        begin_time = time.time()
        # Only intialise pygame if we're player pve or pvp mode
        if c.PLAY_MODE <= 1 or c.RENDER_FOR_AI:
            # Start pygame
            pygame.init()

            # Create renderer
            self.render = Renderer()

            # Render game
            self.render.draw(self.state)
        elif c.PLAY_MODE == 2:
            print("Moves\tPieces remaining\tTime taken")


        # Initially running
        running = True

        # Main game loop
        while running:
            # Print moves and pieces remaining in AIvAI
            if c.PLAY_MODE == 2:
                end_time = (time.time() - begin_time) * 1000
                print(self.total_half_moves, "\t", len(self.state.pieces), "\t", end_time, "ms", sep="")
                begin_time = time.time()
                time.sleep(c.AI_MOVE_DELAY)

            # Update possible moves and break loop if game over
            if not self.has_updated and self.update_game():
                break

            # Computer's turn if it's not player's turn in PvAI or if the mode is AIvAI
            if (c.PLAY_MODE == 0 and c.PLAYER_STARTS is not self.state.turn) or c.PLAY_MODE == 2:
                self.computer_action()

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

            # Render if PvP or PvAI
            if c.PLAY_MODE <= 1 or c.RENDER_FOR_AI:
                self.render.draw(self.state)

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
                    if new_loc[1] == 8 or new_loc[1] == 1:
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
                    self.total_half_moves += 1

    def computer_action(self):
        """
        Handles the computer's turn.

        """
        # Reset highlighted piece
        self.state.selected_piece = None

        # For now, perform a random move
        actions = self.get_action_space(self.state.turn)

        # Ensure list isn't empty
        if len(actions) > 0:
            chosen_action = random.choice(actions)
        else:
            print("Computer in check mate or stale mate")
            return

        piece = chosen_action[0]
        offset = chosen_action[1]
        # Action is pawn promotion
        if len(offset) >= 3:
            promotion = offset[2]
            offset = (offset[0], offset[1])
            self.state = piece.move(self.state, offset, promotion)
            return

        self.state = piece.move(self.state, offset)
        self.has_updated = False
        self.total_half_moves += 1

    def get_action_space(self, colour: bool) -> list:
        """
        Gets the full action space for the given colour.

        :param colour: colour of the side we want to get the action space of
        :return: the action space
        """
        # Action space list
        actions = []

        # Cycle through all pieces
        for piece in self.state.pieces:
            # Only include pieces of the appropriate colour
            if piece.colour == colour:
                # Loop through all possible moves
                for offset in piece.move_list:
                    actions.append((piece, offset))

        return actions

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
            print("Black Wins!")
            return True
        elif self.state.in_check_mate(False):
            print("White Wins!")
            return True
        elif len(self.get_action_space(self.state.turn)) == 0 or len(self.state.pieces) == 2:
            print("Stalemate")
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