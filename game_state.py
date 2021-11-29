from __future__ import annotations
import chess_piece
import constants as c
from typing import Tuple
import time

"""
Representation of the game state of the chess board. Holds a chess piece for each location on the board and whether
each player has castled yet.

"""


def new_location(location: Tuple[int, int], offset: Tuple[int, int]) -> Tuple[int, int]:
    """
    Performs the offset move on the given location and returns the new position.

    :param location: the location of the piece
    :param offset: the direction to move piece in
    :return: the new location after performing the move
    """
    x = location[0] + offset[0]
    y = location[1] + offset[1]
    return x, y


class GameState:
    """
    Initialises the game state with the given parameters:
    board_state = {(x, y): (piece_code, colour), ...}
    has_castled = (white_has_castled, black_has_castled)
    turn = 0 for black's turn and 1 for white's

    """
    def __init__(self, board_state: dict, turn: bool):
        self.board_state = board_state
        self.turn = turn
        self.selected_piece = None
        self.pieces = list(self.board_state.values())
        self.performing_promotion = False
        self.promotion_move = None

        count = 0
        for piece in self.pieces:
            if piece.icon == c.KING:
                count += 1

    def get_board_state(self):
        """
        Returns the board state.

        """
        return self.board_state

    def piece_at(self, location: Tuple[int, int]) -> chess_piece.ChessPiece:
        """
        Gets the piece at the given location.

        :param location: location where we want to find the piece at
        :return: the piece located at location
        """
        return self.board_state.get(location)

    def get_next_state(self, location: Tuple[int, int], offset: Tuple[int, int], pawn_promotion: str = None) -> GameState:
        """
        Returns the next game state if the given offset action was performed on the piece at location. Offsets will be
        performed regardless of whether the move is legal or not.

        :param pawn_promotion: (optional) the icon of the piece we want to promote the pawn to
        :param location: the piece we're performing an action on
        :param offset: the offset action
        :return: the next game state after performing the action
        """
        new_board = self.board_state.copy()

        # Remove piece from board to be moved later (picking piece up to move it)
        if new_board.get(location, None) is None:
            GameState(new_board, not self.turn).show_text_board()
        piece = new_board.pop(location)

        new_loc = new_location(location, offset)

        # In case of pawn promotion, we must change piece
        if pawn_promotion is not None:
            if pawn_promotion == c.QUEEN:
                new_piece = chess_piece.Queen(piece.location, piece.colour)
                new_piece.moves_taken = piece.moves_taken
                piece = new_piece

        # Add piece to new location
        new_board.update({new_loc: piece})

        # Return the new game state with these changes
        return GameState(new_board, not self.turn)

    def in_check(self, colour: bool) -> bool:
        """
        Returns whether the given colour is in check.

        :param colour: the colour we're testing for check
        """
        king_loc = None

        # Find the king's location
        for location in self.board_state.keys():
            piece = self.piece_at(location)
            if piece.colour == colour and piece.icon == c.KING:
                king_loc = location

        # Error message for if king wasn't on map
        if king_loc is None:
            if colour:
                col = "white"
            else:
                col = "black"
            print("Error:", col, "king not found")
            return False

        # Now iterate through opponent pieces and check for any targeting the king's location
        for piece in self.pieces:
            if piece.colour != colour:
                if piece.has_sight(self, king_loc):
                    return True

        return False

    def in_check_mate(self, colour: bool) -> bool:
        """
        Returns whether the given colour is in check mate.

        :param colour: the colour we're testing for check mate
        """
        no_moves = True

        # Cycle through all pieces and look for a valid move
        for piece in self.pieces:
            # Only include pieces of the appropriate colour
            if piece.colour == colour:
                # If there's an action, then we're not in check mate
                if len(piece.move_list) > 0:
                    no_moves = False
                    break

        # If you're in check and have no moves then check mate
        if self.in_check(colour) and no_moves:
            return True

        return False

    def get_action_space(self, colour: bool) -> list:
        """
        Gets the full action space for the given colour.

        :param colour: colour of the side we want to get the action space of
        :return: the action space
        """
        # Action space list
        actions = []

        for piece in self.pieces:
            if piece.icon == c.KING and piece.colour is not colour:
                king = piece

        # Cycle through all pieces
        for piece in self.pieces:
            # Only include pieces of the appropriate colour
            if piece.colour == colour:
                # Loop through all possible moves
                for offset in piece.move_list:
                    actions.append((piece, offset))

        return actions

    def evaluate_position(self, colour: bool):
        """
        Evaluates this state for given colour. Gives a numerical value where positive values are good and negative are
        bad.

        """
        if self.in_check_mate(colour):
            return c.LOSE_PENALTY
        elif self.in_check_mate(not colour):
            return c.WIN_REWARD
        elif len(self.get_action_space(self.turn)) == 0 or len(self.pieces) == 2:
            return c.STALEMATE_PENALTY

        # Material advantage scaled by development factors from heat map
        material_score = 0
        for piece in self.pieces:
            # Set strength depending on orientation of board
            if piece.colour == c.PLAYER_STARTS:
                strength = c.HEATMAPS[piece.icon][piece.location]
            else:
                # Adjust location to account for inverted heat map
                adjusted_location = (piece.location[0], 9 - piece.location[1])
                strength = c.HEATMAPS[piece.icon][adjusted_location]

            # Add or subtract to the material score depending on colour
            if piece.colour == colour:
                material_score += piece.value * strength
            else:
                material_score -= piece.value * strength

        return material_score

    def state_to_fen(self):
        """
        Sends the game state to a FEN string.

        :return: a FEN string that represents the game state
        """
        # Initialise fen
        fen = ""

        # First, make the board part of the FEN string
        num_blanks = 0  # Number of blank spaces
        for y in range(8, 0, -1):
            if y < 8:
                fen += "/"
            for x in range(1, 9):
                piece = self.board_state.get((x, y), None)
                # This is a blank space so increase the number
                if piece is None:
                    num_blanks += 1
                    continue
                else:
                    # If there are some blank spaces, append it to our FEN string and reset blanks
                    if num_blanks > 0:
                        fen += str(num_blanks)
                        num_blanks = 0

                    # Adjust the symbol that represents the piece in this square
                    if piece.colour:
                        symbol = piece.icon.upper()
                    else:
                        symbol = piece.icon.lower()

                    # Now add the piece to the FEN string
                    fen += symbol

            # Also if we reach the end of the x axis, just add the blank spaces and reset them
            if num_blanks > 0:
                fen += str(num_blanks)
                num_blanks = 0

        # Second, add the turn
        turn = "w" if self.turn else "b"
        fen += " " + turn

        pawn_passant = None
        # Third, add castling
        fen += " "
        at_least_one = False
        # Iterate through each location for rooks
        for rook_loc in [(1, 1), (8, 1), (1, 8), (8, 8)]:
            colour = c.PLAYER_STARTS if rook_loc[1] == 1 else not c.PLAYER_STARTS

            # Check if there's a rook there. Continue if it's absent or has taken moves
            rook = self.board_state.get(rook_loc, None)
            if rook is None or rook.colour != colour or rook.icon != c.ROOK or rook.moves_taken > 0:
                continue

            # Find the king
            for loc, piece in self.board_state.items():
                # Find a passant vulnerable pawn at the same time for use later
                if piece.icon == c.PAWN and piece.passant_vulnerable:
                    pawn_passant = piece
                # Find king
                if piece.icon == c.KING and piece.colour == colour:
                    king = piece

            # If the king has moved also continue
            if king.moves_taken > 0:
                continue

            # Otherwise, add to the FEN string. First step to do this is to get the appropriate symbol
            if rook_loc[0] == 1:
                symbol = c.QUEEN
            else:
                symbol = c.KING

            # Adjust case if it's black
            if not rook.colour:
                symbol = symbol.lower()

            at_least_one = True

            # Add to FEN
            fen += symbol

        # Check if at least one symbol was added. If not, then add a "-"
        if not at_least_one:
            fen += "-"

        # Fourth, add passant
        if pawn_passant is not None:
            fen += " " + chr(pawn_passant.location[0] + 96) + str(pawn_passant.location[1])
        else:
            fen += " -"

        # Finally, add moves (since last piece taken or pawn moved) which we don't currently use so default to 0
        fen += " 0 0"

        # Return the FEN string
        return fen

    @staticmethod
    def fen_to_state(fen: str) -> GameState:
        """
        Converts the given FEN string into a game state.

        :param fen: the FEN string we're converting
        :return: the game state represented in the FEN string
        """
        # Split the string into it's parts
        board, turn, castling, passant, _, _ = fen.split(" ")

        # First make the board
        board_state = {}
        y = 9

        # Rows are split by / in fen
        for row in board.split("/"):
            y -= 1
            x = 0
            for elem in row:
                x += 1
                # If the element is a digit, we must skip that many rows
                if elem.isdigit():
                    x += int(elem) - 1
                    continue
                # Otherwise, the element must be a piece
                else:
                    # Get the type of piece
                    if elem.upper() == c.KNIGHT:
                        piece_class = chess_piece.Knight
                    elif elem.upper() == c.BISHOP:
                        piece_class = chess_piece.Bishop
                    elif elem.upper() == c.ROOK:
                        piece_class = chess_piece.Rook
                    elif elem.upper() == c.QUEEN:
                        piece_class = chess_piece.Queen
                    elif elem.upper() == c.KING:
                        piece_class = chess_piece.King
                    else:
                        piece_class = chess_piece.Pawn

                    # Upper case for white, lower for black
                    if elem.isupper():
                        # White
                        piece = piece_class((x, y), True)
                    else:
                        # Black
                        piece = piece_class((x, y), False)
                    board_state.update({(x, y): piece})

        # Update passant
        if passant != "-":
            location = ord(passant[0]) - 96, int(passant[1])
            pawn = board_state[location]
            pawn.passant_vulnerable = True

        # Update castling
        # First change all rooks to have moved already
        for piece in board_state.values():
            if piece.icon == c.ROOK:
                piece.moves_taken = 1

        # Now we reset the moves to 0 if castling can occur
        if castling != "-":
            for elem in castling:
                # Get the location
                if elem == c.KING:
                    location = 8, 1
                elif elem == c.KING.lower():
                    location = 8, 8
                elif elem == c.QUEEN:
                    location = 1, 1
                else:
                    location = 1, 8

                # Adjust the location if white is oriented at the top of the screen
                if not c.PLAYER_STARTS:
                    location = location[0], 9 - location[1]

                # Reset moves to 0
                rook = board_state[location]
                rook.moves_taken = 0

        # Create the game state and return  it
        return GameState(board_state, True if turn == "w" else False)

    def show_text_board(self):
        """
        Prints the current state to the terminal as text.

        """
        # Print top of map
        print("+---+---+---+---+---+---+---+---+")
        for y in range(8, 0, -1):
            line = "+ "
            for x in range(1, 9):
                # Get the piece in the square
                piece = self.board_state.get((x, y), None)

                # Put the icon in the spot if there's a piece there, otherwise, space
                if piece is not None:
                    # If it's black then lower case it
                    if piece.colour:
                        line += piece.icon
                    else:
                        line += piece.icon.lower()
                else:
                    line += " "

                # Add the end part to the line
                line += " + "

            # Print the line and the break between above and below
            print(line)
            print("+---+---+---+---+---+---+---+---+")
