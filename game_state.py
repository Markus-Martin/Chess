from __future__ import annotations
import chess_piece
import constants as c
from typing import Tuple

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
