from game_state import GameState, new_location
import constants as c
import itertools as it
from typing import Tuple

"""
The representation of a chess piece. It can hold the following quantities:
 - Value = numerical value of the piece's worth
 - List of moves = list of the possible changes of position that are possible (e.g. (-3, -3))
 - Location = place in (x, y) space for it to appear (None if the piece is taken)
 
"""


class ChessPiece:
    """
    location = location of the piece in (x, y) beginning at (1, 1) in bottom left corner
    colour = colour of piece, 0 for black, 1 for white
    """
    def __init__(self, location: Tuple[int, int], colour: bool):
        self.location = location
        self.colour = colour
        self.move_list = []
        self.icon = "X"
        self.moves_taken = 0

    def __repr__(self):
        # Colour of piece
        col = "White" if self.colour else "Black"

        # Name of piece
        if self.icon == c.PAWN:
            name = "Pawn"
        elif self.icon == c.KNIGHT:
            name = "Knight"
        elif self.icon == c.BISHOP:
            name = "Bishop"
        elif self.icon == c.ROOK:
            name = "Rook"
        elif self.icon == c.QUEEN:
            name = "Queen"
        elif self.icon == c.KING:
            name = "King"
        else:
            name = "None"

        # Location of piece
        location = "(" + str(self.location[0]) + ", " + str(self.location[1]) + ")"

        return " ".join([col, name, location])

    def check_basic(self, state: GameState, offset: Tuple[int, int]) -> int:
        """
        Does a basic check on the given move to ensure it doesn't put pieces out of bounds or on friendly pieces. It
        also ensures the move doesn't place the player in check. Different outputs depending on the reason for failing
        the basic check:

        1  - Passed the basic check
        0  - Failed because of moving into check
        -1 - Failed because of landing on friendly piece
        -2 - Failed because out of bounds

        :param state: the current state of the board
        :param offset: the move this piece is performing
        :return: different values depending on what type of failure or success
        """
        # New location
        new_loc = new_location(self.location, offset)

        # If new location is OOB ignore this move
        if any(new_loc[i] > 8 or new_loc[i] <= 0 for i in range(2)):
            return -2

        piece = state.piece_at(new_loc)

        # If there's a friendly piece in the new location, ignore it because we cannot take friendly pieces
        if piece is not None and piece.colour is self.colour:
            return -1

        # Check for illegal moves due to check
        next_state = state.get_next_state(self.location, offset)
        if next_state.in_check(self.colour):
            return 0

        return 1

    def has_sight(self, state: GameState, location: Tuple[int, int]) -> bool:
        """
        Checks whether this piece has line of sight on the location. Line of sight means it could attack that location
        assuming it wasn't blocked by check rules from targeted king.

        :param state: the current state of the board
        :param location: the location we're checking for line of sight at
        :return: T/F depending on if there's line of sight
        """
        pass

    def move(self, state: GameState, offset: Tuple[int, int], pawn_promotion: str = None) -> GameState:
        """
        Performs the given offset move on the piece and returns the updated game state.

        :param pawn_promotion: (optional) the icon of the piece we want to promote the pawn to
        :param state: the state of the game before performing the move
        :param offset: the move the piece is performing
        :return: the updated game state
        """
        # Check if the move is possible, and if so update location then return new state
        # Details on the end of this if statement are for pawn promotion moves
        # It's an unnecessary check if the AI mode is 3
        if c.AI_MODE == 3 or offset in self.move_list or (pawn_promotion is not None and (*offset, pawn_promotion) in self.move_list):
            # Piece we're taking
            target_loc = new_location(self.location, offset)
            target_piece = state.piece_at(target_loc)

            # Pawn promotion
            if self.icon == c.PAWN and pawn_promotion is not None:
                if pawn_promotion == c.QUEEN:
                    new_piece = Queen(self.location, self.colour)
                elif pawn_promotion == c.KNIGHT:
                    new_piece = Knight(self.location, self.colour)
                elif pawn_promotion == c.ROOK:
                    new_piece = Rook(self.location, self.colour)
                elif pawn_promotion == c.BISHOP:
                    new_piece = Bishop(self.location, self.colour)
                else:
                    new_piece = self
                    print("Error: chosen pawn promotion wasn't valid")

                new_piece.moves_taken = self.moves_taken
                state.board_state.update({self.location: new_piece})

            # Update positions
            new_state = state.get_next_state(self.location, offset, pawn_promotion)

            # In case of a piece taking a vulnerable pawn we must remove it
            if self.icon == c.PAWN and abs(offset[0]) > 0 and target_piece is None:
                actual_target = new_location(self.location, (offset[0], 0))
                removed = new_state.board_state.pop(actual_target)
                new_state.pieces.remove(removed)

            # In case of castling
            if self.icon == c.KING and abs(offset[0]) == 2 and target_piece is None:
                # Direction of castling
                direction = round(offset[0] / 2)

                # Get rook location from direction of castling
                if direction < 0:
                    rook_loc = new_location(self.location, (-4, 0))
                else:
                    rook_loc = new_location(self.location, (3, 0))

                # Move rook to position directly to the left/right of king
                rook = new_state.board_state.pop(rook_loc)
                new_rook_loc = new_location(target_loc, (-direction, 0))
                new_state.board_state.update({new_rook_loc: rook})
                rook.location = new_rook_loc

            # Update location and moves taken depending on what piece we're moving
            if pawn_promotion is not None:
                new_piece.location = new_location(self.location, offset)
                new_piece.moves_taken += 1
            else:
                self.moves_taken += 1
                self.location = new_location(self.location, offset)

            # If the piece was a pawn that moved with offset 2, make it vulnerable
            if self.icon == c.PAWN and abs(offset[1]) == 2:
                self.passant_vulnerable = True

            # Return the new state
            return new_state

        # In case of impossible move, return state
        return state

    def get_possible_moves(self, state: GameState) -> list:
        pass


class Pawn(ChessPiece):
    """
    The pawn piece that can move forwards when it's not blocked and diagonally forward when there is a piece to take.

    """
    def __init__(self, location: Tuple[int, int], colour: bool):
        super().__init__(location, colour)
        self.value = c.PIECE_VALUES[c.PAWN]
        self.icon = c.PAWN
        self.passant_vulnerable = False

    def has_sight(self, state: GameState, location: Tuple[int, int]) -> bool:
        """
        Checks whether this piece has line of sight on the location. Line of sight means it could attack that location
        assuming it wasn't blocked by check rules from targeted king.

        :param state: the current state of the board
        :param location: the location we're checking for line of sight at
        :return: T/F depending on if there's line of sight
        """
        # Direction the pawn can travel in
        direction = 1 if self.colour == c.PLAYER_STARTS else -1

        # Pawns can only attack the location if it's on the diagonal.
        for offset in [(-1, direction), (1, direction)]:
            if new_location(self.location, offset) == location:
                return True

        return False

    def get_possible_moves(self, state: GameState) -> list:
        """
        Returns the list of possible moves for the pawn given the current game state.

        :param state: the current game state
        :return: a list of possible moves for the pawn
        """
        # Reset passant vulnerable with new update
        if state.turn == self.colour and self.passant_vulnerable:
            self.passant_vulnerable = False

        moves = []

        # Set y direction of piece depending on colour
        direction = 1 if self.colour == c.PLAYER_STARTS else -1

        # Check if move directly forward is possible (up to 2 if pawn hasn't moved yet)
        for i in [1, 2]:
            # Continue if the pawn cannot move two spaces or is blocked by the space directly in front
            if i == 2 and (self.moves_taken > 0 or (0, direction) not in moves):
                continue

            offset = (0, i * direction)
            # Check basic conditions on the given state and action (not OOB, not moving on friendly, etc.)
            if self.check_basic(state, offset) > 0:
                # Add move if there is no piece in the offset direction
                new_loc = new_location(self.location, offset)
                if state.piece_at(new_loc) is None:
                    # Check for pawn promotion
                    if new_loc[1] == 8 or new_loc[1] == 1:
                        for pawn_promotion in [c.QUEEN, c.KNIGHT]:
                            offset2 = (*offset, pawn_promotion)
                            moves.append(offset2)
                    else:
                        moves.append(offset)

        # Check if diagonal move is possible
        for i in [1, -1]:
            offset = (i, direction)
            # Check basic conditions on the given state and action (not OOB, not moving on friendly, etc.)
            if self.check_basic(state, offset) <= 0:
                continue

            new_loc = new_location(self.location, offset)
            piece = state.piece_at(new_loc)
            # Add move if there's an enemy piece there
            if piece is not None and piece.colour is not self.colour:
                # Check for pawn promotion
                if new_loc[1] == 8 or new_loc[1] == 1:
                    for pawn_promotion in [c.QUEEN, c.KNIGHT]:
                        offset2 = (*offset, pawn_promotion)
                        moves.append(offset2)
                else:
                    moves.append(offset)

            # Or add the move if the adjacent targeted piece is passant vulnerable
            piece2 = state.piece_at(new_location(self.location, (offset[0], 0)))
            if piece2 is not None and piece2.colour is not self.colour and piece2.icon == c.PAWN and piece2.passant_vulnerable:
                moves.append(offset)

        self.move_list = moves
        return moves


class Rook(ChessPiece):
    """
    The rook piece that can move in straight lines until it captures a piece or is blocked.

    """

    def __init__(self, location: Tuple[int, int], colour: bool):
        super().__init__(location, colour)
        self.value = c.PIECE_VALUES[c.ROOK]
        self.icon = c.ROOK

    def has_sight(self, state: GameState, location: Tuple[int, int]) -> bool:
        """
        Checks whether this piece has line of sight on the location. Line of sight means it could attack that location
        assuming it wasn't blocked by check rules from targeted king.

        :param state: the current state of the board
        :param location: the location we're checking for line of sight at
        :return: T/F depending on if there's line of sight
        """
        # First check the offset actually matches a possible direction for a rook
        actual_offset = (location[0] - self.location[0], location[1] - self.location[1])

        # One value zero, the other between 1 and 7
        if actual_offset[0] == 0 and 1 <= abs(actual_offset[1]) <= 7:
            index = 1  # Index for x or y direction offset ( x = 0, y = 1 )
        elif actual_offset[1] == 0 and 1 <= abs(actual_offset[0]) <= 7:
            index = 0
        else:
            return False

        # Get direction from the sign of the offset
        end_offset = actual_offset[index]
        direction = round(end_offset / abs(end_offset))

        # Iterate through possible points until the actual offset and ensure line of sight is there
        for distance in range(direction, end_offset, direction):
            if index == 0:
                offset = (distance, 0)
            else:
                offset = (0, distance)

            # Calculate the new location
            new_loc = new_location(self.location, offset)

            # if there's a piece in any of these locations, it doesn't have line of sight (regardless of colour)
            piece = state.piece_at(new_loc)
            if piece is not None:
                return False

        return True

    def get_possible_moves(self, state: GameState) -> list:
        """
        Returns the list of possible moves for the rook given the current game state.

        :param state: the current game state
        :return: a list of possible moves for the pawn
        """
        moves = []

        # Iterate through each possible movement up, down, left, right
        for direction in [1, -1]:
            for direction2 in [0, 1]:
                for i in range(1, 9):
                    # Use direction2 to determine whether horizontal or vertical movements and direction for up/down
                    if direction2:
                        # Horizontal
                        offset = (direction * i, 0)
                    else:
                        # Vertical
                        offset = (0, direction * i)

                    # Check basic conditions on the given state and action (not OOB, not moving on friendly, etc.)
                    val = self.check_basic(state, offset)
                    if val < 0:
                        # Break for OOB and friendly
                        break
                    elif val == 0:
                        # Continue for if state is landing in check
                        continue

                    # New location
                    new_loc = new_location(self.location, offset)

                    piece = state.piece_at(new_loc)

                    # Add move to vacant location or opponent location. Also break on collision with opponent.
                    if piece is None:
                        moves.append(offset)
                    elif piece.colour is not self.colour:
                        moves.append(offset)
                        break

        self.move_list = moves
        return moves


class Knight(ChessPiece):
    """
    The knight piece that can move in L shapes regardless of if there's a piece in the way.

    """

    def __init__(self, location: Tuple[int, int], colour: bool):
        super().__init__(location, colour)
        self.value = c.PIECE_VALUES[c.KNIGHT]
        self.icon = c.KNIGHT

    def has_sight(self, state: GameState, location: Tuple[int, int]) -> bool:
        """
        Checks whether this piece has line of sight on the location. Line of sight means it could attack that location
        assuming it wasn't blocked by check rules from targeted king.

        :param state: the current state of the board
        :param location: the location we're checking for line of sight at
        :return: T/F depending on if there's line of sight
        """
        # First check the offset actually matches a possible direction for a knight
        actual_offset = (location[0] - self.location[0], location[1] - self.location[1])

        if actual_offset in c.KNIGHT_OFFSETS:
            return True

        return False

    def get_possible_moves(self, state: GameState) -> list:
        """
        Returns the list of possible moves for the knight given the current game state.

        :param state: the current game state
        :return: a list of possible moves for the pawn
        """
        moves = []

        # Iterate through each combination of a 1 with a 2 in each order (L shaped offset)
        for offset in c.KNIGHT_OFFSETS:
            # Check basic conditions on the given state and action (not OOB, not moving on friendly, etc.)
            if self.check_basic(state, offset) <= 0:
                continue

            # Now, we know it isn't illegal so add it to the move list
            moves.append(offset)

        self.move_list = moves
        return moves


class Bishop(ChessPiece):
    """
    The bishop piece that can move in diagonal lines until it captures a piece or is blocked.

    """

    def __init__(self, location: Tuple[int, int], colour: bool):
        super().__init__(location, colour)
        self.value = c.PIECE_VALUES[c.BISHOP]
        self.icon = c.BISHOP

    def has_sight(self, state: GameState, location: Tuple[int, int]) -> bool:
        """
        Checks whether this piece has line of sight on the location. Line of sight means it could attack that location
        assuming it wasn't blocked by check rules from targeted king.

        :param state: the current state of the board
        :param location: the location we're checking for line of sight at
        :return: T/F depending on if there's line of sight
        """
        # First check the offset actually matches a possible direction for a bishop
        actual_offset = (location[0] - self.location[0], location[1] - self.location[1])

        # Both values must be same otherwise not valid offset
        if abs(actual_offset[0]) != abs(actual_offset[1]):
            return False

        # Get direction from the sign of the offset
        direction = [0, 0]
        for i in range(2):
            if actual_offset[i] != 0:
                direction[i] = round(actual_offset[i] / abs(actual_offset[i]))

        # Iterate through possible points until the actual offset and ensure line of sight is there
        for distance in range(1, abs(actual_offset[0])):
            offset = (direction[0] * distance, direction[1] * distance)

            # Calculate the new location
            new_loc = new_location(self.location, offset)

            # if there's a piece in any of these locations, it doesn't have line of sight (regardless of colour)
            piece = state.piece_at(new_loc)
            if piece is not None:
                return False

        return True

    def get_possible_moves(self, state: GameState) -> list:
        """
        Returns the list of possible moves for the bishop given the current game state.

        :param state: the current game state
        :return: a list of possible moves for the pawn
        """
        moves = []

        # Iterate through each possible movement up, down, left, right
        for direction_x in [1, -1]:
            for direction_y in [1, -1]:
                for i in range(1, 9):
                    offset = (direction_x * i, direction_y * i)

                    # Check basic conditions on the given state and action (not OOB, not moving on friendly, etc.)
                    val = self.check_basic(state, offset)
                    if val < 0:
                        # Break for OOB and friendly
                        break
                    elif val == 0:
                        # Continue for if state is landing in check
                        continue

                    # New location
                    new_loc = new_location(self.location, offset)

                    piece = state.piece_at(new_loc)

                    # Add move to vacant location or opponent location. Also break on collision with opponent.
                    if piece is None:
                        moves.append(offset)
                    elif piece.colour is not self.colour:
                        moves.append(offset)
                        break

        self.move_list = moves
        return moves


class Queen(ChessPiece):
    """
    The queen piece that can do everything a rook and bishop can.

    """

    def __init__(self, location: Tuple[int, int], colour: bool):
        super().__init__(location, colour)
        self.value = c.PIECE_VALUES[c.QUEEN]
        self.icon = c.QUEEN

    def has_sight(self, state: GameState, location: Tuple[int, int]) -> bool:
        """
        Checks whether this piece has line of sight on the location. Line of sight means it could attack that location
        assuming it wasn't blocked by check rules from targeted king.

        :param state: the current state of the board
        :param location: the location we're checking for line of sight at
        :return: T/F depending on if there's line of sight
        """
        # Queen has line of sight only when either bishop or rook does
        return Bishop(self.location, self.colour).has_sight(state, location) or \
            Rook(self.location, self.colour).has_sight(state, location)

    def get_possible_moves(self, state: GameState) -> list:
        """
        Returns the list of possible moves for the queen given the current game state.

        :param state: the current game state
        :return: a list of possible moves for the pawn
        """
        # Bishop and Rook possible moves
        moves = Bishop(self.location, self.colour).get_possible_moves(state)
        moves.extend(Rook(self.location, self.colour).get_possible_moves(state))

        self.move_list = moves
        return moves


class King(ChessPiece):
    """
    The king piece that can move up to 1 in any direction so long as it doesn't move into check.

    """

    def __init__(self, location: Tuple[int, int], colour: bool):
        super().__init__(location, colour)
        self.value = c.PIECE_VALUES[c.KING]
        self.icon = c.KING
        self.allies = 0

    def has_sight(self, state: GameState, location: Tuple[int, int]) -> bool:
        """
        Checks whether this piece has line of sight on the location. Line of sight means it could attack that location
        assuming it wasn't blocked by check rules from targeted king.

        :param state: the current state of the board
        :param location: the location we're checking for line of sight at
        :return: T/F depending on if there's line of sight
        """
        # First check the offset actually matches a possible direction for a king
        actual_offset = (location[0] - self.location[0], location[1] - self.location[1])

        if actual_offset in list(it.product((-1, 0, 1), (-1, 0, 1))):
            return True

        return False

    def get_possible_moves(self, state: GameState) -> list:
        """
        Returns the list of possible moves for the king given the current game state.

        :param state: the current game state
        :return: a list of possible moves for the pawn
        """
        moves = []

        # Iterate through every cartesian product of (-1, 0, 1) of length 2 excluding (0, 0)
        self.allies = 0
        for offset in list(it.product((-1, 0, 1), (-1, 0, 1))):
            if offset == (0, 0):
                continue
            if self.check_basic(state, offset) == -1:
                self.allies += 1
            if self.check_basic(state, offset) <= 0:
                continue

            # Now, we know it isn't illegal so add it to the move list
            moves.append(offset)

        # Now check for legal castling
        # Left side
        offset = (-2, 0)
        # Test offset passes basic check and king can castle
        if self.castling_condition(state, offset) == 1 and self.moves_taken == 0 and (-1, 0) in moves:
            # Also check (-3, 0) doesn't have a piece (check is allowed for this square). Make sure king isn't in check
            if self.check_basic(state, (-3, 0)) >= 0 and not state.in_check(self.colour):
                # Finally, test if the rook can castle
                rook = state.piece_at(new_location(self.location, (-4, 0)))
                if rook is not None and rook.moves_taken == 0:
                    moves.append(offset)

        # Right side
        offset = (2, 0)
        # Test offset passes basic check and king can castle
        if self.castling_condition(state, offset) == 1 and self.moves_taken == 0 and (1, 0) in moves:
            # Make sure king isn't in check
            if not state.in_check(self.colour):
                # Finally, test if the rook can castle
                rook = state.piece_at(new_location(self.location, (3, 0)))
                if rook is not None and rook.moves_taken == 0:
                    moves.append(offset)

        self.move_list = moves
        return moves

    def castling_condition(self, state: GameState, offset: Tuple[int, int]) -> int:
        """
        Returns different values depending on why it fails.
            Passed castling condition:             1
            The location is in check               0
            There is a piece blocking movement:   -1

        :param state: the current state of the board
        :param offset: the move this piece is performing
        :return: different values depending on what type of failure or success
        """
        # New location
        new_loc = new_location(self.location, offset)

        piece = state.piece_at(new_loc)

        # If there's a piece in the new location, ignore it because the king is blocked
        if piece is not None:
            return -1

        # Check for illegal moves due to check
        next_state = state.get_next_state(self.location, offset)
        if next_state.in_check(self.colour):
            return 0

        return 1
