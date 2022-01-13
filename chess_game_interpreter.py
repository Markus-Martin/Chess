import chess.pgn
import constants as c
from chess_piece import Pawn, Knight, Bishop, Rook, Queen, King


class ChessGameInterpreter:
    """
    An interpreter for chess games in the form of pgn files. It can return move lists for each game so the main program
    can simulate it and extract information.
    """
    def __init__(self, name, elo_range):
        self.name = name
        self.elo_range = elo_range
        self.move_list = []
        self.move_number = 0
        self.file = open("game_database.pgn", encoding='utf-8')
        self.game = 0

    def next_game(self):
        """
        Updates the ChessGameInterpreter so the move list is for the next game. This should be called when the current
        game is finished being analysed and we need the next one.

        """
        # Reset moves
        self.move_number = 0
        self.move_list = []

        while self.game is not None:
            # Get the next game
            self.game = chess.pgn.read_game(self.file)

            # If we've reached the end of the list, just return and empty move list
            if self.game is None:
                return []

            # Get the board
            board = self.game.board()

            # Ignore games without the named player
            if self.name is not None:
                if self.game.headers["White"] != self.name or self.game.headers["Black"] != self.name:
                    continue

            # Also ignore games with average elo outside of the desired elo range
            elo_average = (int(self.game.headers.get("WhiteElo", 0)) + int(self.game.headers.get("BlackElo", 0))) / 2
            if self.elo_range[0] > elo_average or elo_average > self.elo_range[1]:
                continue

            for move in self.game.mainline_moves():
                # Get move in san notation
                move_lan = board.lan(move)

                # Adjust to the notation for the game and append to list
                self.move_list.append(self.lan_to_action(move_lan, board))
                board.push(move)

            # In case of game with no moves, skip it
            if len(self.move_list) == 0:
                continue

            # Set the winner of the game
            game_outcome = self.game.headers.get("Result", 0)
            if game_outcome == "*":
                c.GAME_OUTCOME = -1
            else:
                c.GAME_OUTCOME = int(game_outcome[0]) if len(game_outcome) <= 3 else -1

            return self.move_list

    @staticmethod
    def lan_to_action(move, board):
        """
        Converts the given san formatted move into a move that the program can read as (Piece, Offset).

        :param move: the san formatted move we're converting
        :param board: the state of the board
        :return: the converted move
        """
        # Initialise promotion as None
        promotion = None

        # Get piece colour
        colour = board.turn

        # Split between previous move and current move
        if 'x' in move:
            split_move = move.split('x')
            # For the second position ignore the piece we're taking if it's there. Ke1xNe2 (Ignore N here)
            if split_move[1][0] in c.PIECE_LIST:
                second_pos = split_move[1][1:]
            else:
                second_pos = split_move[1]
        elif '-' in move:
            split_move = move.split('-')
            second_pos = split_move[1]

        # Get the piece type we're moving
        icon = split_move[0][0]
        if icon == 'O':
            # Castling
            # Get king location
            king_loc = (5, 1) if colour else (5, 8)

            # Create King piece
            king = King(king_loc, colour)

            if len(split_move) == 2:
                # King side
                return king, (2, 0)
            else:
                # Queen side
                return king, (-2, 0)
        else:
            # For the first position ignore the piece we're taking if it's there. Ke1xNe2 (Ignore K here)
            if icon in c.PIECE_LIST:
                first_pos = split_move[0][1:]
            else:
                first_pos = split_move[0]

            # Moving piece
            if icon == c.KNIGHT:
                piece_type = Knight
            elif icon == c.BISHOP:
                piece_type = Bishop
            elif icon == c.ROOK:
                piece_type = Rook
            elif icon == c.QUEEN:
                piece_type = Queen
            elif icon == c.KING:
                piece_type = King
            else:
                piece_type = Pawn

                # Pawn promotion
                if "=" in split_move[1]:
                    _, promotion = split_move[1].split("=")

            # Convert first_pos and second_pos to tuple locations
            location1 = ord(first_pos[0]) - 96, int(first_pos[1])
            location2 = ord(second_pos[0]) - 96, int(second_pos[1])
            offset = location2[0] - location1[0], location2[1] - location1[1]

            # Create the piece
            piece = piece_type(location1, colour)
        if promotion is None:
            return piece, offset
        else:
            return piece, (*offset, promotion[0])

    def get_next_move(self, state, actions):
        """
        Returns the next move in the move_list given the number of moves that have already been performed

        :return: the next move in the move list
        """
        self.move_number += 1
        if self.move_number <= len(self.move_list):
            return self.move_list[self.move_number - 1]
        else:
            return None
