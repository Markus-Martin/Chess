"""
Stores various constants that can be adjusted for a different look/performance of the game.

"""
# Rates of RL progress is about 2.5 million state-action pairs per hour. And total states to explore:
# Param1 * Param2 * ... * Action Space * Accuracy (Larger accuracy -> better q values)

# Options
PLAY_MODE = 2  # 0 - PvAI, 1 - PvP, 2 - AIvAI
PLAYER_STARTS = True  # The player's colour
RENDER_FOR_AI = False  # Whether to render the game for AIvAI mode
AI_MOVE_DELAY = 0  # Movement delay for the AI actions in seconds
USE_NEURAL_NETWORK = True  # Whether to use the neural network (DRL instead of RL)
AI_MODE = 3  # 0 - Random Actions, 1 - QLearn (learning), 2 - QLearn (playing), 3 - QLearn from provided games
ALLOCATED_RUN_TIME = 10*60*60  # How long the program is allowed to run for (0 seconds means only do 1 episode)
SHOW_STATS = True  # Show statistics like game length as func of episodes
SAVE_NAME = "learned_info"  # Name of the file that the information for AI is saved into
STATE_SIZE = 10**6  # Size of the state space (larger is more accurate but slower to learn)
LEARN_FROM = None  # This can be set to the name of a grand master like "Carlsen, M" to learn exclusively
ELO_RANGE = (2000, 3000)  # Elo range for which we should learn from

# AI State space representation:
# 0 - Brute force method that doesn't attempt to capture any features in particular - just converts FEN to number
# 1 - Just represented by controlled blocks of squares
# 2 - Incorporates controlled squares, time frame (early, mid, late) and king safety
STATE_TYPE = 2

# Reward options
LOSE_PENALTY = -50
WIN_REWARD = 50
STALEMATE_PENALTY = 0
MOVE_COST = -1

# Piece labels
PAWN = "P"
KNIGHT = "N"
ROOK = "R"
BISHOP = "B"
QUEEN = "Q"
KING = "K"

PIECE_LIST = [PAWN, KNIGHT, ROOK, BISHOP, QUEEN, KING]

# Piece values
PIECE_VALUES = {PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 5, QUEEN: 9, KING: 0}

# Colours
DARK_BROWN = (80, 40, 0)
LIGHT_BROWN = (255, 190, 120)


# GUI elements
TITLE = "Chess"
SCALE = 1.5
BANNER_HEIGHT = round(100 * SCALE)
MAP_WIDTH = BANNER_WIDTH = round(400 * SCALE)
MAP_HEIGHT = MAP_WIDTH + BANNER_HEIGHT
CELL_SIZE = round(50 * SCALE)
PIECE_SIZE = round(CELL_SIZE)
BLACK_COLOUR = DARK_BROWN
WHITE_COLOUR = LIGHT_BROWN
IMAGES = {
    PAWN: "images2/pawn",
    KNIGHT: "images2/knight",
    ROOK: "images2/rook",
    BISHOP: "images2/bishop",
    QUEEN: "images2/queen",
    KING: "images2/king"
}
HEATMAP_FILE = {
    PAWN: "heatmaps/pawn_map.png",
    KNIGHT: "heatmaps/knight_map.png",
    ROOK: "heatmaps/rook_map.png",
    BISHOP: "heatmaps/bishop_map.png",
    QUEEN: "heatmaps/queen_map.png",
    KING: "heatmaps/king_map.png"
}

HEATMAPS = {
    PAWN: {},
    KNIGHT: {},
    ROOK: {},
    BISHOP: {},
    QUEEN: {},
    KING: {}
}


DARK_PURPLE = ACCENT_COLOUR = '#371D33'
DARKEST_PURPLE = '#371D33'
LIGHT_BROWN = MAP_BACKGROUND_COLOUR = '#B5B28F'
LIGHT_GREEN = '#B8D58E'
LIGHT_PURPLE = '#E5E1EF'

KNIGHT_OFFSETS = [(1, 2), (2, 1), (-1, 2), (2, -1), (-2, 1), (1, -2), (-1, -2), (-2, -1)]

# The blocks of squares that we define our state space with. They are 4x2 blocks of squares starting at bottom left.
CONTROL_SQUARES = [[(1, 1), (2, 1), (3, 1), (4, 1), (1, 2), (2, 2), (3, 2), (4, 2)],
                   [(5, 1), (6, 1), (7, 1), (8, 1), (5, 2), (6, 2), (7, 2), (8, 2)],
                   [(1, 3), (2, 3), (3, 3), (4, 3), (1, 4), (2, 4), (3, 4), (4, 4)],
                   [(5, 3), (6, 3), (7, 3), (8, 3), (5, 4), (6, 4), (7, 4), (8, 4)],
                   [(1, 5), (2, 5), (3, 5), (4, 5), (1, 6), (2, 6), (3, 6), (4, 6)],
                   [(5, 5), (6, 5), (7, 5), (8, 5), (5, 6), (6, 6), (7, 6), (8, 6)],
                   [(1, 7), (2, 7), (3, 7), (4, 7), (1, 8), (2, 8), (3, 8), (4, 8)],
                   [(5, 7), (6, 7), (7, 7), (8, 7), (5, 8), (6, 8), (7, 8), (8, 8)]]

GAME_OUTCOME = -1  # Holds the game outcome (white win 1, loss 0, stalemate -1) for each of the games when in AI Mode 3
