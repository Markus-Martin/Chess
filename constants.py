"""
Stores various constants that can be adjusted for a different look/performance of the game.

"""
# Rates of RL progress is about 2.5 million state-action pairs per hour. And total states to explore:
# Param1 * Param2 * ... * Action Space * Accuracy (Larger accuracy -> better q values)

# Options
PLAY_MODE = 0  # 0 - PvAI, 1 - PvP, 2 - AIvAI
PLAYER_STARTS = True  # The player's colour
RENDER_FOR_AI = False  # Whether to render the game for AIvAI mode
AI_MOVE_DELAY = 0  # Movement delay for the AI actions in seconds
AI_MODE = 1  # 0 - Random Actions, 1 - QLearn (learning), 2 - QLearn (playing)
ALLOCATED_RUN_TIME = 1 * 60 * 60  # How long the program is allowed to run for (0 seconds means only do 1 episode)
SHOW_STATS = True  # Show statistics like game length as func of episodes
Q_NAME = "q_table"  # Name of the file that the q table is saved into
STATE_SIZE = 10**6

# AI State space representation:
# 0 - Brute force method that doesn't attempt to capture any features in particular - just converts FEN to number
# 1 - Just represented by counting the number of squares that are being controlled
STATE_TYPE = 0

# Reward options
LOSE_PENALTY = -1000
WIN_REWARD = 1000
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
