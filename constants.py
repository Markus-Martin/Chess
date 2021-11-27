"""
Stores various constants that can be adjusted for a different look/performance of the game.

"""

# Options
PLAY_MODE = 2  # 0 - PvAI, 1 - PvP, 2 - AIvAI
PLAYER_STARTS = True
RENDER_FOR_AI = True  # Whether to render the game for AIvAI mode
AI_MOVE_DELAY = 0  # Movement delay for the AI actions in seconds

# Piece labels
PAWN = "p"
KNIGHT = "N"
ROOK = "R"
BISHOP = "B"
QUEEN = "Q"
KING = "K"

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


DARK_PURPLE = ACCENT_COLOUR = '#371D33'
DARKEST_PURPLE = '#371D33'
LIGHT_BROWN = MAP_BACKGROUND_COLOUR = '#B5B28F'
LIGHT_GREEN = '#B8D58E'
LIGHT_PURPLE = '#E5E1EF'

KNIGHT_OFFSETS = [(1, 2), (2, 1), (-1, 2), (2, -1), (-2, 1), (1, -2), (-1, -2), (-2, -1)]