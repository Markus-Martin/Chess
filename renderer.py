from __future__ import annotations
import constants as c
from game_state import GameState
import pygame
from typing import Tuple

"""
The visual side of chess. The Renderer will handle the graphical user interface the user sees.
"""


def grid2pixel(location: Tuple[int, int]) -> Tuple[int, int]:
    """
    Converts the grid value to a pixel location on the board.

    :param location: the grid position (x, y)
    :return: the pixel position (x2, y2)
    """
    # Convert grid to pixels
    x = (location[0] - 1) * c.CELL_SIZE
    y = (8 - location[1]) * c.CELL_SIZE + c.BANNER_HEIGHT

    return x, y


class Renderer:

    def __init__(self):
        """
        Initialises GUI.

        """
        self.size = 8
        self.banner_offset = 2

        # Create screen
        self.screen = pygame.display.set_mode((c.MAP_WIDTH, c.MAP_HEIGHT))

        # Title and program icon
        icon = pygame.image.load("images2/pawn_white.png")
        pygame.display.set_caption("Chess")
        pygame.display.set_icon(icon)

        # Create and pack the game grid

        # Create and pack the image map

    def get_bbox(self, location: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Returns the bounding box for the grid (x, y) position; this is a tuple containing
        information about the pixel positions of the edges of the shape, in the
        form (x min, y min, x max, y max). Returns None if position is out of bounds or invalid.

        Parameters:
            location (int, int): tuple containing the grid (x, y) position for the box

        Returns:
            (int, int, int, int): bounds on the box in pixel form
        """
        # Ensure input isn't anything invalid
        if not isinstance(location, tuple) or not ((0 <= location[0] < self.size + self.banner_offset and
                                                    0 <= location[1] < self.size + self.banner_offset)):
            print("Invalid input for bounding box")
            return None

        # Convert to closest bound box
        xmin = round(location[1] * c.CELL_SIZE)
        xmax = round((location[1] + 1) * c.CELL_SIZE)
        ymin = round(location[0] * c.CELL_SIZE)
        ymax = round((location[0] + 1) * c.CELL_SIZE)
        return xmin, ymin, xmax, ymax

    def draw_board(self):
        """
        Draws the chess board background pattern.

        """
        # Create the background
        for row in range(self.banner_offset, self.size + self.banner_offset):
            for col in range(0, self.size):
                location = row, col
                # Draw checker board pattern
                if sum(location) % 2 == 1:
                    pygame.draw.rect(self.screen, c.BLACK_COLOUR, pygame.Rect(*self.get_bbox(location)))
                else:
                    pygame.draw.rect(self.screen, c.WHITE_COLOUR, pygame.Rect(*self.get_bbox(location)))

    def draw_entity(self, location: Tuple[int, int], tile_type: str, colour: bool):
        """
        Draws the entity with tile type at the given position using a coloured rectangle with
        superimposed text identifying the entity.

        Parameters:
            location: location of the entity given as grid (x, y) coordinate
            tile_type: code for the type of entity
            colour: the colour of the given entity
        """
        colour_name = "_white" if colour else "_black"

        # Create the image and scale it
        full_name = c.IMAGES[tile_type] + colour_name + ".png"
        image = pygame.transform.scale(pygame.image.load(full_name), (c.PIECE_SIZE, c.PIECE_SIZE))

        # Draw the image
        self.screen.blit(image, grid2pixel(location))

    def draw_promotion(self, colour: bool):
        """
        Draws the graphic for promoting a pawn.

        :param colour: the colour of the pawn being promoted
        """
        colour_name = "_white" if colour else "_black"

        # Create the images
        queen_name = c.IMAGES[c.QUEEN] + colour_name + ".png"
        knight_name = c.IMAGES[c.KNIGHT] + colour_name + ".png"

        # Scale them
        queen_image = pygame.transform.scale(pygame.image.load(queen_name), (2 * c.PIECE_SIZE, 2 * c.PIECE_SIZE))
        knight_image = pygame.transform.scale(pygame.image.load(knight_name), (2 * c.PIECE_SIZE, 2 * c.PIECE_SIZE))

        # Place them on the board
        self.screen.blit(queen_image, grid2pixel((3, 5)))
        self.screen.blit(knight_image, grid2pixel((5, 5)))

    def draw(self, state: GameState):
        # Screen main background
        self.screen.fill((255, 255, 255))

        # Title banner
        banner = pygame.transform.scale(pygame.image.load("images2/banner.png"), (c.BANNER_WIDTH, c.BANNER_HEIGHT))
        self.screen.blit(banner, (0, 0))

        # Draw the board
        self.draw_board()

        # Draw board evaluation
        font = pygame.font.SysFont('Comic Sans MS', 24)
        text = font.render(str(round(state.evaluate_position(True), 2)), False, (0, 0, 0))
        self.screen.blit(text, (c.BANNER_WIDTH - 80, c.BANNER_HEIGHT - 50))

        # Draw every piece
        if len(state.pieces) > 0:
            for piece in state.pieces:
                self.draw_entity(piece.location, piece.icon, piece.colour)

        # Finally, draw the possible moves for the selected piece
        if state.selected_piece is not None:
            for offset in state.selected_piece.move_list:
                # Update offset to the correct value if it's 3 long
                if len(offset) == 3:
                    offset = (offset[0], offset[1])

                location = (state.selected_piece.location[0] + offset[0], state.selected_piece.location[1] + offset[1])
                # Plot transparent
                box = pygame.Surface((c.CELL_SIZE, c.CELL_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(box, (0, 255, 0, 120), box.get_rect())
                self.screen.blit(box, grid2pixel(location))

        # Draw pawn promotion if necessary
        if state.performing_promotion:
            self.draw_promotion(state.selected_piece.colour)

        pygame.display.update()
