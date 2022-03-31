import random

import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image

from FlappyBirdGym.WindowMode import *


class Columns:
    id = 0

    def __init__(self, window_mode: int, canvas: tk.Canvas, pos_x: int, max_height: int, column_width: int, previous_top_height: int):

        """Init Columns. It consists of two columns which are vertically ordered. 
        In other words, one colmn is placed on top of the other one.

        if 1 <= window_mode -> they are down on the canvas 

        Keyword arguments:
        window_mode -- window_mode = 0 -> no window, window_mode = 1 -> game window, window_mode = 2 -> game window with plots
        canvas -- window object on which the Columns are drawn
        pos_x -- x position of the columns
        max_height -- max height of the column
        column_width -- width of the columns
        previous_top_height -- height of the previous column (in x direction)
        """

        self.window_mode = window_mode
        self.canvas = canvas

        self.id = Columns.id

        self.column_width = column_width

        self.bird_flown_over = False

        self.top_height = random.randint(max(previous_top_height - 70, 100), min(previous_top_height + 70, 200))
        self.free_space = random.randint(80, 100)

        self.middle_point = self.top_height + self.free_space / 2

        self.top_pos_x0 = pos_x
        self.top_pos_x1 = pos_x + self.column_width

        self.top_pos_y0 = 0
        self.top_pos_y1 = self.top_height

        self.tag_column_top = f"square_top_{self.id}"

        self.down_pos_x0 = pos_x
        self.down_pos_x1 = pos_x + self.column_width

        self.down_pos_y0 = self.top_height + self.free_space
        self.down_pos_y1 = max_height

        self.tag_column_down = f"square_down_{self.id}"

        if self.window_mode != WindowMode.NO_WINDOW:
            self.column_top = self.canvas.create_rectangle(
                self.top_pos_x0, self.top_pos_y0, self.top_pos_x1, self.top_pos_y1,
                fill='green',
                outline='green',
                tags=self.tag_column_top
            )

            self.column_down = self.canvas.create_rectangle(
                self.down_pos_x0, self.down_pos_y0, self.down_pos_x1, self.down_pos_y1,
                fill='green',
                outline='green',
                tags=self.tag_column_down
            )

        Columns.id += 1

    def get_pos_x(self):
        """
        Return:
        x position of the columns (they are other vertically = one is placed on the other one with a 
        space between them -> same x position)
        """
        return self.down_pos_x0  # self.posX

    def move(self, dx: int, dy: int):
        """
        Changes x and y position of the two columns 

        Keyword arguments:
        dx -- Change the x position of the two columns by dx
        dy -- Change the y position of the two columns by dy
        """

        self.top_pos_x0 += dx
        self.top_pos_x1 += dx
        self.top_pos_y0 += dy
        self.top_pos_y1 += dy

        self.down_pos_x0 += dx
        self.down_pos_x1 += dx
        self.down_pos_y0 += dy
        self.down_pos_y1 += dy

        if self.window_mode != WindowMode.NO_WINDOW:
            self.canvas.move(self.tag_column_top, dx, dy)
            self.canvas.move(self.tag_column_down, dx, dy)

    def delete(self):
        """
        Delete the two columns
        """
        if self.window_mode != WindowMode.NO_WINDOW:
            self.canvas.delete(self.column_top)
            self.canvas.delete(self.column_down)

    def touched(self, pos_bird: tuple):

        """
        Given a brid position, it will be check if the bird touched one of the two columns

        Keyword arguments:
        pos_bird -- y position of the bird which consists of (bird_x1, bird_y1, bird_x2, bird_y2)
        """

        def touched_single_column(pos_bird: tuple, pos_column: tuple):
            """
            Given a brid position and a column position
            it will be check if the bird touches the column

            Keyword arguments:
            pos_bird -- y position of the bird which consists of (bird_x1, bird_y1, bird_x2, bird_y2)
            pos_column -- y position of the column which consists of (square_x1, square_y1, square_x2, square_y2)
            """

            bird_x1, bird_y1, bird_x2, bird_y2 = pos_bird
            square_x1, square_y1, square_x2, square_y2 = pos_column

            left_side = ((square_y1 <= bird_y1 <= square_y2) or
                         (square_y1 <= bird_y2 <= square_y2)) and square_x1 <= bird_x2 <= square_x2

            top_side = ((square_x1 <= bird_x1 <= square_x2) or
                        (square_x1 <= bird_x2 <= square_x2)) and square_y1 <= bird_y2 <= square_y2

            right_side = ((square_y1 <= bird_y1 <= square_y2) or
                          (square_y1 <= bird_y2 <= square_y2)) and square_x2 >= bird_x1 >= square_x1

            down_side = ((square_x1 <= bird_x1 <= square_x2) or
                         (square_x1 <= bird_x2 <= square_x2)) and square_y2 >= bird_y1 >= square_y1

            return left_side or top_side or right_side or down_side

        pos_column_top = self.top_pos_x0, self.top_pos_y0, self.top_pos_x1, self.top_pos_y1

        pos_column_down = self.down_pos_x0, self.down_pos_y0, self.down_pos_x1, self.down_pos_y1

        return touched_single_column(pos_bird, pos_column_top) or touched_single_column(pos_bird, pos_column_down)

    def get_top_height(self):
        """
        Return:
        Height of the column which is placed over the other column.
        """
        return self.top_height

    def was_bird_flown_over(self):
        """
        Return:
        If the bird has passed this obstacle.
        In other words, does the bird has flown between the space which is 
        between these two columns?
        """
        return self.bird_flown_over

    def set_bird_flown_over(self, bird_flown_over):
        """
        Return:
        Set the bool if the bird passed this obstacle.
        (see was_bird_flown_over)
        """
        self.bird_flown_over = bird_flown_over
