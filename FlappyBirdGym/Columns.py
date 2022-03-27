import random

from tkinter import *
from PIL import ImageTk, Image

from FlappyBirdGym.WindowMode import *


class Columns:
    id = 0

    def __init__(self, window_mode, canvas, pos_x, max_height, column_width, previous_top_height):

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
        return self.down_pos_x0  # self.posX

    def move(self, dx, dy):

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
        if self.window_mode != WindowMode.NO_WINDOW:
            self.canvas.delete(self.column_top)
            self.canvas.delete(self.column_down)

    def touched(self, pos_bird):

        def touched_single_column(pos_bird, pos_column):
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
        return self.top_height

    def was_bird_flown_over(self):
        return self.bird_flown_over

    def set_bird_flown_over(self, bird_flown_over):
        self.bird_flown_over = bird_flown_over
