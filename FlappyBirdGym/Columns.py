import random

from tkinter import *
from PIL import ImageTk, Image


class Columns:
    id = 0
    def __init__(self, windowMode, canvas, posX, maxHeight, column_width, previousTopHeight):
        
        self.windowMode = windowMode
        self.canvas = canvas
  

        self.id = Columns.id

        self.column_width = column_width

        self.birdFlownOver = False

        self.top_height = random.randint(max(previousTopHeight - 70, 100), min(previousTopHeight + 70,200))
        self.free_space = random.randint(80, 100)


        self.middle_point = self.top_height + self.free_space/2


        self.top_pos_x0 = posX
        self.top_pos_x1 = posX + self.column_width

        self.top_pos_y0 = 0
        self.top_pos_y1 = self.top_height

        self.tag_column_top = f"square_top_{self.id}"
        

        self.down_pos_x0 = posX
        self.down_pos_x1 = posX + self.column_width

        self.down_pos_y0 = self.top_height + self.free_space
        self.down_pos_y1 = maxHeight

        self.tag_column_down = f"square_down_{self.id}"

        if self.windowMode != "none":
            
            # self.img_top = Image.open("./FlappyBirdGym/pipe-green_top.png")
            
            # self.img_top = self.img_top.resize((self.top_pos_x1 - self.top_pos_x0, self.top_pos_y1 - self.top_pos_y0), Image.ANTIALIAS)
           
            # self.img_top = ImageTk.PhotoImage(self.img_top)

            # self.column_top = self.canvas.create_image((self.top_pos_x0, self.top_pos_y0), image = self.img_top, tags=(self.tag_column_top))

            self.column_top = self.canvas.create_rectangle(
                                self.top_pos_x0, self.top_pos_y0,  self.top_pos_x1, self.top_pos_y1,
                                fill='green',
                                outline='green',
                                tags=(self.tag_column_top)
                                )

            self.column_down = self.canvas.create_rectangle(
                                self.down_pos_x0, self.down_pos_y0, self.down_pos_x1, self.down_pos_y1,
                                fill='green',
                                outline='green',
                                tags=(self.tag_column_down)
                                )

        Columns.id += 1
    
    def getPosX(self):
        return self.down_pos_x0 #self.posX

    def move(self, dx, dy):

        self.top_pos_x0 += dx
        self.top_pos_x1 += dx
        self.top_pos_y0 += dy
        self.top_pos_y1 += dy
        
        self.down_pos_x0 += dx
        self.down_pos_x1 += dx
        self.down_pos_y0 += dy
        self.down_pos_y1 += dy

        if self.windowMode != "none":
            self.canvas.move(self.tag_column_top, dx, dy)
            self.canvas.move(self.tag_column_down, dx, dy)
    

    def delete(self):
        if self.windowMode != "none":
            self.canvas.delete(self.column_top)
            self.canvas.delete(self.column_down)

    def touched(self, pos_bird):

        def touchedSingleColumn(pos_bird, pos_column):
            bird_x1, bird_y1, bird_x2, bird_y2 = pos_bird
            square_x1, square_y1, square_x2, square_y2 = pos_column

            left_side = ( (square_y1 <= bird_y1 and bird_y1 <= square_y2) or 
                          (square_y1 <= bird_y2 and bird_y2 <= square_y2) ) and square_x1 <= bird_x2 and bird_x2 <= square_x2

            top_side =  ( (square_x1 <= bird_x1 and bird_x1 <= square_x2) or 
                          (square_x1 <= bird_x2 and bird_x2 <= square_x2) ) and square_y1 <= bird_y2 and bird_y2 <= square_y2

            right_side = ( (square_y1 <= bird_y1 and bird_y1 <= square_y2) or 
                           (square_y1 <= bird_y2 and bird_y2 <= square_y2) ) and bird_x1 <= square_x2 and square_x1 <= bird_x1

            down_side = ( (square_x1 <= bird_x1 and bird_x1 <= square_x2) or 
                          (square_x1 <= bird_x2 and bird_x2 <= square_x2) ) and bird_y1 <= square_y2 and square_y1 <= bird_y1
    
            return left_side or top_side or right_side or down_side


        pos_column_top = self.top_pos_x0, self.top_pos_y0, self.top_pos_x1, self.top_pos_y1 #self.canvas.coords(self.tag_column_top)
        
        pos_column_down = self.down_pos_x0, self.down_pos_y0, self.down_pos_x1, self.down_pos_y1 #self.canvas.coords(self.tag_column_down)

        return touchedSingleColumn(pos_bird, pos_column_top) or touchedSingleColumn(pos_bird, pos_column_down)

    def getTopHeight(self):
        return self.top_height
    
    def wasBirdFlownOver(self):
        return self.birdFlownOver
    
    def setBirdFlownOver(self, birdFlownOver):
        self.birdFlownOver = birdFlownOver
        