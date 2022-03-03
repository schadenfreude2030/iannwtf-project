import tkinter as tk
from FlappyBirdGym.Columns import *
from FlappyBirdGym.FlappyBirdGym import *
from FlappyBirdGym.WindowMode import * 

import numpy as np


class GameLogic:
    def __init__(self, windowMode, window = None, height=300, width=450):
        
        self.windowMode = windowMode
        self.height = height
        self.width = width
        self.window = window
        
        self.first_column = None
        self.free_space_factor = 0.125

        self.initLogic()
        

    def initLogic(self):
        
        # Bird (rectangle) positions 
        self.bird_posX0 = 15
        self.bird_posX1 = 30
        self.bird_posY0 = int(self.height/2)
        self.bird_posY1 = int(self.height/2) + 15
        
        # Draw it if window is used
        if self.windowMode != WindowMode.NO_WINDOW:
            self.window.canvas.create_rectangle(
                self.bird_posX0, self.bird_posY0, self.bird_posX1, self.bird_posY1,
                fill="red",
                outline='red', 
                tags=('bird')
                )

        # Create logically columns
        self.column_width = 25
        self.columns = []
        for i in range(150, self.width + 50, 100):
            if len(self.columns) == 0:
                self.columns.append( Columns(windowMode=self.windowMode, canvas=self.window.canvas, posX=i, maxHeight=self.height, column_width=self.column_width, previousTopHeight=100) )
            else:
                self.columns.append( Columns(windowMode=self.windowMode, canvas=self.window.canvas, posX=i, maxHeight=self.height, column_width=self.column_width, previousTopHeight=self.columns[-1].getTopHeight()) )
        
        self.first_column = self.columns[0]

        self.add_new_column_cnt = 0

    def nextGameStep(self, action):
        
        #
        # Update columns 
        # 

        # Move columns to left
        for column in self.columns:
            column.move(-5,0)

            # remove if out of screen (only if window is used)
            if self.windowMode != WindowMode.NO_WINDOW:
                if column.getPosX() < -self.column_width:
                    column.delete()
           
        # Logically remove columns
        self.columns = [column for column in self.columns if not column.getPosX() < -self.column_width]
        
        # Add a columns to the right (a columns was too left -> remove it -> add a new column to the right)
        if self.add_new_column_cnt == 20: # threshold reached
            self.columns.append( Columns(windowMode=self.windowMode, canvas=self.window.canvas, posX=450, maxHeight=self.height, column_width=self.column_width, previousTopHeight=self.columns[-1].getTopHeight()) )
            self.add_new_column_cnt = 0
        else:
            self.add_new_column_cnt += 1
        
        #
        # Update bird 
        # 

        # y position update of bird
        delta_y = 10 # default: go down
        if action == 1:
            delta_y = -5 # JUMP!

        # window: also move rectangle
        if self.windowMode != WindowMode.NO_WINDOW:
            self.window.canvas.move('bird', 0, delta_y)
        
        self.bird_posY0 += delta_y
        self.bird_posY1 += delta_y

        #
        # Reward
        # 
        
        # ... is gaussian distributed
        reward = self.gaussianNormal(
                        x=self.bird_posY0, 
                        mu=self.first_column.middle_point, 
                        sigma=self.first_column.free_space*self.free_space_factor
                        )
        # normalize between [0,1]
        reward = reward / self.gaussianNormal(
                        x=self.first_column.middle_point, # <--- reach max
                        mu=self.first_column.middle_point, 
                        sigma=self.first_column.free_space*self.free_space_factor
                        )

        # ... except if bird hit a column or ...
        killed = False

        # ... or too low or too high
        if self.height <= self.bird_posY1 or self.bird_posY0 <= 0:
            killed = True
            reward = -1
    
        else:
            # check for collisions
            for column in self.columns:
                
                if column.getPosX() + column.column_width <= self.bird_posX0 and not column.wasBirdFlownOver():
                    column.setBirdFlownOver(True)
                    self.first_column = self.columns[1]
    

                bird_pos = self.bird_posX0, self.bird_posY0, self.bird_posX1, self.bird_posY1
                if column.touched(bird_pos):
                    killed = True
                    reward = -1
                    break 
        
        return killed, reward
    
    def getState(self):
        # note: top and down column is a rectangle which can be described by four points (x0, x1, y0, y1)
        #       this also applied to the bird

        # Game state = points of bird, points of top column, points of down column  
        return np.array([self.bird_posX0, self.bird_posY0, \
                         self.bird_posX1, self.bird_posY1, \
                         self.first_column.top_pos_x0, self.first_column.top_pos_y0, \
                         self.first_column.top_pos_x1, self.first_column.top_pos_y1,\
                         self.first_column.down_pos_x0, self.first_column.down_pos_y0, \
                         self.first_column.down_pos_x1, self.first_column.down_pos_y1])
    
    def reset(self):
        # Remove columns and bird
        if self.windowMode != WindowMode.NO_WINDOW:
            for column in self.columns:
                column.delete()

            self.window.canvas.delete("bird")

        # Reset logic
        self.initLogic()
    
    
    def gaussianNormal(self,x, mu=0, sigma=1):
        return (1/ (sigma*np.sqrt(2*np.pi)) )* np.exp((-1/2)* ((x-mu)/sigma)**2)