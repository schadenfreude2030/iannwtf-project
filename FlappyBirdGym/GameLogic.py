import tkinter as tk
from FlappyBirdGym.Columns import *

import numpy as np

class GameLogic(tk.Frame):
    def __init__(self, windowMode = False, master = None, height=300, width=450):
        
        self.windowMode = windowMode
        self.height = height
        self.width = width

        self.canvas = None
        if windowMode:
            tk.Frame.__init__(self,master)
            master.geometry(f"{width}x{height}")
            self.canvas = tk.Canvas(master, height=self.height, width=self.width, bg='black')
            self.canvas.pack()

        self.first_column = None

        self.initLogic()

        self.cnt = 0

    def initLogic(self):

        self.bird_posX0 = 15
        self.bird_posX1 = 30

        self.bird_posY0 = int(self.height/2)
        self.bird_posY1 = int(self.height/2) + 15

        if self.windowMode:
            self.canvas.create_rectangle(
                self.bird_posX0, self.bird_posY0, self.bird_posX1, self.bird_posY1,
                fill="white",
                outline='white', 
                tags=('bird')
                )

        self.column_width = 25

        self.columns = []
        for i in range(150, self.width + 50, 100):
            if len(self.columns) == 0:
                self.columns.append( Columns(windowMode=self.windowMode, canvas=self.canvas, posX=i, maxHeight=self.height, column_width=self.column_width, previousTopHeight=100) )
            else:
                self.columns.append( Columns(windowMode=self.windowMode, canvas=self.canvas, posX=i, maxHeight=self.height, column_width=self.column_width, previousTopHeight=self.columns[-1].getTopHeight()) )
        
        self.first_column = self.columns[0]

    def nextGameStep(self, action):
        
        for column in self.columns:
            column.move(-5,0)

            if self.windowMode:
                if column.getPosX() < -self.column_width:
                    column.delete()
           
            
        self.columns = [column for column in self.columns if not column.getPosX() < -self.column_width]
                
        if self.cnt == 20:
            self.columns.append( Columns(windowMode=self.windowMode, canvas=self.canvas, posX=450, maxHeight=self.height, column_width=self.column_width, previousTopHeight=self.columns[-1].getTopHeight()) )
            self.cnt = 0
        else:
            self.cnt += 1
            
        delta_y = 10
        if action == 1:
            delta_y = -5
      
        if self.windowMode:
            self.canvas.move('bird', 0, delta_y)
        
        self.bird_posY0 += delta_y
        self.bird_posY1 += delta_y

        free_space_factor = 0.125
        reward = self.gaussianNormal(
                        x=self.bird_posY0, 
                        mu=self.first_column.middle_point, 
                        sigma=self.first_column.free_space*free_space_factor
                        )
        # normalize between [0,1]
        reward = reward / self.gaussianNormal(
                        x=self.first_column.middle_point, # <--- reach max
                        mu=self.first_column.middle_point, 
                        sigma=self.first_column.free_space*free_space_factor
                        )

        killed = False
        # too low or too high
        if self.height <= self.bird_posY1 or self.bird_posY0 <= 0:
            killed = True
            reward = -1
    
        else:
            # check for collisions
            for column in self.columns:
                
                if column.getPosX() + column.column_width <= self.bird_posX0 and not column.wasBirdFlownOver():
                    column.setBirdFlownOver(True)

                    self.first_column = self.columns[1]
                    #reward = 10

                bird_pos = self.bird_posX0, self.bird_posY0, self.bird_posX1, self.bird_posY1
                if column.touched(bird_pos):
                    killed = True
                    reward = -1
                    break 
        
        return killed, reward
    
    def getState(self):
        return np.array([self.bird_posX0, self.bird_posY0, \
                         self.bird_posX1, self.bird_posY1, \
                         self.first_column.top_pos_x0, self.first_column.top_pos_y0, \
                         self.first_column.top_pos_x1, self.first_column.top_pos_y1,\
                         self.first_column.down_pos_x0, self.first_column.down_pos_y0, \
                         self.first_column.down_pos_x1, self.first_column.down_pos_y1])
    
    def reset(self):
        if self.windowMode:
            for column in self.columns:
                column.delete()

            self.canvas.delete("bird")
        self.cnt = 0
        self.columns = []
      
        self.initLogic()
    
    # def quit(self):
    #     self.master.destroy()

    
    def gaussianNormal(self,x, mu=0, sigma=1):
        return (1/ (sigma*np.sqrt(2*np.pi)) )* np.exp((-1/2)* ((x-mu)/sigma)**2)