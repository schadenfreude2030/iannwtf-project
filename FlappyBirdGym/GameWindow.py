import tkinter as tk
from FlappyBirdGym.Columns import *

import numpy as np

class GameWindow(tk.Frame):
    def __init__(self, master, height=300, width=450):
        
        tk.Frame.__init__(self,master)

        self.height = height
        self.width = width

        master.geometry(f"{width}x{height}")

        self.canvas = tk.Canvas(master, height=self.height, width=self.width, bg='black')
        self.canvas.pack()

        self.first_column = None

        self.createWidgets()

        self.cnt = 0

        


    def createWidgets(self):

        self.bird_posX0 = 15
        self.bird_posX1 = 30

        self.bird_posY0 = int(self.height/2)
        self.bird_posY1 = int(self.height/2) + 15

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
                self.columns.append( Columns(self.canvas, posX=i, maxHeight=self.height, column_width=self.column_width, previousTopHeight=100) )
            else:
                self.columns.append( Columns(self.canvas, posX=i, maxHeight=self.height, column_width=self.column_width, previousTopHeight=self.columns[-1].getTopHeight()) )
        
        self.first_column = self.columns[0]

    def nextGameStep(self, action):
        
        for column in self.columns:
            column.move(-5,0)

            if column.getPosX() < -self.column_width:
                column.delete()
           
            
        self.columns = [column for column in self.columns if not column.getPosX() < -self.column_width]
                
        if self.cnt == 20:
            self.columns.append( Columns(self.canvas, 450, maxHeight=self.height, column_width=self.column_width, previousTopHeight=self.columns[-1].getTopHeight()) )
            self.cnt = 0
        else:
            self.cnt += 1
            
        delta_y = 10
        if action == 1:
            delta_y = -5
      
        self.canvas.move('bird', 0, delta_y)
        
        self.bird_posY0 += delta_y
        self.bird_posY1 += delta_y


        #distance = abs(self.bird_posY0 - self.first_column.middle_point)/20
        reward = self.gaussianNormal(
                        x=self.bird_posY0, 
                        mu=self.first_column.middle_point, 
                        sigma=self.first_column.free_space/8
                        )
        reward *= 200

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

                if column.touched():
                    killed = True
                    reward = -1
                    break 
        
        return killed, reward
    
    def resetGame(self):
        for column in self.columns:
            column.delete()

        self.cnt = 0
        self.columns = []
        self.canvas.delete("bird")

        self.createWidgets()
    
    def quit(self):
        self.master.destroy()

    
    def gaussianNormal(self,x, mu=0, sigma=1):
        return (1/ (sigma*np.sqrt(2*np.pi)) )* np.exp((-1/2)* ((x-mu)/sigma)**2)