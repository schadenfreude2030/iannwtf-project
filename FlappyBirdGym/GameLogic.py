import tkinter as tk
from FlappyBirdGym.Columns import *
 

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class GameLogic(tk.Frame):
    def __init__(self, windowMode = "none", master = None, height=300, width=450):
        
        self.windowMode = windowMode
        self.height = height
        self.width = width

        self.canvas = None
        if windowMode == "game":
            tk.Frame.__init__(self,master)
            master.geometry(f"{width}x{height}")
            self.canvas = tk.Canvas(master, height=self.height, width=self.width, bg='white')

            self.canvas.pack()
        
        elif windowMode == "stats":
            
            tk.Frame.__init__(self,master)
            master.geometry(f"{width + 1200}x{height+ 85}")
            self.canvas = tk.Canvas(master, height=self.height, width=self.width, bg='white')
            self.canvas.pack( side = LEFT)

            self.fig = Figure(figsize=(5,5), dpi=100)
            self.canvas_plot = FigureCanvasTkAgg(self.fig,master=master)
            self.canvas_plot.get_tk_widget().pack(side=RIGHT, fill=tk.BOTH, expand=True)

        self.first_column = None
        self.free_space_factor = 0.125

        self.initLogic()
        
        self.cnt = 0

 
        self.collectedRewards = []
        self.steps = []
        self.step_cnt = 0

    def updatePlots(self, v, a):

        self.fig.clf()

        #
        # Plot 1
        #
        stateAdventage_plt = self.fig.add_subplot(131)

        # remove batch dim
        v = v[0]
        a = a[0]
        y = np.concatenate((v, a))
        
        stateAdventage_barPlt = stateAdventage_plt.bar(["V(s)", "A(s, No jump)", "A(s, Jump)"], y, color="deepskyblue")
     
        idxMaxAdventage = np.argmax(a) + 1 # ignore state

        stateAdventage_barPlt[idxMaxAdventage].set_color("orange")
        stateAdventage_barPlt[idxMaxAdventage].set_label("Best action")

        stateAdventage_plt.set_title("State and adventage")

        stateAdventage_plt.set_ylim(-2,3)
        stateAdventage_plt.legend()
        stateAdventage_plt.grid(True)
        
        #
        # Plot 2
        #

        rewardsDistribution_plt = self.fig.add_subplot(132)
        x = np.arange(0, self.height + 1 ) # inclusive
        y = self.gaussianNormal(
                        x=x,
                        mu=self.first_column.middle_point, 
                        sigma=self.first_column.free_space*self.free_space_factor
                        )
        y = y / self.gaussianNormal(
                        x=self.first_column.middle_point, # <--- reach max
                        mu=self.first_column.middle_point, 
                        sigma=self.first_column.free_space*self.free_space_factor
                        )

        rewardsDistribution_plt.plot(x,y)
        rewardsDistribution_plt.set_title("Reward distribution of y positions")
        rewardsDistribution_plt.set_xlabel("y position")
        rewardsDistribution_plt.set_ylabel("Reward")
        rewardsDistribution_plt.grid(True)

        rewardsDistribution_plt.hlines(y=y[self.bird_posY0], xmin=0, xmax=self.bird_posY0, color='red')
        rewardsDistribution_plt.vlines(x=self.bird_posY0, ymin=0, ymax=y[self.bird_posY0], color="r")

        rewardsDistribution_plt.text(self.bird_posY0 + 5, 0.5, "Bird pos", transform=rewardsDistribution_plt.transData, rotation=90, verticalalignment='center', color="r")

        rewardsDistribution_plt.set_xlim(0, self.height)
        rewardsDistribution_plt.set_ylim(0, 1)

        #
        # Plot 3
        #

        collectedRewards_plt = self.fig.add_subplot(133)
       

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

        self.steps.append(self.step_cnt)
        self.collectedRewards.append(reward)

        collectedRewards_plt.plot(self.steps,self.collectedRewards)
        collectedRewards_plt.set_xlim(left=max(0, self.step_cnt - 50), right=self.step_cnt + 50)
        collectedRewards_plt.set_title("Obtained rewards")
        collectedRewards_plt.set_xlabel("Step")
        collectedRewards_plt.set_ylabel("Reward")
        collectedRewards_plt.grid(True)
        collectedRewards_plt.set_ylim(0, 1)
        self.step_cnt += 1
        
        self.canvas_plot.draw()
        

    def initLogic(self):

        self.bird_posX0 = 15
        self.bird_posX1 = 30

        self.bird_posY0 = int(self.height/2)
        self.bird_posY1 = int(self.height/2) + 15
        
        if self.windowMode != "none":
            self.canvas.create_rectangle(
                self.bird_posX0, self.bird_posY0, self.bird_posX1, self.bird_posY1,
                fill="red",
                outline='red', 
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

            if self.windowMode != "none":
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
      
        if self.windowMode != "none":
            self.canvas.move('bird', 0, delta_y)
        
        self.bird_posY0 += delta_y
        self.bird_posY1 += delta_y

        
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