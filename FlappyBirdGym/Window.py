import tkinter as tk
from tkinter import *
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from FlappyBirdGym.FlappyBirdGym import *
from FlappyBirdGym.WindowMode import *

matplotlib.use('TkAgg')


def gaussian_normal(x, mu=0, sigma=1):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)


class Window(tk.Frame):

    def __init__(self, window_mode, master=None, game_height=300, game_width=450):
        self.canvas = None
        self.gameLogic = None

        if window_mode == WindowMode.GAME_WINDOW:

            self.window_height = game_height
            self.window_width = game_width

            # Create window
            tk.Frame.__init__(self, master)
            master.geometry(f"{self.window_width}x{self.window_height}")
            self.canvas = tk.Canvas(master, height=self.window_height, width=self.window_width, bg='white')

            self.canvas.pack()

        elif window_mode == WindowMode.GAME_WINDOW_PLOTS:

            # Window contains also plots
            self.window_height = game_height + 550
            self.window_width = game_width + 1200

            # Create window
            tk.Frame.__init__(self, master)
            master.geometry(f"{self.window_width}x{self.window_height}")
            self.canvas = tk.Canvas(master, height=game_height, width=game_width, bg='white')
            self.canvas.pack(side=LEFT)

            # Plots
            self.fig = Figure(figsize=(5, 5), dpi=100)
            self.canvas_plot = FigureCanvasTkAgg(self.fig, master=master)
            self.canvas_plot.get_tk_widget().pack(side=RIGHT, fill=tk.BOTH, expand=True)

            self.collectedRewards = []
            self.steps = []
            self.step_cnt = 0

    def update_plots(self, v, a, reward, layer_activations):

        self.fig.clf()

        #
        # Plot 1: State and advantage
        #
        state_advantage_plt = self.fig.add_subplot(231)

        # remove batch dim
        v = v[0]
        a = a[0]
        y = np.concatenate((v, a))

        state_advantage_bar_plt = state_advantage_plt.bar(["V(s)", "A(s, No jump)", "A(s, Jump)"], y,
                                                          color="deepskyblue")

        idx_max_advantage = np.argmax(a) + 1  # ignore state

        state_advantage_bar_plt[idx_max_advantage].set_color("orange")
        state_advantage_bar_plt[idx_max_advantage].set_label("Best action")

        state_advantage_plt.set_title("State and adventage")
        state_advantage_plt.set_ylabel("Magnitude")

        state_advantage_plt.set_ylim(-1, 8)
        state_advantage_plt.legend()
        state_advantage_plt.grid(True)

        #
        # Plot 2: Reward distribution
        #

        rewards_distribution_plt = self.fig.add_subplot(232)
        x = np.arange(0, self.gameLogic.height + 1)  # inclusive
        y = gaussian_normal(
            x=x,
            mu=self.gameLogic.first_column.middle_point,
            sigma=self.gameLogic.first_column.free_space * self.gameLogic.free_space_factor
        )
        y = y / gaussian_normal(
            x=self.gameLogic.first_column.middle_point,  # <--- reach max
            mu=self.gameLogic.first_column.middle_point,
            sigma=self.gameLogic.first_column.free_space * self.gameLogic.free_space_factor
        )

        rewards_distribution_plt.plot(x, y)
        rewards_distribution_plt.set_title("Reward distribution of y positions")
        rewards_distribution_plt.set_xlabel("y position")
        rewards_distribution_plt.set_ylabel("Reward")
        rewards_distribution_plt.grid(True)

        rewards_distribution_plt.hlines(y=y[self.gameLogic.bird_pos_y0], xmin=0, xmax=self.gameLogic.bird_pos_y0,
                                        color='red')
        rewards_distribution_plt.vlines(x=self.gameLogic.bird_pos_y0, ymin=0, ymax=y[self.gameLogic.bird_pos_y0],
                                        color="r")

        rewards_distribution_plt.text(self.gameLogic.bird_pos_y0 + 5, 0.5, "Bird pos",
                                      transform=rewards_distribution_plt.transData, rotation=90,
                                      verticalalignment='center', color="r")

        rewards_distribution_plt.set_xlim(0, self.gameLogic.height)
        rewards_distribution_plt.set_ylim(0, 1)

        #
        # Plot 3: Collected rewards
        #

        collected_rewards_plt = self.fig.add_subplot(233)

        self.steps.append(self.step_cnt)
        self.collectedRewards.append(reward)

        collected_rewards_plt.plot(self.steps, self.collectedRewards)
        collected_rewards_plt.set_xlim(left=max(0, self.step_cnt - 50), right=self.step_cnt + 50)
        collected_rewards_plt.set_title("Obtained rewards")
        collected_rewards_plt.set_xlabel("Step")
        collected_rewards_plt.set_ylabel("Reward")
        collected_rewards_plt.grid(True)
        collected_rewards_plt.set_ylim(0, 1)
        self.step_cnt += 1

        #
        # Plot 4: Activation input layer (= inputs)
        #

        activation_input_plt = self.fig.add_subplot(234)

        activations = layer_activations[0]  # input layer
        activations = activations.numpy()[0]  # remove batch dim
        activations = np.reshape(activations, newshape=(4, 6))  # input shape: (24,) -> (4,6)

        activation_input_plt.set_title("Input")
        activation_input_plt.set_axis_off()

        img = activation_input_plt.imshow(activations)
        img.set_clim(0, 300)
        self.fig.colorbar(img)
        #
        # Plot 5: Activations hidden layer 1
        #

        activation_h1_plt = self.fig.add_subplot(235)

        activations = layer_activations[1]  # input layer
        activations = activations.numpy()[0]  # remove batch dim
        activations = np.reshape(activations, newshape=(8, 8))  # h1 shape: (64,) -> (8,8)

        activation_h1_plt.set_title("Activation hidden layer 1")
        activation_h1_plt.set_axis_off()
        img = activation_h1_plt.imshow(activations)
        self.fig.colorbar(img)
        #
        # Plot 6: Activations hidden layer 2
        #

        activation_h2_plt = self.fig.add_subplot(236)

        activations = layer_activations[2]  # input layer
        activations = activations.numpy()[0]  # remove batch dim
        activations = np.reshape(activations, newshape=(8, 16))  # h1 shape: (128,) -> (8,16)

        activation_h2_plt.set_title("Activation hidden layer 2")
        activation_h2_plt.set_axis_off()
        img = activation_h2_plt.imshow(activations)
        self.fig.colorbar(img)

        self.canvas_plot.draw()
