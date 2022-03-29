import argparse

import imageio
import pyautogui

from DDQN import *
from EnvManager import *
from FlappyBirdGym.FlappyBirdGym import *
from FlappyBirdGym.WindowMode import *


class dummy_context_mgr():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False


def checkMode(mode):
    
    if mode != "0" and mode != "1":
        raise argparse.ArgumentTypeError("Invalid mode option. Use \"0\" = game window or \"1\" = game window with plots")

    return mode

def main():

    # Set up ArgumentParser
    parser = argparse.ArgumentParser(description="An AI plays Flappy Bird.")
    parser.add_argument("--mode", help="Define the window mode (default: \"0\") \"0\" = game window or \"1\" = game window with plots", type=checkMode, required=False)
    parser.add_argument("--gif", help="File path where the GIF (screenshots of the window) will be saved.", required=False)

    args = parser.parse_args()

    window_mode = WindowMode.GAME_WINDOW
    if args.mode == "1":
        window_mode = WindowMode.GAME_WINDOW_PLOTS

    gif_path = ""
    if args.gif != None:
        gif_path = args.gif

    # Init env
    env = EnvMananger(window_mode=window_mode)

    # Load model
    q_net = DDDQN(num_actions=env.num_actions)

    q_net.build((1, *env.observation_space_shape))  # need a batch size
    q_net.load_weights("./saved_models/trainied_weights_epoch_810")

    q_net.summary()

    state = env.get_state()

    # Let the user time to move the mouse from the window
    if gif_path != "":
        time.sleep(3)

    with imageio.get_writer(gif_path, mode='I') if gif_path != "" else dummy_context_mgr() as gif_writer:
        while True:
            # Add batch dim
            state = np.expand_dims(state, axis=0)
            # Predict best action
            target, v, a, layer_activations = q_net(state, return_info=True)

            target = target[0]  # Remove batch dim
            best_action = np.argmax(target)

            # Execute best action
            state, reward, done = env.step(best_action)

            if done:
                env.reset()

            # Update window plot
            if window_mode == WindowMode.GAME_WINDOW_PLOTS:
                env.window.updatePlots(v, a, reward, layer_activations)

            # slow it down 
            if window_mode == WindowMode.GAME_WINDOW:
                time.sleep(0.05)

            if gif_path != "":
                img = get_window_image(env)
                gif_writer.append_data(img)


def get_window_image(env):
    canvas = env.gym.window
    x, y = canvas.winfo_rootx(), canvas.winfo_rooty()
    w, h = canvas.window_width, canvas.window_height  # canvas.winfo_width(), canvas.winfo_height()

    img = pyautogui.screenshot(region=(x, y, w, h))
    img = np.array(img, dtype=np.uint8)

    return np.array(img, dtype=np.uint8)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
