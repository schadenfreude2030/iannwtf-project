import pyautogui
import imageio
import cv2
import time

from DDQN import *
from EnvManager import *
from FlappyBirdGym.FlappyBirdGym import *
from FlappyBirdGym.WindowMode import * 

def main():

    env = EnvMananger(windowMode=WindowMode.GAME_WINDOW_PLOTS)
    
    # Load model
    q_net = DDDQN(num_actions=env.num_actions)
    q_net.build((1,*env.observation_space_shape)) # need a batch size
    q_net.load_weights("./saved_models/trainied_weights_epoch_4100")
    
    q_net.summary()

    state = env.getState()

    with imageio.get_writer('test.gif', mode='I') as writer:
        while True:
            # Add batch dim
            state = np.expand_dims(state, axis=0)
            # Predict best action
            target, v, a, layerActivations  = q_net(state)
        
            target = target[0] # Remove batch dim
            best_action = np.argmax(target)
            
            # Execute best action
            state, reward, done = env.step(best_action)

            # Update window plot
            env.window.updatePlots(v, a, reward, layerActivations)

            if done:
                env.reset()
            
            #time.sleep(0.1)
            #writer.append_data(getWindowImage(env))

def getWindowImage(env):
    
    canvas = env.env.gameLogic.canvas
    x, y = canvas.winfo_rootx(), canvas.winfo_rooty()
    w, h = canvas.winfo_width(), canvas.winfo_height()
        
    img = pyautogui.screenshot(region=(x, y, w, h))
    img = np.array(img, dtype=np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (img.shape[1]//3, img.shape[0]//3)) 

    return np.array(img, dtype=np.uint8)
    
   

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")