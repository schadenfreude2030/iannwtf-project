from DDQN import *
from EnvManager import *
import pyautogui

import imageio
import cv2

def main():
    # game
    # stats
    # none
    env = EnvMananger(windowMode="stats")
    
    q_net = DDDQN(num_actions=2)
    q_net.build((32,*env.observation_space_shape))

    q_net.load_weights("./saved_models/trainied_weights_epoch_4100")

    state = env.getState()

    q_net.summary()


    with imageio.get_writer('test.gif', mode='I') as writer:
        while True:
            state = np.expand_dims(state, axis=0)
            target, v, a  = q_net.predict(state)

            best_action = np.argmax(target, axis=1)[0]
            state, reward, done = env.step(best_action)

            env.env.window.updatePlots(v, a)

            if done:
                env.reset()
            
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