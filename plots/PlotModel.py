import sys
sys.path.append("../")

from DDQN import *
from EnvManager import *
from FlappyBirdGym.FlappyBirdGym import *
from FlappyBirdGym.WindowMode import *

def main():

    env = EnvMananger(window_mode=WindowMode.NO_WINDOW)

    q_net = DDDQN(num_actions=env.num_actions)
    q_net.build((None, *env.observation_space_shape))

    x = tf.keras.Input(shape=(env.observation_space_shape), name="[Previous game state, current game state]")
    model = tf.keras.Model(inputs=[x], outputs=q_net.call_onlyForPlotPurpose(x))
    
    model.summary()
    tf.keras.utils.plot_model(model,show_shapes=True, show_layer_names=True, to_file="../media/modelPlot.png")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")