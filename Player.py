from DDQN import *
from EnvManager import *

def main():

    env = EnvMananger(windowMode=True)

    q_net = DDDQN(num_actions=2)
    q_net.build((32,*env.observation_space_shape))

    q_net.load_weights("./saved_models/trainied_weights_epoch_4100")

    state = env.reset()

    q_net.summary()

    while True:
        state = np.expand_dims(state, axis=0)
        target = q_net.predict(state)

        best_action = np.argmax(target, axis=1)[0]
        state, reward, done = env.step(best_action)

        if done:
            env.reset()
    
   

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")