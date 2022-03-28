# Play Flappy Bird by applying Dueling Double Deep Q Learning 
## by Alaa Adam, Tobias Kretschel and Tim Niklas Witte 

This repository contains an AI which is able to play Flappy Bird.
The AI is based on Dueling Double Deep Q Learning.
In order to train the AI, a Flappy Bird gym was developed which is also contained in this repository (see `./FlappyBirdGym`).
Its usage will be explained at the end of the README.
It supports both a window and a no window mode.

![Alt Text](./media/game_window.gif)

The AI was pretrainied for 810 episodes.
The weights are saved in `./saved_model`. 
Besides, `PlayGame.py` enables to see the AI in action (see "Game Window: See results").

## Reward System

The reward is gaussian distributed where x is `bird_posY0`.

![Alt Text](./media/formular.png)

µ denotes the y coordinate of the center of space between the two vertical columns. 
σ denotes the length in y direction of the space between the two vertical columns.
Note that, the two vertical columns are considers which are nearest to the bird.

![Alt Text](./media/rewardSystem.png)

## Model

### Input 
The artifical neural network receives the previous and last game state as its input.
Both game states are concatenated.

```python
[previous_game_state, current_game_state]
```

A game state is described as the coordinates of the bird and the so called "first column"
which is nearest column to the bird.
Both the bird and the column are rectangles.
Note that, in order to represent a rectangle there are four points aka coordinates
necessary: `x0, y0, x1, y1`

```python
bird_pos_x0, bird_pos_y0,
bird_pos_x1, bird_pos_y1,
first_column.top_pos_x0, first_column.top_pos_y0,
first_column.top_pos_x1, first_column.top_pos_y1,
first_column.down_pos_x0, first_column.down_pos_y0,
first_column.down_pos_x1, first_column.down_pos_y1
```

At the beginning, the `previous_game_state` is a zero vector.

### Artificial Neural Network

The model has the following architecture:
```bash
Model: "DDQN"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 [Previous game state, current   [(None, 24)]        0           []                               
 game state] (InputLayer)                                                                         
                                                                                                  
 tanh_0 (Dense)                 (None, 64)           1600        ['[Previous game state, current g
                                                                 ame state][0][0]']               
                                                                                                  
 tanh_1 (Dense)                 (None, 128)          8320        ['tanh_0[0][0]']                 
                                                                                                  
 adventage (Dense)              (None, 2)            258         ['tanh_1[0][0]']                 
                                                                                                  
 tf.math.reduce_mean (TFOpLambd  (None, 1)           0           ['adventage[0][0]']              
 a)                                                                                               
                                                                                                  
 state (Dense)                  (None, 1)            129         ['tanh_1[0][0]']                 
                                                                                                  
 subtract (Subtract)            (None, 2)            0           ['adventage[0][0]',              
                                                                  'tf.math.reduce_mean[0][0]']    
                                                                                                  
 add (Add)                      (None, 2)            0           ['state[0][0]',                  
                                                                  'subtract[0][0]']               
                                                                                                  
==================================================================================================
Total params: 10,307
Trainable params: 10,307
Non-trainable params: 0
__________________________________________________________________________________________________

```

![Alt Text](./media/modelPlot.png)

## Pre-trained

As mentioned above, the model was trained for 810 episodes.

![Alt Text](./media/trainingPlot_noSmoothing.png)



## Evaluation

In order to demonstrate the models performance, it performs 250 steps.

![Alt Text](./media/performancePlot.png)

Overall, it archived an average reward of about 0.8.
It never occured a collison with a column aka pipe.

## Usage

### Start training

Run `Training.py`.

```bash
python3 Training.py
```

Each 10th episodes the model weights are stored in `./saved_model/trainied_weights_epoch_X`.
X denotes the number of the episode.
Besides, corresponding TensorBoard files are saved in `./test_logs/`

### Game Window: See results  

### Without plots

![Alt Text](./media/game_window.gif)

Run the `PlayGame.py`.

```bash
python3 PlayGame.py
```

### With plots

![Alt Text](./media/game_window_plots.gif)

Run the `PlayGame.py`.

```bash
python3 PlayGame.py
```

### Flappy Bird Gym