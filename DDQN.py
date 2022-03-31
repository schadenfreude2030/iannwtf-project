import tensorflow as tf


class DDDQN(tf.keras.Model):
    def __init__(self, num_actions: int):
        """Init the DDDQN. 

        Keyword arguments:
        num_actions -- Number of possible actions which can be taken in the gym.
        """
        super(DDDQN, self).__init__()

        self.front_end_layer_list = [
            tf.keras.layers.Dense(64, activation='tanh', name="tanh_0"),
            tf.keras.layers.Dense(128, activation='tanh', name="tanh_1"),
        ]

        self.v = tf.keras.layers.Dense(1, activation=None, name="state")
        self.a = tf.keras.layers.Dense(num_actions, activation=None, name="adventage")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_function = tf.keras.losses.MeanSquaredError()

    @tf.function
    def call(self, x: tf.Tensor, return_info=False):
        """Forward pass through the network. 

        Keyword arguments:
        x -- network input
        return_info -- shall be state, adventage and layer_activations also returned as a tupel?

        Return:
        return_info = False -> return only network output
        return_info = True -> return network output, state, adventage, layer_activations

        note that, layer_activations are return as a list
        """

        layer_activations = [x]
        for layer in self.front_end_layer_list:
            x = layer(x)
            layer_activations.append(x)

        v = self.v(x)
        a = self.a(x)
        q = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))

        if return_info:
            return q, v, a, layer_activations
        else:
            return q
    
    # @tf.function causes problems when plotting the model
    def call_onlyForPlotPurpose(self, x: tf.Tensor):
        """Forward pass through the network. 
        This function shall not be used for training.
        It is only used for plotting the model.

        Keyword arguments:
        x -- network input
        """

        for layer in self.front_end_layer_list:
            x = layer(x)
    
        v = self.v(x)
        a = self.a(x)

        tmp = tf.keras.layers.subtract([a, tf.math.reduce_mean(a, axis=1, keepdims=True)])
        result = tf.keras.layers.Add()([v, tmp])
        return result
        

    @tf.function
    def train_step(self, x: tf.Tensor, target: tf.Tensor):
        """Train the network based on input and target,

        Keyword arguments:
        x -- network input
        target -- target
        """
        
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.loss_function(target, predictions)  # + tf.reduce_sum(self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
