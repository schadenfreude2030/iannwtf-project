import tensorflow as tf


class DDDQN(tf.keras.Model):
    def __init__(self, num_actions):
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
    def call(self, x, return_info=False):

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
    def call_onlyForPlotPurpose(self, x):

        for layer in self.front_end_layer_list:
            x = layer(x)
    
        v = self.v(x)
        a = self.a(x)

        tmp = tf.keras.layers.subtract([a, tf.math.reduce_mean(a, axis=1, keepdims=True)])
        result = tf.keras.layers.Add()([v, tmp])
        return result
        

    @tf.function
    def train_step(self, x, target):
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.loss_function(target, predictions)  # + tf.reduce_sum(self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
