import tensorflow as tf

class DDDQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DDDQN, self).__init__()

        self.front_end_layer_list = [
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(128, activation='tanh'),
        ]

        self.v = tf.keras.layers.Dense(1, activation=None)
        self.a = tf.keras.layers.Dense(num_actions, activation=None)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_function = tf.keras.losses.MeanSquaredError()
    
    @tf.function
    def call(self, x, returnInfo=False):
        
        layerActivations = [x]
        for layer in self.front_end_layer_list:
            x = layer(x)
            layerActivations.append(x)

        v = self.v(x)
        a = self.a(x)
        q = v +(a -tf.math.reduce_mean(a, axis=1, keepdims=True))
        
        if returnInfo:
            return q, v, a, layerActivations
        else:
            return q

    @tf.function
    def train_step(self, x, target):
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.loss_function(target, predictions) #+ tf.reduce_sum(self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))