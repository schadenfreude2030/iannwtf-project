import tensorflow as tf

class DDDQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DDDQN, self).__init__()

        self.front_end_layer_list = [
            tf.keras.layers.Conv2D(8, kernel_size=(3, 3), strides=2, padding='valid', activation='relu'),
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=2, padding='valid', activation='relu'),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=2, padding='valid', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(64, activation='tanh'),
            # tf.keras.layers.Dense(128, activation='tanh'),
            
        ]

        self.v = tf.keras.layers.Dense(1, activation=None)
        self.a = tf.keras.layers.Dense(num_actions, activation=None)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_function = tf.keras.losses.MeanSquaredError()
    
    @tf.function
    def call(self, x):
        
        for layer in self.front_end_layer_list:
            x = layer(x)

        v = self.v(x)
        a = self.a(x)
        q = v +(a -tf.math.reduce_mean(a, axis=1, keepdims=True))
        return q

    @tf.function
    def train_step(self, input_seq, target_token):
        with tf.GradientTape() as tape:
            predictions = self(input_seq, training=True)
            loss = self.loss_function(target_token, predictions) #+ tf.reduce_sum(self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))