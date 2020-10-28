import tensorflow as tf


class RBFLayer(tf.keras.layers.Layer):

    def __init__(self, units=32, name=None):
        super(RBFLayer, self).__init__(name=name)
        self.units = units
        self.c = None
        self.delta = None

    def build(self, input_shape):
        self.c = self.add_weight(shape=(self.units, input_shape[-1]), initializer='random_normal', trainable=True, name='c')
        self.delta = self.add_weight(shape=(1, self.units), initializer='random_normal', trainable=True, name='delta')

    def call(self, inputs, **kwargs):
        inputs = tf.expand_dims(inputs, axis=1)
        dis = tf.reduce_sum(tf.square(tf.subtract(tf.tile(inputs, [1, self.units, 1]), self.c)), axis=-1)
        delta_square = tf.square(self.delta)
        hidden_out = tf.exp(tf.multiply(-1., tf.divide(dis, tf.multiply(2., delta_square))))
        return hidden_out

    def get_config(self):
        config = {
            'units': self.units
        }
        base_config = super(RBFLayer, self).get_config()
        return base_config.update(config)