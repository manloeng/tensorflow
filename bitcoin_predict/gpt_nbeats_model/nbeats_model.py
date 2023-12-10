# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
#
# class Block(tf.keras.layers.Layer):
#     def __init__(self, units, theta_dim, backcast_length, forecast_length, name='block', **kwargs):
#         super(Block, self).__init__(name=name, **kwargs)
#         self.units = units
#         self.theta_dim = theta_dim
#         self.backcast_length = backcast_length
#         self.forecast_length = forecast_length
#         self.hidden1 = Dense(units, activation='relu', name='hidden1')
#         self.hidden2 = Dense(units, activation='relu', name='hidden2')
#         self.theta_layer = Dense(theta_dim, activation='linear', name='theta')
#
#     def build(self, input_shape):
#         super(Block, self).build(input_shape)
#
#     def call(self, x):
#         x = self.hidden1(x)
#         x = self.hidden2(x)
#         theta = self.theta_layer(x)
#
#         # Reshape theta to match the desired split dimensions
#         theta = tf.reshape(theta, [-1, self.forecast_length + self.backcast_length])
#
#         trend, seasonality = tf.split(theta, [self.forecast_length, self.backcast_length], axis=-1)
#
#         # Concatenate along the correct axis
#         x = Concatenate(axis=-1)([x, trend, seasonality])
#
#         return x
#
# class NBEATS(tf.keras.Model):
#     def __init__(self, input_size, stack_types, stack_units, horizon, name='nbeats', **kwargs):
#         super(NBEATS, self).__init__(name=name, **kwargs)
#         self.input_size = input_size
#         self.stack_types = stack_types
#         self.stack_units = stack_units
#         self.horizon = horizon
#         self.blocks = []
#
#         for i in range(len(stack_types)):
#             block = Block(units=stack_units[i],
#                           theta_dim=2 * horizon,
#                           backcast_length=input_size,
#                           forecast_length=horizon,
#                           name=f'block_{i}')
#             self.blocks.append(block)
#
#     def build(self, input_shape):
#         # Create input layer
#         input_layer = Input(shape=input_shape[1:])
#         # Call the blocks to build their internal layers
#         x = input_layer
#         for block in self.blocks:
#             x = block(x)
#         # Create the model using the input and output
#         super(NBEATS, self).build(input_layer)
#
#     def call(self, x):
#         # Forward pass through the blocks
#         for block in self.blocks:
#             x = block(x)
#         return x
#
# # Example usage:
# input_size = 10  # Adjust this based on the input size of your time series
# stack_types = ['generic', 'generic']  # You can customize the stack types
# stack_units = [256, 256]  # You can customize the number of units in each stack
# horizon = 3  # Forecast horizon
#
# # Create N-BEATS model
# model = NBEATS(input_size, stack_types, stack_units, horizon)
#
# # Display model summary
# model.build(input_shape=(128, input_size))
# model.summary()
