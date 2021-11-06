

# Artificial Intelligence - Mimic intelligence in a computer
    # Machine learning - System that can learn and adapt without following explicit algorithms with human-made features
        # Deep learning - extracts features automatically to learn from. Deep Neural networks


# Perceptron (Artificual Neuron)

# Take in a set of inputs, multiply each input by a weight, and add them together
# w_1x_1 + w_2x_2 + ... w_nx_n + w_0 --> Nonlinear activiation function --> output (y-hat`)


# Dense Layer:
# import tensorflow as tf
# layer = tf.keras.Dense(units=2)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(n),
    tf.keras.layers.Dense(2)
])