# tf_models/model.py
import tensorflow as tf
from .config import LEARNING_RATE

def make_ann(output_units):
    def f(neurons, input_dim):
        model = tf.keras.Sequential([
            # now specify input_dim rather than None
            tf.keras.layers.Dense(neurons, activation="relu", input_dim=input_dim),
            tf.keras.layers.Dense(neurons, activation="relu"),
            tf.keras.layers.Dense(output_units),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
            loss="mse",
            metrics=["mse"]
        )
        return model
    return f

# re-define your factories
create_sig_model    = make_ann(output_units=2)
create_source_model = make_ann(output_units=1)
