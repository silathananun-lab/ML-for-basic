import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


def create_model(input_size, config="A"):

    model = Sequential()

    # Input Layer
    model.add(
        Dense(
            128,
            activation="relu",
            input_shape=(input_size,)
        )
    )

    if config == "A":

        model.add(Dense(64, activation="relu"))

    elif config == "B":

        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))

    elif config == "C":

        model.add(Dense(512, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))

    # increase regularization to reduce overfitting
    model.add(Dropout(0.2))

    # Output for Regression
    model.add(Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mae",
        metrics=["mae", "mse"]
    )

    return model