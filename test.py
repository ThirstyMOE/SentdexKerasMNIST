import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt  # can't seem to import tkinter dependency

def create_model_architecture():
    # Get the keras model. Feedforward model (Sequential)
    model = tf.keras.models.Sequential()

    # Use model.add() to add layers to the keras model
    # Flatten the data from 28 x 28 to 756 1 dim tensor
    model.add(tf.keras.layers.Flatten())
    # Add hidden layers of 128 neurons with relu activation
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    # Add output layer of 10 neurons with softmax activation (for probablistic distr.)
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    return model

# 28x28 images of the number digits
mnist = tf.keras.datasets.mnist

# Load in mnist data into training data and labels and test data and labels
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Use keras to put all input data between 0 and 1. Normalization
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = create_model_architecture()

# Choooose your dueling weapon. Optimizers, loss type, metrics for evaluation, regularization too I bet
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
# Train the model!
model.fit(x_train, y_train, epochs=3)

# Calculate the validation data
val_loss, val_acc = model.evaluate(x_test, y_test)
print("Validation loss: " + str(val_loss))
print("Validation accuracy: " + str(val_acc))

# Save your model weights and architecture! Checkpointing
model.save("tmoe_mnist.model")

# # Reload your trained model back into a keras model object
# new_model = tf.keras.models.load_model("tmoe_mnist.model")
# # only takes a [list], but have your model run through input data and spit out predictions list
# predictions = new_model.predict([x_test])
