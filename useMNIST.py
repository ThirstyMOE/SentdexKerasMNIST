import tensorflow as tf
import numpy as np

def cross_compare_answers(predictions, validation_labels):
    correct = 0.0
    total = 0.0
    for index in range(len(predictions)):
        # numpy's argmax to take the onehot array and bring out the strongest fired output
        predict_digit = str(np.argmax(predictions[index]))
        real_digit = str(validation_labels[index])
        total += 1
        if predict_digit == real_digit:
            correct += 1
        print("Us: " + predict_digit + " -- Them: " + real_digit)
    print("You got an accuracy of: " + str(correct/total))


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = tf.keras.models.load_model("tmoe_mnist.model")

predictions = model.predict([x_test])

# Let's play a little game 1v1 me
cross_compare_answers(predictions, y_test)
