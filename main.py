import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import matplotlib.pyplot as plt

multi_task_mri_image_path = "MRI_Image_Matrix.multi.npy"  # The path of dataset - image
mri_image_matrix_multi_task = np.load(multi_task_mri_image_path)  # The operation to load the

X = np.delete(mri_image_matrix_multi_task, 262144, 1)  # The process to delete the last column - labels
Y = mri_image_matrix_multi_task[:, -1]  # The process to obtain the labels
# The process to split the data
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=0)
# The process to reshape the training set
x_train = x_train.reshape(x_train.shape[0], 512, 512, 1) / 255
# The process to reshape the validation set
x_valid = x_valid.reshape(x_valid.shape[0], 512, 512, 1) / 255
# The process to operate One-Hot Encoding for y_train
y_trainOneHot = np_utils.to_categorical(y_train)
# The process to operate One-Hot Encoding for y_valid
y_validOneHot = np_utils.to_categorical(y_valid)

# Construct the Sequential model
model = Sequential()
# Add the first convolution layer
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(512, 512, 1)))
# Add the first pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# Add the second convolution layer
model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='same', activation='relu'))
# Add the second pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# Add dropout function to avoid overfitting
model.add(Dropout(0.5))
# Change multi-dimensional array to one dimension
model.add(Flatten())
# Add fully-connected layer 1 with 128 neurons
model.add(Dense(128, activation='relu', kernel_initializer='normal'))
# Add dropout function to avoid overfitting
model.add(Dropout(0.25))
# Add fully-connected layer 2 as the output layer with 4 neurons
model.add(Dense(4, activation='softmax', kernel_initializer='normal'))  # Task B Multiclass Task
# model.add(Dense(2, activation='sigmoid'))  # Task A Binary Task
# Show model summary
print(model.summary())

# Define training methods:
# Parameters:
# loss function: Use categorical cross-entropy loss
# optimizer: Implement Adam optimizer to make training converge faster
# metrics: Set to evaluate model:

model.compile(loss=tf.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])  # Task B Multiclass task
# Task A Binary task model.compile(loss=tf.losses.binary_crossentropy,
#                                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#                                  metrics=['accuracy'])

# Count the amount of wall time
t0 = time.time()
# The process to start training
# Parameters:
# x_train: Feature value of digital image
# y_trainOneHot: True labeling of digital images
# validation_split: Ratio of training to validation data
# batch_size: The number of data items in each batch
# epochs: Training period
# verbose: Show the training process
train_process = model.fit(x=x_train, y=y_trainOneHot, validation_split=0.2, batch_size=16, epochs=10, verbose=2)
t1 = time.time()
Time_fit = float(t1 - t0)
print("The Time taken:{} seconds".format(Time_fit))
# Evaluate the accuracy of the model
scores = model.evaluate(x_valid, y_validOneHot)
print("Final accuracy=", scores[1])


# The process to show the training process
# Parameters:
# train_process: The parameter location where the training result is stored
# train: The execution result of the training data
# validation: The execution result of the validation data


def show_train_process(train_accuracy, val_accuracy):
    plt.plot(train_process.history[train_accuracy])
    plt.plot(train_process.history[val_accuracy])
    plt.title('The Training history')
    plt.xlabel('The number of Epoch')
    plt.xlim(0, 10)
    plt.ylabel('The accuracy')
    plt.ylim(0, 1)
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()


# Plot accuracy rate execution curve
show_train_process('accuracy', 'val_accuracy')
# Plot the loss function execution curve
show_train_process('loss', 'val_loss')
