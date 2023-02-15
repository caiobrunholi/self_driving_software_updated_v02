# debug CB
print("code start")

# Print Plataform
print(30*"-")
print("Versions: ")
from platform import python_version
print("Python Version: ", python_version())
print(30*"-")

from keras.losses import mean_squared_error

# -----------------------------------------------------------------------------
# Imports

# Data analysis toolkit - create, read, update, delete datasets
import pandas as pd
# Matrix math
import numpy as np
# To split out training and testing data 
from sklearn.model_selection import train_test_split
# keras is a high level wrapper on top of tensorflow (machine learning library)
# The Sequential container is a linear stack of layers
from keras.models import Sequential
#popular optimization strategy that uses gradient descent 
from keras.optimizers import Adam
#to save our model periodically as checkpoints for loading later
from keras.callbacks import ModelCheckpoint
#what types of layers do we want our model to have?
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
#helper class to define input shape and generate training images given image paths & steering angles
from utils_v02 import INPUT_SHAPE, batch_generator
#for command line arguments
import argparse
#for reading files
import os


# For debugging, allows for reproducible (deterministic) results 
np.random.seed(0)


# -----------------------------------------------------------------------------
# Functions

def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    # Reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    # Yay dataframes, we can select rows and columns by their names
    # Camera images saved as input data
    X = data_df[['center', 'left', 'right']].values
    # Steering commands saved as output data
    y = data_df['steering'].values

    # Split the data into a training (80), testing(20), and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid



def build_model(args):
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)

    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
    """
    model = Sequential() # It's a sequential model
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE)) # Image normalization layer using a lambda function, it avoids saturation and makes gradients work better
    model.add(Conv2D(filters=24, kernel_size=(5, 5), activation='elu', strides=(2,2))) # Convolutional layer
    model.add(Conv2D(filters=36, kernel_size=(5, 5), activation='elu', strides=(2,2))) # Convolutional layer
    model.add(Conv2D(filters=48, kernel_size=(5, 5), activation='elu', strides=(2,2))) # Convolutional layer
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu')) # Convolutional layer
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu')) # Convolutional layer
    model.add(Dropout(args.keep_prob)) # Dropout layer, percentage of dropout passed as arg by user (use 50%)
    model.add(Flatten()) # Flattens the data
    model.add(Dense(100, activation='elu')) # Connected layers
    model.add(Dense(50, activation='elu')) # Connected layers
    model.add(Dense(10, activation='elu')) # Connected layers
    model.add(Dense(1)) # Connected layers
    model.summary() # Makes a summary of the neural network

    return model # Returns the model



def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    # Saves the model after every epoch.
    # Quantity to monitor, verbosity i.e logging mode (0 or 1), 
    # If save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
    # Mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity. For val_acc, this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically inferred from the name of the monitored quantity.
    checkpoint = ModelCheckpoint(
        filepath='model_v02-{epoch:03d}.h5',
        monitor='val_loss',
        save_best_only=args.save_best_only,
        mode='auto'              
    )

    # Calculate the difference between expected steering angle and actual steering angle
    # Square the difference
    # Add up all those differences for as many data points as we have
    # Divide by the number of them
    # That value is our mean squared error! this is what we want to minimize via gradient descent

    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=args.learning_rate))    

    # The generator is run in parallel to the model, for efficiency.     
    # For instance, this allows you to do real-time data augmentation on images on CPU in parallel to training your model on GPU.
    # So we reshape our data into their appropriate batches and train our model simulatenously
    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                    args.samples_per_epoch,
                    args.nb_epoch,
                    max_queue_size=1,
                    validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                    validation_steps=len(X_valid),
                    callbacks=[checkpoint],
                    verbose=1)
    # model.fit(
    #     batch_size=batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
    #     steps_per_epoch=args.samples_per_epoch,
    #     epochs=args.nb_epoch,
    #     max_queue_size=1,
    #     validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
    #     validation_steps=len(X_valid),
    #     callbacks=[checkpoint],
    #     verbose=1
    #     )


# For command line args
def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


# -----------------------------------------------------------------------------
# Main Program

def main():
    # Get parameters and set default values
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    # Print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    # Load data
    data = load_data(args)
    # Build model
    model = build_model(args)
    # Train model on data, it saves as model.h5 
    train_model(model, args, *data)


if __name__ == '__main__':
    main()









# debug CB
print("code end")