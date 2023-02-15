# debug CB
print("code start")

from platform import python_version
print("Python Version: ", python_version())


# -----------------------------------------------------------------------------
# Imports

# Parsing command line arguments
import argparse
# Decoding camera images
import base64
# Frametimestamp saving
from datetime import datetime
# Reading and writing files
import os
# High level file operations
import shutil
# Matrix math
import numpy as np
# Real-time server
import socketio
# # Concurrent networking 
# import eventlet
# # Web server gateway interface
# import eventlet.wsgi
# # Image manipulation
# from PIL import Image
# # Web framework
# from flask import Flask
# # Input output
# from io import BytesIO


# # Load pre-trained model
# from keras.models import load_model # To import the keras package, first you need to need to install the tensorflow package


# # Helper class
# import utils_v02 as utils


# # -----------------------------------------------------------------------------
# # Main program setup

# # Intialize the Server
# sio = socketio.Server() 
# # Initialize Flask Web App
# app = Flask(__name__)

# # Initialize model variable and image array as empty
# model = None
# prev_image_array = None

# # Set min/max speed for our autonomous car
# MAX_SPEED = 25
# MIN_SPEED = 10
# speed_limit = MAX_SPEED


# # -----------------------------------------------------------------------------
# # Events Handlers

# # Registering event handler for the server
# @sio.on('telemetry')
# def telemetry(sid, data):
#     if data:
#         # The current steering angle of the car
#         steering_angle = float(data["steering_angle"])
#         # The current throttle of the car, how hard to push peddle
#         throttle = float(data["throttle"])
#         # The current speed of the car
#         speed = float(data["speed"])
#         # The current image from the center camera of the car
#         image = Image.open(BytesIO(base64.b64decode(data["image"])))
#         print("opened image")
#         try:
#             image = np.asarray(image)       # From PIL image to numpy array
#             image = utils.preprocess(image) # Apply the preprocessing
#             image = np.array([image])       # The model expects 4D array

#             # Predict the steering angle for the image
#             steering_angle = float(model.predict(image, batch_size=1))
#             # Lower the throttle as the speed increases
#             # If the speed is above the current speed limit, we are on a downhill.
#             # Make sure we slow down first and then go back to the original max speed.
#             global speed_limit
#             if speed > speed_limit:
#                 speed_limit = MIN_SPEED  # slow down
#             else:
#                 speed_limit = MAX_SPEED
#             throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

#             print('{} {} {}'.format(steering_angle, throttle, speed))
#             send_control(steering_angle, throttle)



#         except Exception as e:
#             print(e)



# @sio.on('connect')
# def connect(sid, environ):
#     print("connect ",sid)
#     send_control(0,0)

# # -----------------------------------------------------------------------------
# # Functions

# # Send control commands to server
# def send_control(steering_angle, throttle):
#     sio.emit(
#         "steer",
#         data={
#             'steering_angle': steering_angle.__str__(),
#             'throttle': throttle.__str__()
#         },
#         skip_sid=True
#     )


# # -----------------------------------------------------------------------------
# # Main Program

# if __name__ == '__main__':
#     # Parser to get the arguments to run the code with
#     parser = argparse.ArgumentParser(description='Remote Driving')
#     # Add the pre-trainned model argument
#     parser.add_argument(
#         'model',
#         type=str,
#         help='Path to model h5 file. Model should be on the same path.'
#     )
#     args = parser.parse_args()

#     # Load the model
#     model = load_model(args.model)
#     print("h5 model loaded!")

#     # Wrap Flask application with engineio's middleware
#     app = socketio.Middleware(sio, app)

#     # Deploy as an eventlet WSGI server
#     eventlet.wsgi.server(eventlet.listen(('', 4567)), app)









# debug CB
print("code end")