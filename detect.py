import numpy as np
import cv2
from numpy import zeros, newaxis
from scipy.misc import imresize
from moviepy.editor import VideoFileClip

from keras.models import model_from_json

# Load Keras model
json_file = open('full_CNN_model.json', 'r')
json_model = json_file.read()
json_file.close()
model = model_from_json(json_model)
model.load_weights('full_CNN_model.h5')
cap = cv2.VideoCapture("challenge.mp4")
# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """

    # Get image ready for feeding into model
    #ret, frame = cap.read()
    #small_img = imresize(image, (80, 160, 3))
    small_img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    #cv2.imshow("test",small_img)
    #cv2.waitKey(0)

    small_img = np.array(small_img)
    small_img = small_img[None,:,:,1]
    small_img = small_img[:,:,:,newaxis]
    #small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = imresize(lane_drawn, (720, 1280, 3))

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 0.2, lane_image, 1, 0)

    return result

lanes = Lanes()

model.summary()

while True:
    ret, frame = cap.read()
    temp=road_lines(frame)
    cv2.imshow('a',temp)

    k = cv2.waitKey(1)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
        break

