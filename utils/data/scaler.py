import copy 
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
#from skimage import transform
import tensorflow as tf
#import cv2

def scale(data, scale_per_image=True):
    """
        Scales data between 0 and 1 on a per image basis

        data (np.array) is either the test or training data
        per_image (bool) determines if data is processed on a per image basis

    """
    # Charl: output.astype('float32') uses more than 33GB RAM for LOFAR, which kills the execution on my machine.
    # data type is almost always already float32, so bypass this conversion
    # output = copy.deepcopy(data)
    output = np.empty(data.shape, dtype='float32')
    if scale_per_image:
        #output = output.astype('float32')
        for i, image in enumerate(data):
            # x,y,z = image.shape
            # output[i,...] = MinMaxScaler(feature_range=(0,1)
            #                               ).fit_transform(image.reshape([x*y,z])).reshape([x,y,z])
            mi, ma = np.min(image), np.max(image)
            output[i, ...] = (image - mi)/(ma -mi)
    else:
        mi, ma = np.min(data), np.max(data)
        output = (data - mi)/(ma -mi)
        if not (output.dtype == np.dtype('float64')):
            output = output.astype('float32')
    return output


