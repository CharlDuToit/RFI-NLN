import numpy as np

def rgb2gray(rgb):
    """
        Convert rgb images to gray

        Parameters
        ----------
        rgb (np.array) array of rgb imags

        Returns
        -------
        np.array
    """
    if rgb.shape[-1] ==3:
        return np.expand_dims(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]),axis=-1)
    else: return rgb
