import tensorflow as tf

def resize(data, dim):
    """
        Overloaded method for resizing input image

        data (np.array)  3D matrix containing image data (#images,X,Y,RGB/G)
        dim  (tuple) Tuple with 4 entires (#images, X, Y, RGB)

    """
    #return transform.resize(data,(data.shape[0], dim[0], dim[1], dim[2]), anti_aliasing=False)
    return tf.image.resize(data, [dim[0],dim[1]],antialias=False).numpy()
