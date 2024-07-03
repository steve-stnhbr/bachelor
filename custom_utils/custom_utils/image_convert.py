import tensorflow as tf
import numpy as np
import skimage.color as skimage_color
import cv2

def transform(img, target_size=(224,224), smart_resize=False, lab=False, rescale=False, bipolar=False):
    img = img.numpy().astype(np.float32)
    def local_rescale(img,  lab):
        if (lab):
            # JP prefers bipolar input [-2,+2]
            if (bipolar):
                img[:,:,0:3] /= [25, 50, 50]
                img[:,:,0] -= 2.
            else:
                img[:,:,0:3] /= [100, 200, 200]
                img[:,:,1:3] += 0.5
        else:
            if (bipolar):
                img /= 64.
                img -= 2.
            else:
                img /= 255.
    def add_padding_to_make_img_array_squared(img):
        """ Adds padding to make the image squared.
        # Arguments
            img: an image as an array.
        """
        sizex = img.shape[0]
        sizey = img.shape[1]
        if (sizex == sizey):
            return img
        else:
            maxsize = np.max([sizex, sizey])
            padx = (maxsize - sizex) // 2
            pady = (maxsize - sizey) // 2
            return np.pad(img, pad_width=((padx, maxsize - sizex - padx), (pady, maxsize - sizey - pady), (0, 0)))
            #return tf.pad(img, [[padx, padx], [pady, pady]])
    
    def pad_to_square(img):
        # Get the current size of the image
        shape = tf.shape(img)
        height, width = shape[0], shape[1]

        # Determine the size of the new square image
        max_dim = tf.maximum(height, width)

        # Pad the image to make it square
        squared_img = tf.image.resize_with_crop_or_pad(img, target_height=max_dim, target_width=max_dim)

        return squared_img.numpy()
    if (smart_resize):
        if (lab):
            img /= 255
            img = skimage_color.rgb2lab(img)
        if(rescale):
            local_rescale(img,  lab)
        img = pad_to_square(img)
        if ((img.shape[0] != target_size[0]) or (img.shape[1] != target_size[1])):
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
    else:
        if (lab):
            img /= 255.
            img = skimage_color.rgb2lab(img)
        if(rescale):
            local_rescale(img,  lab)
    return img