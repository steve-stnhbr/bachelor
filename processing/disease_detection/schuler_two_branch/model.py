# Here's a codeblock just for fun. You should be able to upload an image here 
# and have it classified without crashing
import numpy as np
import camclient
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from skimage import color as skimage_color
import cai
import cai.layers
import asyncio

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
    return np.pad(img, pad_width=((padx,padx),(pady,pady),(0,0)))

def transform_image(img, target_size=(224,224), smart_resize=False, lab=False, rescale=False, bipolar=False):
    def local_rescale(img,  lab):
        if (lab):
            if (bipolar):
                img[:,:,0:3] /= [25, 50, 50]
                img[:,:,0] -= 2
            else:
                img[:,:,0:3] /= [100, 200, 200]
                img[:,:,1:3] += 0.5
        else:
            if (bipolar):
                img /= 64
                img -= 2
            else:
                img /= 255
    if (smart_resize):
        img = img_to_array(img, dtype='float32')
        if (lab):
            img /= 255
            img = skimage_color.rgb2lab(img)
        if(rescale):
            local_rescale(img,  lab)
        img = add_padding_to_make_img_array_squared(img)
        if ((img.shape[0] != target_size[0]) or (img.shape[1] != target_size[1])):
            img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_NEAREST)
            print(img.shape)
    else:
        img = img_to_array(img, dtype='float32')
        if (lab):
            img /= 255
            img = skimage_color.rgb2lab(img)
        if(rescale):
            local_rescale(img,  lab)
    return img

def main():
    model = tf.keras.models.load_model('data/model/0.8_best.hdf5',custom_objects={'CopyChannels': cai.layers.CopyChannels})
    model.summary()

    
    def predict(img):
        imm_array = transform_image(img, smart_resize=True, lab=True)
        imm_array = np.expand_dims(imm_array, 0)
        
        predictions = model.predict(imm_array)
        prediction_score = tf.math.reduce_mean(tf.nn.softmax(predictions)).numpy()
        predicated_class = np.argmax(prediction_score)
        print(predicated_class, prediction_score)

    asyncio.run(camclient.setup(predict))

if __name__ == '__main__':
    main()