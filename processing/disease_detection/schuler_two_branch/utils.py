import os
from tensorflow.keras.preprocessing.image import img_to_array
import skimage

def get_classes(paths):
    return [os.path.normpath(path).split(os.sep)[-2] for path in paths]

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
            # JP prefers bipolar input [-2,+2]
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
    else:
        img = img_to_array(img, dtype='float32')
        if (lab):
            img /= 255
            img = skimage_color.rgb2lab(img)
        if(rescale):
            local_rescale(img,  lab)
    return img
    # def local_rescale(img,  lab):
    #     if (lab):
    #         if (bipolar):
    #             img[:,:,0:3] /= [25, 50, 50]
    #             img[:,:,0] -= 2
    #         else:
    #             img[:,:,0:3] /= [100, 200, 200]
    #             img[:,:,1:3] += 0.5
    #     else:
    #         if (bipolar):
    #             img /= 64
    #             img -= 2
    #         else:
    #             img /= 255
    # if (smart_resize):
    #     img = img_to_array(img, dtype='float32')
    #     if (lab):
    #         img /= 255
    #         img = skimage_color.rgb2lab(img)
    #     if(rescale):
    #         local_rescale(img,  lab)
    #     img = add_padding_to_make_img_array_squared(img)
    #     if ((img.shape[0] != target_size[0]) or (img.shape[1] != target_size[1])):
    #         img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_NEAREST)
    # else:
    #     img = img_to_array(img, dtype='float32')
    #     if (lab):
    #         img /= 255
    #         img = skimage_color.rgb2lab(img)
    #     if(rescale):
    #         local_rescale(img,  lab)
    # return img