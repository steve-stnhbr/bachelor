import cai.layers
import cai.datasets
import cai.models
import numpy as np
from keras import backend
from keras import layers
import keras.applications
import keras.applications.inception_v3
from keras.applications.inception_v3 import InceptionV3
from keras_applications import imagenet_utils
from keras_applications import get_submodules_from_kwargs
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils import np_utils
from sklearn.utils import class_weight
from skimage import color as skimage_color
import gc
from keras.callbacks import EarlyStopping
import multiprocessing
import random
import glob

def load_plantvillage(seed=None, root_dir=None, lab=False, 
  verbose=True, bipolar=False, base_model_name='plant_leaf'):
  if root_dir == None:
    print("define where the plantvillage folder is.")
    return
  random.seed(seed)
  def read_from_paths(paths):
    x=[]
    for path in paths:
      img = load_img(path, target_size=(224,224))
      img = img_to_array(img)
      x.append(img)
    return x
  
  classes = os.listdir(root_dir)
  classes = sorted(classes)

  train_path = []
  val_path = []
  test_path = []

  train_x,train_y = [],[]
  val_x,val_y = [],[]
  test_x,test_y =[],[]

  #read path and categorize to three groups , 6,2,2
  for i,_class in enumerate(classes):
      paths = glob.glob(os.path.join(root_dir,_class,"*"))
      paths = [n for n in paths if n.endswith(".JPG") or n.endswith(".jpg")]
      random.shuffle(paths)
      cat_total = len(paths)
      train_path.extend(paths[:int(cat_total*0.6)])
      train_y.extend([i]*int(cat_total*0.6))
      val_path.extend(paths[int(cat_total*0.6):int(cat_total*0.8)])
      val_y.extend([i]*len(paths[int(cat_total*0.6):int(cat_total*0.8)]))
      test_path.extend(paths[int(cat_total*0.8):])
      test_y.extend([i]*len(paths[int(cat_total*0.8):]))
  
  if (verbose):
    print ("loading train images")
  train_x = np.array(read_from_paths(train_path), dtype='float16')
  
  if (verbose):
    print ("loading validation images")
  val_x = np.array(read_from_paths(val_path), dtype='float16')

  if (verbose):
    print ("loading test images")
  test_x = np.array(read_from_paths(test_path), dtype='float16')

  train_y = np.array(train_y)
  val_y = np.array(val_y)
  test_y = np.array(test_y)

  #convert everything to numpy
  #train_x = np.array(train_x)/255.
  #val_x = np.array(val_x)/255.
  #test_x = np.array(test_x)/255.

  if (lab):
        # LAB datasets are cached
        cachefilename = 'cache-lab-'+base_model_name+'-'+str(bipolar)+'-'+str(train_x.shape[1])+'-'+str(train_x.shape[2])+'.npz'
        if (verbose):
            print("Converting RGB to LAB: "+cachefilename)
        if not os.path.isfile(cachefilename):            
            train_x /= 255
            val_x /= 255
            test_x /= 255
            if (verbose):
                print("Converting training.")
            cai.datasets.skimage_rgb2lab_a(train_x,  verbose)
            if (verbose):
                print("Converting validation.")
            cai.datasets.skimage_rgb2lab_a(val_x,  verbose)
            if (verbose):
                print("Converting test.")
            cai.datasets.skimage_rgb2lab_a(test_x,  verbose)
            gc.collect()
            if (bipolar):
                # JP prefers bipolar input [-2,+2]
                train_x[:,:,:,0:3] /= [25, 50, 50]
                train_x[:,:,:,0] -= 2
                val_x[:,:,:,0:3] /= [25, 50, 50]
                val_x[:,:,:,0] -= 2
                test_x[:,:,:,0:3] /= [25, 50, 50]
                test_x[:,:,:,0] -= 2
            else:
                train_x[:,:,:,0:3] /= [100, 200, 200]
                train_x[:,:,:,1:3] += 0.5
                val_x[:,:,:,0:3] /= [100, 200, 200]
                val_x[:,:,:,1:3] += 0.5
                test_x[:,:,:,0:3] /= [100, 200, 200]
                test_x[:,:,:,1:3] += 0.5
            #if (verbose):
              #print("Saving: "+cachefilename)
              #np.savez_compressed(cachefilename, a=train_x,  b=val_x, c=test_x)
        else:
            if (verbose):
              print("Loading: "+cachefilename)
            loaded = np.load(cachefilename)
            train_x = loaded['a']
            val_x = loaded['b']
            test_x = loaded['c']
            gc.collect()
  else:
        if (verbose):
            print("Loading RGB.")
        if (bipolar):
            train_x /= 64
            val_x /= 64
            test_x /= 64
            train_x -= 2
            val_x -= 2
            test_x -= 2
        else:
            train_x /= 255
            val_x /= 255
            test_x /= 255

  if (verbose):
        for channel in range(0, train_x.shape[3]):
            sub_matrix = train_x[:,:,:,channel]
            print('Channel ', channel, ' min:', np.min(sub_matrix), ' max:', np.max(sub_matrix))
  #calculate class weight
  classweight = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)

  #convert to categorical
  train_y = np_utils.to_categorical(train_y, 38)
  val_y = np_utils.to_categorical(val_y, 38)
  test_y = np_utils.to_categorical(test_y, 38)
  print("loaded")

  return train_x,val_x,test_x,train_y,val_y,test_y,classweight,classes

url_zip_file="https://data.mendeley.com/datasets/tywbtsjrjv/1/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/Plant_leaf_diseases_dataset_without_augmentation.zip?dl=1"
local_zip_file="plant_leaf.zip"
expected_folder_name="plant_leaf"
Verbose=True
cai.datasets.download_zip_and_extract(
    url_zip_file=url_zip_file, local_zip_file=local_zip_file, 
    expected_folder_name=expected_folder_name, Verbose=Verbose)


data_dir = "plant_leaf/Plant_leave_diseases_dataset_without_augmentation/"
print(os.listdir(data_dir))

train_x, val_x, test_x, train_y, val_y, test_y, classweight, classes = load_plantvillage(seed=7, root_dir=data_dir, lab=lab)
print(train_x.shape,val_x.shape,test_x.shape)
print(train_y.shape,val_y.shape,test_y.shape)

for two_paths_second_block in [False, True]:
  for l_ratio in [0.1, 0.2, 0.33, 0.5, 0.66, 0.8, 0.9]:
    basefilename = 'two-path-inception-v6-'+str(two_paths_second_block)+'-'+str(l_ratio)
    print('Running: '+basefilename)
    model = cai.models.compiled_two_path_inception_v3(
      input_shape=(224,224,3),
      classes=38, 
      two_paths_first_block=True,
      two_paths_second_block=two_paths_second_block,
      l_ratio=l_ratio,
      ab_ratio=(1-l_ratio),
      max_mix_idx=5, 
      model_name='two_path_inception_v3'
      )    
    monitor='val_accuracy'
    best_result_file_name = basefilename+'-best_result.hdf5'
    save_best = keras.callbacks.ModelCheckpoint(
      filepath=best_result_file_name, 
      monitor=monitor, 
      verbose=1, 
      save_best_only=True, 
      save_weights_only=False, 
      mode='max', 
      period=1)
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=32,
      validation_data=(val_x,val_y),
      callbacks=[save_best],class_weight=classweight,
      workers=multiprocessing.cpu_count())
    print('Testing Last Model: '+basefilename)
    evaluated = model.evaluate(test_x,test_y)
    for metric, name in zip(evaluated,["loss","acc","top 5 acc"]):
      print(name,metric)
    print('Best Model Results: '+basefilename)
    model = keras.models.load_model(best_result_file_name, custom_objects={'CopyChannels': cai.layers.CopyChannels})
    evaluated = model.evaluate(test_x,test_y)
    cai.models.save_model(model, basefilename)
    for metric, name in zip(evaluated,["loss","acc","top 5 acc"]):
      print(name,metric)
    print('Finished: '+basefilename)