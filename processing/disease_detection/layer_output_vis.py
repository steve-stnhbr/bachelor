import os
import click
import keras
from tensorflow.keras import backend as K
from keras import Function as k_func
import cv2
from custom_utils import transform
import tensorflow as tf
import numpy as np

@click.command()
@click.option('-m', '--model')
@click.option('-i', '--input')
@click.option('-o', '--output')
@click.option('-l', '--lab', is_flag=True)
def main(model, input, output, lab):
    if not os.path.exists(input):
        raise FileNotFoundError(input)
    
    if ',' in model:
        models = model.split(',')
        for model in models:
            handle_model(model, input, output, lab)
    else:
        handle_model(model, input, output, lab)

def handle_model(model, input, output, lab):
    model_name = os.path.basename(model)
    model = keras.models.load_model(model)

    if os.path.isdir(input):
        for file in os.listdir(input):
            visualize(model, model_name, os.path.join(input, file), output, lab)
    else:
        visualize(model, model_name, input, output, lab)

def visualize(model, model_name, file, output, lab=False):
    # create output dir
    file_name = os.path.basename(file)
    folder = os.path.join(output, model_name[:model_name.index('.')], file_name[:file_name.index('.')])
    os.makedirs(folder, exist_ok=True)

    # read input
    img = cv2.imread(file)
    img = transform(img, lab=lab, rescale=True, smart_resize=True)
    
    model(keras.layers.Input(img.shape))

    img = np.expand_dims(img, 0)

    layer_outputs = [layer.output for layer in model.layers[1:]]
    visual_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)

    # run your image through the network; make a prediction
    feature_maps = visual_model(img)

    # Plotting intermediate representations for your image

    # Collect the names of each layer except the first one for plotting
    layer_names = [layer.name for layer in model.layers[1:]]

    # Plotting intermediate representation images layer by layer
    for layer_name, feature_map in zip(layer_names, feature_maps):
        if True or len(feature_map.shape) == 4: # skip fully connected layers
            # number of features in an individual feature map
            n_features = feature_map.shape[-1]
            # The feature map is in shape of (1, size, size, n_features)
            size = feature_map.shape[1]
            # Tile our feature images in matrix `display_grid
            display_grid = np.zeros((size, size * n_features))
            # Fill out the matrix by looping over all the feature images of your image
            for i in range(n_features):
                # Postprocess each feature of the layer to make it pleasible to your eyes
                x = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                # We'll tile each filter into this big horizontal grid
                display_grid[:, i * size : (i + 1) * size] = x
            # Display the grid
            print(f"Writing output for layer {layer_name} of image {file}")
            output = (output[0] * 255).astype("uint8")
            cv2.imwrite(os.path.join(folder, layer_name + ".png"), display_grid)    
        

if __name__ == '__main__':
    main()