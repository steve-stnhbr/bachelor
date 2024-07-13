import os

def process_example(image_path, imgs_dir):

    def masks_to_boxes(masks, area_threshold=50):
        # if masks.numel() == 0:
        #     return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

        n = masks.shape[0]

        bounding_boxes = np.zeros(
            (n, 4), dtype=np.float16)

        for index, mask in enumerate(masks):
            if mask.sum() < area_threshold:
                continue
            y, x = np.nonzero(mask)
            bounding_boxes[index, 0] = np.min(x)
            bounding_boxes[index, 1] = np.min(y)
            bounding_boxes[index, 2] = np.max(x)
            bounding_boxes[index, 3] = np.max(y)
        bounding_boxes_area = bounding_boxes.sum(axis=1)
        bounding_boxes = bounding_boxes[~(bounding_boxes_area==0)]
        return bounding_boxes, bounding_boxes_area  

    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    
    image = tf.image.resize(image, IMAGE_SIZE)
    mask = tf.image.resize(mask, IMAGE_SIZE)
    image = tf.keras.applications.resnet.preprocess_input(image)
    
    filename = tf.strings.split(image_path, os.path.sep)[-1]
    name, img_format = tf.strings.split(filename, '.')
    filename = os.path.join(imgs_dir, filename)
    
    # You'll need to implement masks_to_boxes function
    #boxes, areas = tf.py_function(masks_to_boxes, [mask], [tf.float32, tf.float32])
    boxes, areas = masks_to_boxes(mask)

    # Get the shape of the input array
    x, y, n = mask.shape

    # Find unique values in the n dimension
    unique_values = np.unique(mask)

    # Number of unique values
    u = len(unique_values)

    # Initialize the new array with zeros
    masks = np.zeros((x, y, u), dtype=int)

    # Create the binary mask for each unique value
    for i, val in enumerate(unique_values):
        masks[:, :, i] = np.any(mask == val, axis=2).astype(int)

    keys_to_features = {
        'image/encoded':
            tfrecord_lib.convert_to_feature(image.numpy()),
        'image/filename':
            tfrecord_lib.convert_to_feature(filename.encode('utf8')),
        'image/format':
            tfrecord_lib.convert_to_feature(img_format.encode('utf8')),
        'image/height':
            tfrecord_lib.convert_to_feature(image.shape[1]),
        'image/width':
            tfrecord_lib.convert_to_feature(image.shape[0]),
        'image/source_id':
            tfrecord_lib.convert_to_feature(str(name).encode('utf8')),
        'image/object/bbox/xmin':
            tfrecord_lib.convert_to_feature(boxes[:,0]),
        'image/object/bbox/xmax':
            tfrecord_lib.convert_to_feature(boxes[:,2]),
        'image/object/bbox/ymin':
            tfrecord_lib.convert_to_feature(boxes[:,1]),
        'image/object/bbox/ymax':
            tfrecord_lib.convert_to_feature(boxes[:,3]),
        'image/object/class/text':
            tfrecord_lib.convert_to_feature('leaf'),
        'image/object/class/label':
            tfrecord_lib.convert_to_feature(1),
        'image/object/is_crowd':
            tfrecord_lib.convert_to_feature(0),
        'image/object/area':
            tfrecord_lib.convert_to_feature(areas, 'float_list'),
        'image/object/mask':
            tfrecord_lib.convert_to_feature(masks)
    }
    
    example = tf.train.Example(
        features=tf.train.Features(feature=keys_to_features))
    return example

@click.command()
@click.argument("input_path")
@click.argument('output_path')
@click.option('-n', '--num-shards', type=int)
def main(input_path, output_path, num_shards):
    writers = [
        tf.io.TFRecordWriter(
            output_path + prefix +'-%05d-of-%05d.tfrecord' % (i, num_shards))
        for i in range(num_shards)
    ]

    imgs_dir = os.path.join(output_path, "imgs")
    os.mkdirs(imgs_dir, exist_ok=True)

    for idx, file_name in enumerate(tqdm.tqdm(os.listdir(input_path))):
        file = os.path.join(input_path, file_name)
        with open(os.path.join(imgs_dir, img_name), 'wb') as handler:
            handler.write(img_data)
        tf_example = process_example(file, imgs_dir)
        writers[idx % num_shards].write(tf_example.SerializeToString())