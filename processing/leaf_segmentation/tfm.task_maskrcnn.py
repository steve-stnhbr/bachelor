import tensorflow as tf
import tensorflow_models as tfm

from official.vision.configs import maskrcnn as exp_cfg
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.vision.tasks import maskrcnn
import os


IMAGE_SIZE = (640, 640)
BATCH_SIZE = 4

def build_experiment_config(train_dataset):
    # Create a base experiment config
    exp_config = exp_factory.get_exp_config('maskrcnn_mobilenet_coco')

    # Modify the config as needed
    exp_config.task.model.num_classes = 2  # Adjust based on your number of classes
    exp_config.task.model.mask_head.num_convs = 4
    exp_config.task.model.mask_head.num_filters = 256
    exp_config.task.model.mask_head.use_separable_conv = False
    
    # Set the input config to use your custom dataset
    exp_config.task.train_data.input_path = ''  # Not used with custom dataset
    exp_config.task.train_data.global_batch_size = 16  # Adjust as needed
    exp_config.task.train_data.dtype = 'float32'
    exp_config.task.train_data.parser = None  # We're not using the default COCO parser

    exp_config.trainer.train_steps = 10000  # Adjust based on your dataset size
    exp_config.trainer.optimizer_config.learning_rate.type = 'stepwise'
    exp_config.trainer.optimizer_config.learning_rate.stepwise.boundaries = [6000, 8000]
    exp_config.trainer.optimizer_config.learning_rate.stepwise.values = [0.08, 0.008, 0.0008]

    # Create the task instance
    task = maskrcnn.MaskRCNNTask(exp_config.task)

    # Override the task's build_inputs method to use your custom dataset
    def build_inputs(self, params, input_context=None):
        return train_dataset

    task.build_inputs = build_inputs.__get__(task)
    exp_config.task = task
    return exp_config


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

def _load_data(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    
    image = tf.image.resize(image, IMAGE_SIZE)
    mask = tf.image.resize(mask, IMAGE_SIZE)
    image = tf.keras.applications.resnet.preprocess_input(image)
    
    filename = tf.strings.split(image_path, os.path.sep)[-1]
    image_id = tf.strings.to_number(tf.strings.split(filename, '.')[0], out_type=tf.int32)
    
    # You'll need to implement masks_to_boxes function
    boxes, areas = tf.py_function(masks_to_boxes, [mask], [tf.float32, tf.float32])
    
    return {
        'image': image,
        'image/filename': filename,
        'image/id': image_id,
        'objects': {
            'id': tf.range(tf.shape(boxes)[0], dtype=tf.int32),
            'bbox': boxes,
            'area': areas,
            'is_crowd': tf.zeros(tf.shape(boxes)[0], dtype=tf.bool),
            'label': tf.ones(tf.shape(boxes)[0], dtype=tf.int32)
        }
    }

path = "_data/combined/train"
mask_path = "leaf_instances"

image_files = [os.path.join(path, 'images', file) for file in os.listdir(os.path.join(path, 'images'))]
image_files = tf.convert_to_tensor(image_files, dtype=tf.string)

mask_files = [os.path.join(path, mask_path, file) for file in os.listdir(os.path.join(path, mask_path))]
mask_files = tf.convert_to_tensor(mask_files, dtype=tf.string)

train_dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
train_dataset = train_dataset.map(_load_data, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)

# Build the task with your custom dataset
exp_config = build_experiment_config(train_dataset)

logical_device_names = [logical_device.name for logical_device in tf.config.list_logical_devices()]
distribution_strategy = tf.distribute.OneDeviceStrategy(logical_device_names[0])

with distribution_strategy.scope():
    model_dir = "out"
    task = tfm.core.task_factory.get_task(exp_config.task, logging_dir=model_dir)

    for images, labels in task.build_inputs(exp_config.task.train_data).take(1):
        print()
        print(f'images.shape: {str(images.shape):16}  images.dtype: {images.dtype!r}')
        print(f'labels.shape: {str(labels.shape):16}  labels.dtype: {labels.dtype!r}')

    model, eval_logs = tfm.core.train_lib.run_experiment(
        distribution_strategy=distribution_strategy,
        task=task,
        mode='train_and_eval',
        params=exp_config,
        model_dir=model_dir,
        run_post_eval=True)
    
    tf.keras.utils.plot_model(model, show_shapes=True)

    for key, value in eval_logs.items():
        if isinstance(value, tf.Tensor):
            value = value.numpy()
        print(f'{key:20}: {value:.3f}')

    for images, labels in task.build_inputs(exp_config.task.train_data).take(1):
        predictions = model.predict(images)
        predictions = tf.argmax(predictions, axis=-1)

        show_batch(images, labels, tf.cast(predictions, tf.int32))