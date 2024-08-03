import tensorflow as tf
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.vision.configs import maskrcnn as maskrcnn_cfg
from official.vision.configs import backbones
from official.vision.configs import decoders

@exp_factory.register_config_factory('maskrcnn_resnet_fpn')
def maskrcnn_resnet_fpn(path, classes=2, image_size=(640, 640)):
    # Create a base experiment config
    exp_config = exp_factory.get_exp_config('maskrcnn_resnetfpn_coco')
    
    exp_config.task.init_checkpoint = None
    exp_config.task.init_checkpoint_module = None

    exp_config.task.model.input_size = [image_size[1], image_size[0], 3]

    # Modify the config as needed
    exp_config.task.model.num_classes = classes  # Adjust based on your number of classes

    return config_from_task(exp_config.task, path)

@exp_factory.register_config_factory('maskrcnn_mobilenet_fpn')
def maskrcnn_mobilenet_fpn(path, classes=2, image_size=(640, 640)):
    # Create a base experiment config
    exp_config = exp_factory.get_exp_config('maskrcnn_mobilenet_coco')
    
    exp_config.task.init_checkpoint = None
    exp_config.task.init_checkpoint_module = None

    exp_config.task.model.input_size = [image_size[1], image_size[0], 3]

    # Modify the config as needed
    exp_config.task.model.num_classes = classes  # Adjust based on your number of classes

    return config_from_task(exp_config.task, path)

@exp_factory.register_config_factory('maskrcnn_vit_fpn')
def maskrcnn_vit_fpn(path, classes=2, image_size=(640, 640)):
    task = maskrcnn_cfg.MaskRCNNTask(
        model=maskrcnn_cfg.MaskRCNN(
            input_size = [image_size[1], image_size[0], 3],
            num_classes=classes,
            backbone=backbones.Backbone(
                type='vit',
                vit=backbones.VisionTransformer(
                    model_name='vit-b16',
                    representation_size=768,
                    init_stochastic_depth_rate=0.1,
                    output_2d_feature_maps=True,
                )
            ),
            # decoder=decoders.Decoder(
            #     type='fpn',
            #     fpn=decoders.FPN(
            #         num_filters=256,
            #         use_separable_conv=False,
            #     )
            # ),
            decoder=decoders.Decoder(
                  type='identity', identity=decoders.Identity()),
            roi_sampler=maskrcnn_cfg.ROISampler(
                mix_gt_boxes=True,
                num_sampled_rois=512,
                foreground_fraction=0.25,
                foreground_iou_threshold=0.5,
                background_iou_high_threshold=0.5,
                background_iou_low_threshold=0.0,
            ),
            roi_aligner=maskrcnn_cfg.ROIAligner(
                crop_size=14,
                sample_offset=0.5,
            ),
            detection_head=maskrcnn_cfg.DetectionHead(
                num_convs=4,
                num_filters=256,
                use_separable_conv=False,
                num_fcs=1,
                fc_dims=1024,
                class_agnostic_bbox_pred=False,
            ),
            mask_head=maskrcnn_cfg.MaskHead(
                upsample_factor=2,
                num_convs=4,
                num_filters=256,
                use_separable_conv=False,
                class_agnostic=False,
            ),
        ),
        train_data=maskrcnn_cfg.DataConfig(
            input_path=path + '/*train*',
            is_training=True,
            global_batch_size=64,
            parser=maskrcnn_cfg.Parser(
                aug_rand_hflip=True,
                aug_scale_min=0.8,
                aug_scale_max=1.25,
            ),
        ),
        validation_data=maskrcnn_cfg.DataConfig(
            input_path=path + '/*val*',
            is_training=False,
            global_batch_size=8,
        ),
        losses=maskrcnn_cfg.Losses(
            l2_weight_decay=0.00004,
        ),
    )

    config = config_from_task(task, path)
    return config


@exp_factory.register_config_factory('retinanet_resnet_fpn')
def retinanet_resnet_fpn(path, batch_size=8, image_size=(640, 640)):
    exp_config = exp_factory.get_exp_config('retinanet_resnetfpn_coco')

    exp_config.task.init_checkpoint = None
    exp_config.task.init_checkpoint_module = None

    exp_config.task.model.input_size = [image_size[1], image_size[0], 3]

    config = config_from_task(exp_config.task, path)
    return config


def config_from_task(task, path, batch_size=8):
    config = cfg.ExperimentConfig(
        task=task,
        trainer=cfg.TrainerConfig(
            train_steps=584_200,
            validation_steps=512,
            steps_per_loop=1000,
            summary_interval=1000,
            checkpoint_interval=1000,
            max_to_keep=3,
            optimizer_config=cfg.OptimizationConfig({
                'optimizer': {
                    'type': 'sgd',
                    'sgd': {
                        'momentum': 0.9
                    }
                },
                'learning_rate': {
                    'type': 'stepwise',
                    'stepwise': {
                        'boundaries': [15000, 20000],
                        'values': [0.12, 0.012, 0.0054],
                    }
                },
                'warmup': {
                    'type': 'linear',
                    'linear': {
                        'warmup_steps': 500,
                        'warmup_learning_rate': 0.0067
                    }
                }
            })
        ),
        restrictions=[
            'task.train_data.is_training != None',
            'task.validation_data.is_training != None',
        ],
    )

    # Modify the config as needed
    config.task.model.num_classes = 2  # Adjust based on your number of classes
    
    # Configure for custom dataset
    config.task.train_data.input_path = path + "*train*"
    config.task.validation_data.input_path = path + "*val*"
    # config.task.train_data.tfds_name = "leaf_instance_dataset"
    # config.task.train_data.tfds_name = "train"
    # config.task.validation_data.tfds_name = "leaf_instance_dataset"
    # config.task.validation_data.tfds_name = "val"
    config.task.train_data.global_batch_size = batch_size
    config.task.validation_data.global_batch_size = batch_size

    # Disable COCO-specific configurations
    config.task.annotation_file = None
    config.task.use_coco_metrics = True
    config.task.use_wod_metrics = False
    config.task.use_approx_instance_metrics = False
    
    config.task.losses.frcnn_class_use_binary_cross_entropy = True
    
    config.trainer.validation_interval = 5000
    
#    exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = 200
#    exp_config.trainer.optimizer_config.learning_rate.type = 'cosine'
#    exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = train_steps
#    exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = 0.07
#    exp_config.trainer.optimizer_config.warmup.linear.warmup_learning_rate = 0.05
    
    return config