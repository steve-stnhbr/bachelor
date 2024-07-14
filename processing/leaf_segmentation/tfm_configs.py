import tensorflow as tf
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.vision.configs import maskrcnn as maskrcnn_cfg
from official.vision.configs import backbones
from official.vision.configs import decoders

def maskrcnn_vit_fpn(path):
    task = maskrcnn_cfg.MaskRCNNTask(
        model=maskrcnn_cfg.MaskRCNN(
            backbone=backbones.Backbone(
                type='vit',
                vit=backbones.VisionTransformer(
                    model_name='vit-b16',
                    representation_size=768,
                    init_stochastic_depth_rate=0.1,
                )
            ),
            decoder=decoders.Decoder(
                type='fpn',
                fpn=decoders.FPN(
                    num_filters=256,
                    use_separable_conv=False,
                )
            ),
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
                num_classes=91,
                num_convs=4,
                num_filters=256,
                use_separable_conv=False,
                num_fcs=1,
                fc_dims=1024,
                class_agnostic_bbox_pred=False,
            ),
            mask_head=maskrcnn_cfg.MaskHead(
                num_classes=91,
                upsample_factor=2,
                num_convs=4,
                num_filters=256,
                use_separable_conv=False,
                class_agnostic_mask_pred=False,
            ),
            use_gt_boxes_for_masks=True,
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

    config_from_task(task)
    return config

def retinanet_resnet_fpn():
    exp_config = exp_factory.get_exp_config('retinanet_resnetfpn_coco')

    exp_config.task.model.input_size = [IMAGE_SIZE[1], IMAGE_SIZE[0], 3]

    # Modify the config as needed
    exp_config.task.model.num_classes = 2  # Adjust based on your number of classes
    
    # Configure for custom dataset
    exp_config.task.train_data.input_path = INPUT_PATH + "*train*"
    exp_config.task.validation_data.input_path = INPUT_PATH + "*val*"
    exp_config.task.train_data.global_batch_size = BATCH_SIZE
    exp_config.task.validation_data.global_batch_size = BATCH_SIZE

    # Disable COCO-specific configurations
    exp_config.task.annotation_file = INPUT_PATH + "instances.json"
    exp_config.task.use_coco_metrics = False

    # Configure data parsers
    exp_config.task.train_data.parser = exp_cfg.Parser()
    exp_config.task.validation_data.parser = exp_cfg.Parser()

    # Training parameters
    train_steps = 224_000
    exp_config.trainer.steps_per_loop = 200
    exp_config.trainer.summary_interval = 200
    exp_config.trainer.checkpoint_interval = 200
    exp_config.trainer.validation_interval = 200
    exp_config.trainer.validation_steps = 200
    exp_config.trainer.train_steps = train_steps
    exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = 200
    exp_config.trainer.optimizer_config.learning_rate.type = 'cosine'
    exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = train_steps
    exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = 0.07
    exp_config.trainer.optimizer_config.warmup.linear.warmup_learning_rate = 0.05

    return exp_config


def config_from_task(task):
    config = cfg.ExperimentConfig(
        task=task,
        trainer=cfg.TrainerConfig(
            train_steps=22500,
            validation_steps=278,
            steps_per_loop=100,
            summary_interval=100,
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
                        'values': [0.12, 0.012, 0.0012],
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

    exp_factory.register_config_factory('maskrcnn_vit_coco', lambda: config)
    return config