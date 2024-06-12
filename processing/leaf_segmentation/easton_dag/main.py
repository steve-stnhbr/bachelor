from leafmask import LeafMask, Config, ModelConfig, LeafMaskConfig, PanopticFPNConfig, CombineConfig, FPNConfig, BasisModuleConfig
import torch
import torchvision.transforms as T
# from detectron2.structures import Instances, Boxes
# from detectron2.data import detection_utils as utils
# from detectron2.data import transforms as T_detectron
# from detectron2.structures import BitMasks
# from detectron2.config import get_cfg
# from detectron2.modeling import build_model
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.data.datasets import register_coco_instances
import os
from dataset import LeafMaskDataset
from torch.utils.data import DataLoader
import torch.optim as optim

config_instance = Config(
    MODEL=ModelConfig(
        DEVICE='cuda',  # Assuming you have a GPU available
        LEAFMASK=LeafMaskConfig(
            INSTANCE_LOSS_WEIGHT=1.0,  # Commonly used value for instance loss weight
            IN_FEATURES=['p2', 'p3', 'p4', 'p5'],  # Typical features from FPN used in mask heads
            ATTN_SIZE=7  # Example value for attention size
        ),
        PANOPTIC_FPN=PanopticFPNConfig(
            COMBINE=CombineConfig(
                ENABLED=True,  # Enable combining semantic and instance segmentation
                OVERLAP_THRESH=0.5,  # Threshold for overlapping regions
                STUFF_AREA_LIMIT=4096,  # Minimum area for stuff segments
                INSTANCES_CONFIDENCE_THRESH=0.5  # Confidence threshold for instances
            )
        ),
        FPN=FPNConfig(
            OUT_CHANNELS=256  # Common output channels for FPN in detectron2
        ),
        BASIS_MODULE=BasisModuleConfig(
            NUM_BASES=4  # Example number of basis masks
        ),
        PIXEL_MEAN=[103.53, 116.28, 123.675],  # Typical mean values for input normalization
        PIXEL_STD=[1.0, 1.0, 1.0]  # Typical std values for input normalization
    )
)

# Directory to save models
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)


# Define image transformations
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=config_instance.MODEL.PIXEL_MEAN, std=config_instance.MODEL.PIXEL_STD),
])

# # Example function to prepare a single input
# def prepare_image(image, annotations=None, height=800, width=800):
#     # Apply transforms to the image
#     image = transform(image)
    
#     # Resize and pad the image
#     transform_gen = T_detectron.ResizeShortestEdge(
#         [height, width], max_size=1333
#     )
#     image, transforms = T_detectron.apply_transform_gens(transform_gen, image)
    
#     # Prepare instances annotations (if available)
#     if annotations:
#         for anno in annotations:
#             anno["bbox_mode"] = BoxMode.convert(anno["bbox"], anno["bbox_mode"], BoxMode.XYXY_ABS)
#         instances = utils.annotations_to_instances(annotations, image.shape[:2])
#         instances = utils.filter_empty_instances(instances)
#     else:
#         instances = None
    
#     return {"image": image, "instances": instances}


model = LeafMask()
model.to(config_instance.MODEL.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
dataset = LeafMaskDataset("../weyler_phenobench_maskrcnn/data/train")
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        losses = model(batch)
        total_loss = sum(loss for loss in losses.values())
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

    # Save model after each epoch
    model_save_path = os.path.join(save_dir, f"leaf_mask_model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), model_save_path)

# Final model saving
final_model_save_path = os.path.join(save_dir, "leaf_mask_model_final.pth")
torch.save(model.state_dict(), final_model_save_path)

