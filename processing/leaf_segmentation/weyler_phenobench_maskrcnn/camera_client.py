import camera_client
import yaml
import cv2
import torch
import models
import numpy as np
import colorsys

CHECKPOINT_FILE = "data/checkpoint.pt"
CONFIG_FILE = "configs/maskrcnn_leaves.yaml"
HUE_STEP = 15

def main():
    cfg = yaml.safe_load(open(CONFIG_FILE))

    model = models.get_model(cfg)
    weights = torch.load(CHECKPOINT_FILE)["model_state_dict"]
    model.load_state_dict(weights)

    with torch.autograd.set_detect_anomaly(True):
        model.network.eval()

        with torch.no_grad():
            def step(frame: cv2.typing.MatLike):
                i = 0
                # TODO: probably conversion necessary
                _, _, predictions = model.test_step(frame)

                size = item['image'][i].shape[1]
                scores = predictions[i]['scores'].cpu().numpy()
                labels = predictions[i]['labels'].cpu().numpy()
                # converting boxes to center, width, height format
                boxes_ = predictions[i]['boxes'].cpu().numpy()
                num_pred = len(boxes_)

                cx = (boxes_[:,2] + boxes_[:,0])/2
                cy = (boxes_[:,3] + boxes_[:,1])/2
                bw = boxes_[:,2] - boxes_[:,0]
                bh = boxes_[:,3] - boxes_[:,1]

                cv2_image = frame
                #cv2_image = np.transpose(numpy_image, (1, 2, 0))
                #cv2_image *= 255
                #cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
                cv2_image_bbox = cv2_image.copy()
                for j in range(num_pred):
                    low_x = int(cx[j]-bw[j]/2)
                    low_y = int(cy[j]-bh[j]/2)
                    high_x = int(cx[j]+bw[j]/2)
                    high_y = int(cy[j]+bh[j]/2)
                    top_left = (low_x, low_y)
                    bottom_right = (high_x, high_y)
                    if low_x - high_x == 0 or low_y - high_y == 0:
                        continue
                    cv2.rectangle(cv2_image_bbox, top_left, bottom_right, colorsys.hsv_to_rgb(j * HUE_STEP, 1, 1), 2)
                cv2.imshow(cv2_image_bbox)

            camera_client.setup(step)

