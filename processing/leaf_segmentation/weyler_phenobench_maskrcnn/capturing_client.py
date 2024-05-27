import camclient
import yaml
import cv2
import torch
import models
import numpy as np
import colorsys
import asyncio
from time import gmtime, strftime
import torchvision.ops as tops

CHECKPOINT_FILE = "data/checkpoint.pt"
CONFIG_FILE = "configs/maskrcnn_leaves.yaml"
HUE_STEP = 15

OVERLAPPING_THRESHOLD = .1
PROBABILITY_THRESHOLD = .8

def main():
    cfg = yaml.safe_load(open(CONFIG_FILE))

    model = models.get_model(cfg)
    weights = torch.load(CHECKPOINT_FILE)["model_state_dict"]
    model.load_state_dict(weights)
    model.to("cuda")

    with torch.autograd.set_detect_anomaly(True):
        model.network.eval()

        with torch.no_grad():
            def step(frame: cv2.typing.MatLike):
                print(frame.shape)
                t = torch.from_numpy(frame / 255)
                t = t.type(torch.FloatTensor)
                t = t.permute((2, 0, 1))
                t = t.unsqueeze(0)
                t = t.to('cuda')
                print(t.shape)
                
                prediction = model.network(t)[0]

                predictions_dictionaries = []

                scores = prediction['scores']
                boxes = prediction['boxes']
                labels = prediction['labels']

                # non maximum suppression
                refined = tops.nms(boxes, scores, OVERLAPPING_THRESHOLD)
                refined_boxes = boxes[refined]
                refined_scores = scores[refined]
                refined_labels = labels[refined]

                # keeping only high scores
                high_scores = refined_scores > PROBABILITY_THRESHOLD

                # if any scores are above self.prob_th we can compute metrics
                if high_scores.sum():
                    surviving_boxes = refined_boxes[high_scores]
                    surviving_scores = refined_scores[high_scores]
                    surviving_labels = refined_labels[high_scores]
                    
                    surviving_dict = {}
                    surviving_dict['boxes'] = surviving_boxes.cuda()
                    surviving_dict['labels'] = surviving_labels.cuda()
                    surviving_dict['scores'] = surviving_scores.cuda()
                else:
                    surviving_dict = {}
                    surviving_dict['boxes'] = torch.empty((0, 4)).cuda()
                    surviving_dict['labels'] = torch.empty(0).cuda()
                    surviving_dict['scores'] = torch.empty(0).cuda()

                # scores = prediction['scores'].cpu().numpy()
                labels = surviving_dict['labels'].cpu().numpy()
                # converting boxes to center, width, height format
                boxes_ = surviving_dict['boxes'].cpu().numpy()
                num_pred = len(boxes_)

                cx = (boxes_[:,2] + boxes_[:,0])/2
                cy = (boxes_[:,3] + boxes_[:,1])/2
                bw = boxes_[:,2] - boxes_[:,0]
                bh = boxes_[:,3] - boxes_[:,1]

                cv2_image = frame
                #cv2.imshow("Orig", frame)
                #cv2_image = np.transpose(cv2_image, (1, 2, 0))
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
                cv2.imshow("Prediction", cv2_image_bbox)
                cv2.waitKey(1) 

            asyncio.run(camclient.setup(step))

if __name__ == "__main__":
    main()