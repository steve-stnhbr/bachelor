import numpy as np
import cv2
import click
import os
import torch
from os.path import join, dirname, abspath
from torch.utils.data import DataLoader
from dataloaders.datasets import PlantsBase, collate_pdc
import models
import yaml
import colorsys
import torchvision.ops as tops

HUE_STEP = 30
OVERLAPPING_THRESHOLD = .1
PROBABILITY_THRESHOLD = .8
CONFIG_FILE = "configs/maskrcnn_leaves.yaml"


def save_model(model, epoch, optim, name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        }, name)

@click.command()
@click.option('--ckpt_file',
              '-w',
              type=str,
              help='path to trained weights (.pt)',
              default=join(dirname(abspath(__file__)),'checkpoints/best.pt'))
@click.option('--input',
              '-i',
              type=str,
              help='input directory',
              default=join(dirname(abspath(__file__)),'data/test_own/'))
def main(ckpt_file, input):
    cfg = yaml.safe_load(open(CONFIG_FILE))
    loaded_models = {}
    for checkpoint in ckpt_file.split(","):
        model = models.get_model(cfg)
        weights = torch.load(checkpoint)["model_state_dict"]
        
        model.load_state_dict(weights)
        loaded_models[checkpoint] = model

    with torch.no_grad():

        for file in os.listdir(input):
            img = cv2.imread(os.path.join(input, file))
            images = []

            for name, model in loaded_models.items():
                model.network.eval()
                results = step(model, img)
                # converting boxes to center, width, height format
                boxes_ = results['boxes'].cpu().numpy()
                ins_out = results["instances"]
                num_pred = len(boxes_)

                cx = (boxes_[:,2] + boxes_[:,0])/2
                cy = (boxes_[:,3] + boxes_[:,1])/2
                bw = boxes_[:,2] - boxes_[:,0]
                bh = boxes_[:,3] - boxes_[:,1]
                cv2_image = cv2.resize(img, (512, 512))
                # cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
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
                    cv2.rectangle(cv2_image_bbox, top_left, bottom_right, colorsys.hsv_to_rgb(j * HUE_STEP, 255, 255), 2)
                cv2_image_bbox //= 2
                if ins_out.shape == cv2_image.shape[:2]:
                    cv2_image_bbox[ins_out > 0] *= 2
                else:
                    print("Could not show instances")
                
                images.append(cv2_image_bbox)
            
            total = np.hstack(images)

            cv2.imshow("Prediction", total)
            cv2.waitKey(0) 

def step(model, frame: cv2.typing.MatLike):
    _, h, w = frame.shape
    frame = cv2.resize(frame, (512, 512))
    t = torch.from_numpy(frame / 255)
    t = t.type(torch.FloatTensor)
    t = t.permute((2, 0, 1))
    t = t.unsqueeze(0)
    t = t.to('cuda')
    print(t.shape)
    
    prediction = model.network(t)[0]

    predictions_dictionaries = []

    masks = prediction['masks'].squeeze()
    scores = prediction['scores']
    boxes = prediction['boxes']
    labels = prediction['labels']

    # non maximum suppression
    refined = tops.nms(boxes, scores, OVERLAPPING_THRESHOLD)
    refined_boxes = boxes[refined]
    refined_scores = scores[refined]
    refined_labels = labels[refined]
    refined_masks = masks[refined]

    # keeping only high scores
    high_scores = refined_scores > PROBABILITY_THRESHOLD

    # if any scores are above self.prob_th we can compute metrics
    if high_scores.sum():
        surviving_boxes = refined_boxes[high_scores]
        surviving_scores = refined_scores[high_scores]
        surviving_labels = refined_labels[high_scores]
        surviving_masks = refined_masks[high_scores]
        
        surviving_dict = {}
        surviving_dict['boxes'] = surviving_boxes.cuda()
        surviving_dict['labels'] = surviving_labels.cuda()
        surviving_dict['scores'] = surviving_scores.cuda()

        surviving_masks[surviving_masks>=0.5] = 1
        surviving_masks[surviving_masks<0.5] = 0

        sem_out = surviving_labels.unsqueeze(dim=1).unsqueeze(dim=1)*surviving_masks
        sem_out, _ = sem_out.max(dim=0)
        sem_out = sem_out.cpu().numpy()
        surviving_dict["semantics"] = sem_out

        ins_out = (torch.arange(surviving_masks.shape[0]).unsqueeze(dim=1).unsqueeze(dim=1).cuda()+1)*surviving_masks
        ins_out, _ = ins_out.max(dim=0)
        ins_out = ins_out.cpu().numpy()
        surviving_dict["instances"] = ins_out
    else:
        surviving_dict = {}
        surviving_dict['boxes'] = torch.empty((0, 4)).cuda()
        surviving_dict['labels'] = torch.empty(0).cuda()
        surviving_dict['scores'] = torch.empty(0).cuda()
        surviving_dict['masks'] = torch.empty((0, h, w)).cuda()
        sem_out = torch.zeros((h, w)).cuda()
        ins_out = torch.zeros((h, w)).cuda()
        surviving_dict["semantics"] = sem_out
        surviving_dict["instances"] = ins_out

    return surviving_dict

if __name__ == "__main__":
    main()
