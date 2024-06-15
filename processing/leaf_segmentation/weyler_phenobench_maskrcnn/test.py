import numpy as np
import cv2
import click
import os
import torch
from os.path import join, dirname, abspath
from torch.utils.data import DataLoader
from dataloaders.datasets import Leaves, collate_pdc
import models
import yaml
import colorsys

HUE_STEP = 30


def save_model(model, epoch, optim, name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        }, name)

@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'configs/cfg.yaml'))
@click.option('--ckpt_file',
              '-w',
              type=str,
              help='path to trained weights (.pt)',
              default=join(dirname(abspath(__file__)),'checkpoints/best.pt'))
@click.option('--out',
              '-o',
              type=str,
              help='output directory',
              default=join(dirname(abspath(__file__)),'results/'))
def main(config, ckpt_file, out):
    cfg = yaml.safe_load(open(config))

    val_dataset = Leaves(datapath=cfg['data']['val'], overfit=cfg['train']['overfit'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['train']['batch_size'], collate_fn=collate_pdc, shuffle=False, drop_last=False, num_workers=cfg['train']['workers'])

    model = models.get_model(cfg)
    weights = torch.load(ckpt_file)["model_state_dict"]
    model.load_state_dict(weights)
    
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out,'predictions/leaf_instances/'), exist_ok=True)
    os.makedirs(os.path.join(out,'predictions/semantics/'), exist_ok=True)
    os.makedirs(os.path.join(out,'leaf_bboxes/'), exist_ok=True)
    os.makedirs(os.path.join(out, 'leafs_segmented'), exist_ok=True)
    os.makedirs(os.path.join(out, 'leafs_segmented_masked'), exist_ok=True)
    os.makedirs(os.path.join(out, 'leafs_masked'), exist_ok=True)

    with torch.autograd.set_detect_anomaly(True):
        model.network.eval()
        for idx, item in enumerate(iter(val_loader)):
            with torch.no_grad():
                print("Starting inference")
                size = item['image'][0].shape[1]
                semantic, instance, predictions = model.test_step(item)
                
                res_names = item['name']
                for i in range(len(res_names)):
                    fname_ins = os.path.join(out,'predictions/leaf_instances/',res_names[i])
                    fname_sem = os.path.join(out,'predictions/semantics/',res_names[i])
                    fname_box = os.path.join(out,'leaf_bboxes',res_names[i].replace('png','txt'))

                    inst = instance[i].cpu().long().numpy()

                    cv2.imwrite(fname_sem, semantic[i].cpu().long().numpy())
                    cv2.imwrite(fname_ins, inst)


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
                    
                    # Convert the numpy array to a cv2 image
                    numpy_image = item['image'][i].cpu().numpy()
                    cv2_image = np.transpose(numpy_image, (1, 2, 0))
                    cv2_image *= 255
                    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
                    cv2_image_bbox = cv2_image.copy()
                    cv2_image_masked = cv2_image.copy()
                    cv2_image_masked[inst == 0] = 0
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
                        cv2.imwrite(os.path.join(out,'leafs_segmented',"{}_{}.png".format(res_names[i].replace('.png', ''), j)), cv2_image[low_y:high_y, low_x:high_x])
                        cv2.imwrite(os.path.join(out,'leafs_segmented_masked',"{}_{}.png".format(res_names[i].replace('.png', ''), j)), cv2_image_masked[low_y:high_y, low_x:high_x])
                    cv2.imwrite(os.path.join(out,'leaf_bboxes',res_names[i]), cv2_image_bbox)
                    cv2.imwrite(os.path.join(out,'leafs_masked',"{}_{}.png".format(res_names[i].replace('.png', ''), j)), cv2_image_masked)
                    cv2.imshow("Preview", cv2_image_bbox)

                    # ready to be saved
                    pred_cls_box_score = np.hstack((labels.reshape(num_pred,1), 
                                         cx.reshape(num_pred,1)/size,
                                         cy.reshape(num_pred,1)/size,
                                         bw.reshape(num_pred,1)/size,
                                         bh.reshape(num_pred,1)/size,
                                         scores.reshape(num_pred,1)
                                        ))
                    np.savetxt(fname_box, pred_cls_box_score, fmt='%f')

if __name__ == "__main__":
    main()
