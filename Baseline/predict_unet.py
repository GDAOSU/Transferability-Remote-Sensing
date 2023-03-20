import os
import os.path as osp
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random

import metrics
from loader.datasetval import datasetval
import segmentation_models_pytorch as smp



NUM_CLASSES = 4
HEIGHT = True
DATASET = 'London_Haiti'
SOURCE = DATASET.split("_")[0]
TARGET = DATASET.split("_")[1]
GPU = 0
SEED = 2023

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
cudnn.benchmark = False
cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.cuda.manual_seed_all(SEED)

def get_arguments():

    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--source", type=str, default=SOURCE)
    parser.add_argument("--target", type=str, default=TARGET)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--height", type=bool, default=HEIGHT,
                        help="Use height or not")
    parser.add_argument("--gpu", type=int, default=GPU,
                        help="choose gpu device.")
    return parser.parse_args()

args = get_arguments()

def main():
    """Create the model and start the training."""

    # save checkpoint direction
    here = osp.dirname(osp.abspath(__file__))
    if args.height:
        out_dir = osp.join(here,'results', SOURCE + '_' + TARGET + '_height_unet')
    else:
        out_dir = osp.join(here,'results', SOURCE + '_' + TARGET + '_no_height_unet')

    # dataloader
    MAIN_FOLDER = '../Data/' + DATASET
    DATA_FOLDER_val = MAIN_FOLDER + '/valB/images'
    LABEL_FOLDER_val = MAIN_FOLDER + '/valB/labels'
    HEIGHT_FOLDER_val = MAIN_FOLDER + '/valB/heights'
    val_set = datasetval(DATA_FOLDER_val, LABEL_FOLDER_val, HEIGHT_FOLDER_val, args)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=4)

    if args.height:
        input = 5
    else:
        input = 4

    model = smp.Unet("resnet18", 
                    classes=args.num_classes, 
                    in_channels=input,
                    encoder_weights=None)

    for i in range(20):
        resume = (i+1)*5000
        device = torch.device('cuda:{}'.format(str(args.gpu)))
        if args.height:
            path = osp.join(here,'results', SOURCE + '_' + TARGET + '_height_unet', 'checkpoints', 'iter' + str(resume) +'.pth')
        else:
            path = osp.join(here,'results', SOURCE + '_' + TARGET + '_no_height_unet', 'checkpoints', 'iter' + str(resume) +'.pth')
        checkpoint = torch.load(path, map_location=str(device))
        model.load_state_dict(checkpoint)
        print('load trained model from ' + path)  
        model.eval()
        model.to(device)
        val(model, val_loader, args, out_dir, resume, device)

def val(model, dataloader, args, out_dir, resume, device):

    pbar = tqdm(total=len(dataloader), desc="[Val]")
    conf_mat = metrics.ConfMatrix(4)

    for i, batch in enumerate(dataloader):

        if args.height:
            img, gt, height = batch['image'], batch['label'], batch["height"]
            img = torch.cat((img, height[:, None, :, :]), 1)
        else:
            img, gt = batch['image'], batch['label']

        with torch.no_grad():
            pred = model(img.to(device))
            pred = np.argmax(pred.cpu().numpy(), axis=1)
        conf_mat.add_batch(gt, pred)
        pbar.update()

    pbar.set_description("[Val] OA: {:.2f}%".format(conf_mat.get_oa() * 100))
    pbar.close()
    print("OA\t", conf_mat.get_oa())
    print("AA\t", conf_mat.get_aa())
    print("mIoU\t", conf_mat.get_mIoU())

    log_file=osp.join(out_dir, TARGET + '.log')
    message = 'Step: {}\nOA: {:.6f}\nAA: {:.6f}\nmIoU: {:.6f}'.format(
            resume, conf_mat.get_oa(), conf_mat.get_aa(), conf_mat.get_mIoU())
    with open(log_file, "a") as log_file:
        log_file.write('%s\n' % message) 

    return

if __name__ == '__main__':
    main()
