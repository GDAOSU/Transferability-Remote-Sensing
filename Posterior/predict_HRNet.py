import os
import os.path as osp
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
import glob
from loader.datasetval import datasetval
from models.hrnet import get_seg_model
from models_CLAN.hrnet import get_seg_model_CLAN
from models.config import cfg
import torch.nn.functional as F


NUM_CLASSES = 4
HEIGHT = [False,True]
DATASET = ['Baseline','DA_AdaptSegNet','DA_CLAN','DA_ScaleAware']
SOURCE = ['JAX','London','OMA','Haiti']
TARGET = ['JAX','London','OMA','Haiti']
GPU = 1
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
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--gpu", type=int, default=GPU,
                        help="choose gpu device.")
    parser.add_argument(
        "--cfg",
        default="models/HRNet.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    return parser.parse_args()

args = get_arguments()
cfg.merge_from_file(args.cfg)

def main():

    root = r'/research/GDA/GuixiangZhang/transferability'
    # root = r'S:/GDA/GuixiangZhang/transferability'
    for height in HEIGHT:
        for source in SOURCE:
            for target in TARGET:
                for dataset in DATASET:

                    if dataset != 'Baseline':
                        if source == target or source == 'OMA' or source == 'Haiti':
                            continue
                    else:
                        if source == 'OMA' or source == 'Haiti':
                            if source != target:
                                continue   

                    # dataloader
                    MAIN_FOLDER = '../Data/' + target + '_' + target
                    DATA_FOLDER_val = MAIN_FOLDER + '/valB/images'
                    HEIGHT_FOLDER_val = MAIN_FOLDER + '/valB/heights'
                    val_set = datasetval(DATA_FOLDER_val, HEIGHT_FOLDER_val, target)
                    val_loader = torch.utils.data.DataLoader(val_set, batch_size=4)

                    if height:
                        cfg.MODEL.EXTRA['INPUT'] = 5
                    else:
                        cfg.MODEL.EXTRA['INPUT'] = 4

                    if dataset == 'DA_CLAN':
                        model = get_seg_model_CLAN(cfg)
                    else:
                        model = get_seg_model(cfg)

                    device = torch.device('cuda:{}'.format(str(args.gpu)))

                    if dataset == 'Baseline':
                        if height:
                            path = osp.join(root, dataset, 'results', source + '_' + source, source + '_' + source + '_height_hrnet', source + '_' + target)
                        else:
                            path = osp.join(root, dataset, 'results', source + '_' + source, source + '_' + source + '_no_height_hrnet', source + '_' + target)

                    else:
                        if height:
                            path = osp.join(root, dataset, 'results', source + '_' + target,  source + '_' + target + '_height_hrnet', 'checkpoints')
                        else:
                            path = osp.join(root, dataset, 'results', source + '_' + target,  source + '_' + target + '_no_height_hrnet', 'checkpoints')

                    path_check = glob.glob(os.path.join(path, '*.pth'))
                    checkpoint = torch.load(path_check[0], map_location=str(device))

                    model.load_state_dict(checkpoint)
                    print('load trained model from ' + path)  
                    model.eval()
                    model.to(device)
                    out_dir = './records'
                    if height:
                        filename = dataset + '_' + source + '_' + target + '_height_hrnet.log'
                    else:
                        filename = dataset + '_' + source + '_' + target + '_no_height_hrnet.log'
                    val(model, val_loader, height, out_dir, filename, device, dataset)

def val(model, dataloader, use_height, out_dir, filename, device, dataset):

    pbar = tqdm(total=len(dataloader), desc="[Val]")

    posterior = []
    for i, batch in enumerate(dataloader):

        if use_height:
            img, height = batch['image'], batch["height"]
            img = torch.cat((img, height[:, None, :, :]), 1)
        else:
            img = batch['image']

        with torch.no_grad():

            if dataset == 'DA_CLAN':
                pred1, pred2 = model(img.to(device)) 
                pred = pred1 + pred2
            else:
                pred = model(img.to(device)) 
                ph, pw = pred.size(2), pred.size(3)
                if ph != 512 or pw != 512:
                    score = F.interpolate(
                            input=pred, size=(512, 512), mode='nearest')
                pred = score

            pred = F.softmax(pred,dim=1)
            pred = pred.max(1)[0].mean()
            pred = pred.cpu().numpy()

        posterior.append(pred)
        pbar.update()

    mean_posterior = np.mean(np.array(posterior))
    log_file=os.path.join(out_dir, filename)
    message = 'mean_posterior: {:.6f}\n'.format(mean_posterior)
    with open(log_file, "a") as log_file:
        log_file.write('%s\n' % message)  

    return


if __name__ == '__main__':
    main()
