import os
import os.path as osp
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
import pandas as pd
import metrics
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
                    LABEL_FOLDER_val = MAIN_FOLDER + '/valB/labels'
                    HEIGHT_FOLDER_val = MAIN_FOLDER + '/valB/heights'
                    INDEX_FOLDER_val = './index-based label/' + target
                    val_set = datasetval(DATA_FOLDER_val, LABEL_FOLDER_val, HEIGHT_FOLDER_val, INDEX_FOLDER_val, target)
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

                    # if dataset == 'Baseline':
                    #     if height: 
                    #         if 'conv1_height.weight' in checkpoint:               
                    #             checkpoint.pop('conv1.weight')
                    #             checkpoint['conv1.weight'] = checkpoint.pop('conv1_height.weight')
                    #     else:
                    #         if 'conv1_height.weight' in checkpoint:
                    #             checkpoint.pop('conv1_height.weight')

                    model.load_state_dict(checkpoint)
                    print('load trained model from ' + path)  
                    model.eval()
                    model.to(device)
                    out_dir = './records'
                    if height:
                        filename = dataset + '_' + source + '_' + target + '_height_hrnet.csv'
                    else:
                        filename = dataset + '_' + source + '_' + target + '_no_height_hrnet.csv'
                    val(model, val_loader, height, out_dir, filename, device, dataset)

def val(model, dataloader, use_height, out_dir, filename, device, dataset):

    pbar = tqdm(total=len(dataloader), desc="[Val]")
    conf_mat1 = metrics.ConfMatrix(4)
    conf_mat2 = metrics.ConfMatrix(4)
    id_list = []
    for i, batch in enumerate(dataloader):

        if use_height:
            img, gt, index, height = batch['image'], batch['label'], batch['index'], batch["height"]
            img = torch.cat((img, height[:, None, :, :]), 1)
        else:
            img, gt, index = batch['image'], batch['label'], batch['index']

        id = batch["id"]
        id_list.append(id)

        with torch.no_grad():

            if dataset == 'DA_CLAN':
                pred1, pred2 = model(img.to(device)) 
                pred = np.argmax((pred1 + pred2).cpu().numpy(), axis=1)
            else:
                pred = model(img.to(device)) 
                ph, pw = pred.size(2), pred.size(3)
                h, w = gt.size(1), gt.size(2)
                if ph != h or pw != w:
                    score = F.interpolate(
                            input=pred, size=(h, w), mode='nearest')
                pred = np.argmax(score.cpu().numpy(), axis=1)

        conf_mat1.add_batch(index, pred)
        conf_mat2.add_batch(gt, pred)
        pbar.update()

    recalls = np.zeros((3, 3))
    precisions = np.zeros((3, 3))
    CMs = []
    CMs.append(conf_mat1.state)
    CMs.append(conf_mat2.state)

    for p in range(2): #metric kind
        for q in range(3): #class    
            recalls[q, p] = CMs[p][1+q,1+q]/CMs[p][1+q,:].sum()        
            precisions[q, p] = CMs[p][1+q,1+q]/CMs[p][:,1+q].sum() 

    df = pd.DataFrame({'id': id_list,
                        'index_OA': conf_mat1.get_oa(),
                        'accur_OA': conf_mat2.get_oa(),
                        'index_AA': conf_mat1.get_aa(),
                        'accur_AA': conf_mat2.get_aa(),                    
                        'index_mIoU': conf_mat1.get_mIoU(),
                        'accur_mIoU': conf_mat2.get_mIoU(),                    
                        'index_tree_precision':precisions[0,0],
                        'index_tree_recall':recalls[0,0],
                        'index_building_precision':precisions[1,0],
                        'index_building_recall':recalls[1,0],
                        'index_water_precision':precisions[2,0],
                        'index_water_recall':recalls[2,0],   
                        'accur_tree_precision':precisions[0,1],
                        'accur_tree_recall':recalls[0,1],
                        'accur_building_precision':precisions[1,1],
                        'accur_building_recall':recalls[1,1],
                        'accur_water_precision':precisions[2,1],
                        'accur_water_recall':recalls[2,1],   	                    
                        })
    df.to_csv(os.path.join(out_dir, filename), index = False)
    return


if __name__ == '__main__':
    main()
