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
import segmentation_models_pytorch as smp
from attention import CAM_Module
from typing import Optional
from segmentation_models_pytorch.deeplabv3.decoder import DeepLabV3PlusDecoder
from segmentation_models_pytorch.base.heads import SegmentationHead

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
    return parser.parse_args()

args = get_arguments()

class DeepLabV3Plus_attention(smp.DeepLabV3Plus):
    def __init__(self,encoder_name,encoder_weights,in_channels,classes):
        super().__init__(encoder_name=encoder_name,encoder_weights=encoder_weights,in_channels=in_channels,classes=classes)
        self.attention = CAM_Module()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        x,_=self.attention(decoder_output)
        masks = self.segmentation_head(x)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

class DeepLabV3Plus_CLAN_attention(smp.DeepLabV3Plus):
    def __init__(self,encoder_name,encoder_weights,in_channels,classes, 
    decoder_channels: int = 256, 
    decoder_atrous_rates: tuple = (12, 24, 36),
    encoder_output_stride: int = 16,
    activation: Optional[str] = None,
    upsampling: int = 4):
    
        super().__init__(encoder_name=encoder_name,encoder_weights=encoder_weights,in_channels=in_channels,classes=classes)
        self.decoder_2 = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )

        self.segmentation_head_2 = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )
        self.attention = CAM_Module()
        self.attention_2 = CAM_Module()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)

        decoder_output = self.decoder(*features)
        x,_=self.attention(decoder_output)
        masks = self.segmentation_head(x)

        decoder_output_2 = self.decoder_2(*features)
        x_2,_=self.attention_2(decoder_output_2)
        masks_2 = self.segmentation_head_2(x_2)

        return masks, masks_2
        
def main():

    root = r'/research/GDA/GuixiangZhang/transferability'
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
                        input = 5
                    else:
                        input = 4

                    if dataset == 'DA_CLAN':
                        model =DeepLabV3Plus_CLAN_attention("resnet18", 
                                    classes=args.num_classes,    
                                    in_channels=input,
                                    encoder_weights=None)
                    else:
                        model = DeepLabV3Plus_attention("resnet18",
                                    classes=args.num_classes, 
                                    in_channels=input,
                                    encoder_weights=None)

                    device = torch.device('cuda:{}'.format(str(args.gpu)))

                    if dataset == 'Baseline':
                        if height:
                            path = osp.join(root, dataset, 'results', source + '_' + source, source + '_' + source + '_height_attention', source + '_' + target)
                        else:
                            path = osp.join(root, dataset, 'results', source + '_' + source, source + '_' + source + '_no_height_attention', source + '_' + target)
                    else:
                        if height:
                            path = osp.join(root, dataset, 'results', source + '_' + target,  source + '_' + target + '_height_attention', 'checkpoints')
                        else:
                            path = osp.join(root, dataset, 'results', source + '_' + target,  source + '_' + target + '_no_height_attention', 'checkpoints')

                    path_check = glob.glob(os.path.join(path, '*.pth'))
                    checkpoint = torch.load(path_check[0], map_location=str(device))

                    model.load_state_dict(checkpoint)
                    print('load trained model from ' + path)  
                    model.eval()
                    model.to(device)
                    out_dir = './records'
                    if height:
                        filename = dataset + '_' + source + '_' + target + '_height_attention.csv'
                    else:
                        filename = dataset + '_' + source + '_' + target + '_no_height_attention.csv'
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
                pred = np.argmax(pred.cpu().numpy(), axis=1)
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
