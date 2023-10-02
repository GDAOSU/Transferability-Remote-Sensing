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
from typing import Optional, Union, List
from loader.datasetval import datasetval
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base.heads import SegmentationHead


NUM_CLASSES = 4
HEIGHT = [True]
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
    return parser.parse_args()

args = get_arguments()

class Unet_CLAN(smp.Unet):
    def __init__(self,encoder_name,encoder_weights,in_channels,classes, 
    encoder_depth: int = 5,
    decoder_channels: List[int] = (256, 128, 64, 32, 16),
    decoder_use_batchnorm: bool = True,
    decoder_attention_type: Optional[str] = None,
    activation: Optional[Union[str, callable]] = None):

        super().__init__(encoder_name=encoder_name,encoder_weights=encoder_weights,in_channels=in_channels,classes=classes)

        self.decoder_2 = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head_2 = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)

        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)

        decoder_output_2 = self.decoder_2(*features)
        masks_2 = self.segmentation_head_2(decoder_output_2)

        return masks, masks_2

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

                    if source == 'JAX':
                        if target == 'JAX' or target == 'London':
                            continue     
                    if source == 'JAX' and target == 'OMA' and dataset == 'Baseline':
                        continue  

                    # dataloader
                    MAIN_FOLDER = '../Data/' + target + '_' + target
                    DATA_FOLDER_val = MAIN_FOLDER + '/valB/images'
                    HEIGHT_FOLDER_val = MAIN_FOLDER + '/valB/heights'
                    val_set = datasetval(DATA_FOLDER_val, HEIGHT_FOLDER_val, target)
                    val_loader = torch.utils.data.DataLoader(val_set, batch_size=4)

                    if height:
                        input = 5
                    else:
                        input = 4

                    if dataset == 'DA_CLAN':
                        model = Unet_CLAN("resnet18", 
                                    classes=args.num_classes,    
                                    in_channels=input,
                                    encoder_weights=None)
                    else:
                        model = smp.Unet("resnet18", 
                                    classes=args.num_classes, 
                                    in_channels=input,
                                    encoder_weights=None)

                    device = torch.device('cuda:{}'.format(str(args.gpu)))

                    if dataset == 'Baseline':
                        if height:
                            path = osp.join(root, dataset, 'results', source + '_' + source, source + '_' + source + '_height_unet', source + '_' + target)
                        else:
                            path = osp.join(root, dataset, 'results', source + '_' + source, source + '_' + source + '_no_height_unet', source + '_' + target)
                    else:
                        if height:
                            path = osp.join(root, dataset, 'results', source + '_' + target,  source + '_' + target + '_height_unet', 'checkpoints')
                        else:
                            path = osp.join(root, dataset, 'results', source + '_' + target,  source + '_' + target + '_no_height_unet', 'checkpoints')

                    path_check = glob.glob(os.path.join(path, '*.pth'))
                    checkpoint = torch.load(path_check[0], map_location=str(device))

                    model.load_state_dict(checkpoint)
                    print('load trained model from ' + path)  
                    model.eval()
                    model.to(device)
                    out_dir = './records'
                    if height:
                        filename = dataset + '_' + source + '_' + target + '_height_unet.log'
                    else:
                        filename = dataset + '_' + source + '_' + target + '_no_height_unet.log'

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
