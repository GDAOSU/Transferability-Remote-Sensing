import os
import os.path as osp
import numpy as np
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

import metrics
from loader.dataset import dataset
from loader.dataset_multiscale import dataset_multi
from loader.datasetval import datasetval
from modeling.discriminator import FCDiscriminator
import segmentation_models_pytorch as smp
from modeling.attention import CAM_Module

BATCH_SIZE = 2
NUM_CLASSES = 4
SAVE_PRED_EVERY = 5000
NUM_STEPS = 100001
RESUME = 0
HEIGHT = True
DATASET = 'JAX_OMA'
SOURCE = DATASET.split("_")[0]
TARGET = DATASET.split("_")[1]

LEARNING_RATE = 0.02
LEARNING_RATE_D = 1e-4
MOMENTUM = 0.9
POWER = 0.9
WEIGHT_DECAY = 0.0001
GPU = 0
SEED = 2021

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
cudnn.benchmark = False
cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.cuda.manual_seed_all(SEED)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")

    parser.add_argument("--source", type=str, default=SOURCE)
    parser.add_argument("--target", type=str, default=TARGET)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--resume", type=int, default=RESUME,
                        help="restart the training or not")
    parser.add_argument("--height", type=bool, default=HEIGHT,
                        help="Use height or not")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=GPU,
                        help="choose gpu device.")
    parser.add_argument('--lambda-adv-domain', type=float, default=0.005, help='weight for domain adversarial loss')
    parser.add_argument('--lambda-adv-scale', type=float, default=0.005, help='weight for scale adversarial loss')
    return parser.parse_args()

args = get_arguments()

def accuracy(pred,gt):
    return 100*float(np.count_nonzero(pred==gt)/gt.size)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, CAM_Module):
            group_no_decay.append(m.gamma)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

def create_optimizers(nets, args):
    train_params = group_weight(nets)
    optimizer = torch.optim.SGD(
        train_params,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    return optimizer

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

def main():
    """Create the model and start the training."""

    # save checkpoint direction
    here = osp.dirname(osp.abspath(__file__))
    if args.height:
        out_dir = osp.join(here,'results', DATASET + '_height_attention')
    else:
        out_dir = osp.join(here,'results', DATASET + '_no_height_attention')
    if not osp.isdir(out_dir):
        os.makedirs(out_dir) 
    checkpoint_dir=osp.join(out_dir,'checkpoints')
    if not osp.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir) 

    # dataloader
    MAIN_FOLDER = '../Data/data_DA' + DATASET
    DATA_FOLDER = MAIN_FOLDER + '/trainA/images'
    LABEL_FOLDER = MAIN_FOLDER + '/trainA/labels'
    HEIGHT_FOLDER = MAIN_FOLDER + '/trainA/heights'
    train_set = dataset(DATA_FOLDER, LABEL_FOLDER, HEIGHT_FOLDER, args=args)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size)
    trainloader_iter = iter(train_loader)

    DATA_FOLDER1 = MAIN_FOLDER + '/trainB/images'
    LABEL_FOLDER1 = MAIN_FOLDER + '/trainB/labels'
    HEIGHT_FOLDER1 = MAIN_FOLDER + '/trainB/heights'
    DATA_FOLDER2 = DATA_FOLDER1
    LABEL_FOLDER2 = LABEL_FOLDER1
    HEIGHT_FOLDER2 = HEIGHT_FOLDER1
    target_set = dataset_multi(DATA_FOLDER1, LABEL_FOLDER1, HEIGHT_FOLDER1, DATA_FOLDER2, LABEL_FOLDER2, HEIGHT_FOLDER2, args=args)
    target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size)
    targetloader_iter = iter(target_loader)

    DATA_FOLDER_val = MAIN_FOLDER + '/valB/images'
    LABEL_FOLDER_val = MAIN_FOLDER + '/valB/labels'
    HEIGHT_FOLDER_val = MAIN_FOLDER + '/valB/heights'
    val_set = datasetval(DATA_FOLDER_val, LABEL_FOLDER_val, HEIGHT_FOLDER_val, args=args)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=4)

    if args.height:
        input = 5
    else:
        input = 4

    # model
    model = DeepLabV3Plus_attention(encoder_name="resnet18",
                classes=args.num_classes, 
                in_channels=input,
                encoder_weights=None)
    netD_domain = FCDiscriminator(num_classes=args.num_classes)
    netD_scale = FCDiscriminator(num_classes=args.num_classes)

    device = torch.device('cuda:{}'.format(str(args.gpu)))
    if args.height:
        model_path = osp.join(here,'results', DATASET + '_height_attention', 'checkpoints', 'model_iter' + str(args.resume) +'.pth')
        netD_domain_path = osp.join(here,'results', DATASET + '_height_attention', 'checkpoints', 'netD_domain_iter' + str(args.resume) +'.pth')
        netD_scale_path = osp.join(here,'results', DATASET + '_height_attention', 'checkpoints', 'netD_scale_iter' + str(args.resume) +'.pth')
    else:
        model_path = osp.join(here,'results', DATASET + '_no_height_attention', 'checkpoints', 'model_iter' + str(args.resume) +'.pth')
        netD_domain_path = osp.join(here,'results', DATASET + '_no_height_attention', 'checkpoints', 'netD_domain_iter' + str(args.resume) +'.pth')
        netD_scale_path = osp.join(here,'results', DATASET + '_no_height_attention', 'checkpoints', 'netD_scale_iter' + str(args.resume) +'.pth')
    i_iter=0
    if args.resume>0:
        i_iter=args.resume+1
        checkpoint = torch.load(model_path, map_location=str(device))
        model.load_state_dict(checkpoint)
        checkpoint = torch.load(netD_domain_path, map_location=str(device))
        netD_domain.load_state_dict(checkpoint)
        checkpoint = torch.load(netD_scale_path, map_location=str(device))
        netD_scale.load_state_dict(checkpoint)
    model.train()
    netD_domain.train()
    netD_scale.train()
    model.to(device)
    netD_domain.to(device)
    netD_scale.to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=args.num_classes, reduction='mean')
    optim_netG = create_optimizers(model, args)
    optim_netG.zero_grad()
    optim_netD_domain = optim.Adam(netD_domain.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optim_netD_scale = optim.Adam(netD_scale.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optim_netD_domain.zero_grad()
    optim_netD_scale.zero_grad()
        
    bce_loss = torch.nn.BCEWithLogitsLoss()

    max_iter=args.num_steps

    source_label = 0
    target_label = 1

    source_scale_label=0
    target_scale_label=1

    train_loss=[]
    train_acc=[]
    target_acc_s1=[]
    target_acc_s2=[]

    while i_iter < max_iter:

        optim_netG.zero_grad()
        adjust_learning_rate(optim_netG, i_iter)
        optim_netD_domain.zero_grad()
        optim_netD_scale.zero_grad()
        adjust_learning_rate_D(optim_netD_domain, i_iter)
        adjust_learning_rate_D(optim_netD_scale, i_iter)

        for param in netD_domain.parameters():
            param.requires_grad = False

        for param in netD_scale.parameters():
            param.requires_grad = False

        batch = trainloader_iter.next()
        if args.height:
            im_s, label_s, height_s = batch["image"], batch["label"], batch["height"]
            im_s = torch.cat((im_s, height_s[:, None, :, :]), 1)
            im_s = Variable(im_s).to(device)
            label_s = Variable(label_s).to(device)
        else:
            im_s, label_s = batch["image"], batch["label"]
            im_s = Variable(im_s).to(device)
            label_s = Variable(label_s).to(device)

        batch = targetloader_iter.next()
        if args.height:
            im_t_s1, label_t_s1, height_t_s1, im_t_s2, label_t_s2, height_t_s2 = batch
            im_t_s1 = torch.cat((im_t_s1, height_t_s1[:, None, :, :]), 1)
            im_t_s2 = torch.cat((im_t_s2, height_t_s2[:, None, :, :]), 1)
            im_t_s1 = Variable(im_t_s1).to(device)
            im_t_s2 = Variable(im_t_s2).to(device)
            label_t_s1 = Variable(label_t_s1).to(device)
            label_t_s2 = Variable(label_t_s2).to(device)
        else:
            im_t_s1, label_t_s1, height_t_s1, im_t_s2, label_t_s2, height_t_s2 = batch # s2 source scale
            im_t_s1 = Variable(im_t_s1).to(device)
            im_t_s2 = Variable(im_t_s2).to(device)
            label_t_s1 = Variable(label_t_s1).to(device)
            label_t_s2 = Variable(label_t_s2).to(device)

        ############
        #TRAIN NETG#
        ############
        #train with source 
        #optimize segmentation network with source data

        pred_seg = model(im_s)
        seg_loss = loss_fn(pred_seg, label_s)
        seg_loss.backward()

        loss_data = seg_loss.data.item()
        pred = np.argmax(pred_seg.data.cpu().numpy()[0], axis=0)
        gt = label_s.data.cpu().numpy()[0]

        train_acc.append(accuracy(pred,gt))
        train_loss.append(loss_data)

        #train with target
        pred_s1 = model(im_t_s1)
        pred = np.argmax(pred_s1.data.cpu().numpy()[0], axis=0)
        gt = label_t_s1.data.cpu().numpy()[0]
        target_acc_s1.append(accuracy(pred,gt))

        pred_s2 = model(im_t_s2)
        pred = np.argmax(pred_s2.data.cpu().numpy()[0], axis=0)
        gt = label_t_s2.data.cpu().numpy()[0]
        target_acc_s2.append(accuracy(pred,gt))

        pred_d=netD_domain(F.softmax(pred_s2,dim=1))
        pred_s=netD_scale(F.softmax(pred_s1,dim=1))

        loss_adv_domain = bce_loss(pred_d,Variable(torch.FloatTensor(pred_d.data.size()).fill_(source_label)).to(device)) 
        loss_adv_scale = bce_loss(pred_s,Variable(torch.FloatTensor(pred_s.data.size()).fill_(source_scale_label)).to(device))

        loss=args.lambda_adv_domain*loss_adv_domain+args.lambda_adv_scale*loss_adv_scale
        loss /= len(im_t_s1)
        loss.backward()

        ############
        #TRAIN NETD#
        ############

        for param in netD_domain.parameters():
            param.requires_grad = True

        for param in netD_scale.parameters():
            param.requires_grad = True

        #train with source domain and source scale
        pred_seg,pred_s2=pred_seg.detach(),pred_s2.detach()
        pred_d=netD_domain(F.softmax(pred_seg,dim=1))
        pred_s=netD_scale(F.softmax(pred_s2,dim=1))

        loss_D_domain = bce_loss(pred_d,Variable(torch.FloatTensor(pred_d.data.size()).fill_(source_label)).to(device))
        loss_D_scale = bce_loss(pred_s,Variable(torch.FloatTensor(pred_s.data.size()).fill_(source_scale_label)).to(device))

        loss_D_domain=loss_D_domain/len(im_s)/2
        loss_D_scale=loss_D_scale/len(im_s)/2

        loss_D_domain.backward()
        loss_D_scale.backward()

        #train with target domain and target scale
        pred_s1,pred_s2=pred_s1.detach(),pred_s2.detach()
        pred_d=netD_domain(F.softmax(pred_s2,dim=1))
        pred_s=netD_scale(F.softmax(pred_s1,dim=1))

        loss_D_domain = bce_loss(pred_d,Variable(torch.FloatTensor(pred_d.data.size()).fill_(target_label)).to(device))
        loss_D_scale = bce_loss(pred_s,Variable(torch.FloatTensor(pred_s.data.size()).fill_(target_scale_label)).to(device))

        loss_D_domain=loss_D_domain/len(im_s)/2
        loss_D_scale=loss_D_scale/len(im_s)/2

        loss_D_domain.backward()
        loss_D_scale.backward()

        optim_netG.step()
        optim_netD_domain.step()
        optim_netD_scale.step()

        log_file=osp.join(out_dir, 'training.log')

        if i_iter % 500 == 0:
            print('Train [{}/{} Source loss:{:.6f} acc:{:.4f} % Target s1 acc:{:4f}% Target s2 acc:{:4f}%]'.format(
                i_iter, max_iter, sum(train_loss)/len(train_loss),sum(train_acc)/len(train_acc),
                sum(target_acc_s1)/len(target_acc_s1),sum(target_acc_s2)/len(target_acc_s2)))
        
            message = 'Train  [{}/{} Source loss:{:.6f} acc:{:.4f} % Target s1 acc:{:4f}% Target s2 acc:{:4f}%]'.format(
                    i_iter, max_iter, sum(train_loss)/len(train_loss),sum(train_acc)/len(train_acc),
                    sum(target_acc_s1)/len(target_acc_s1),sum(target_acc_s2)/len(target_acc_s2))

            with open(log_file, "a") as log_file:
                log_file.write('%s\n' % message)  # save the message

            train_loss=[]
            train_acc=[]
            target_acc_s1=[]
            target_acc_s2=[]

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('saving checkpoint.....')
            torch.save(model.state_dict(), osp.join(checkpoint_dir,'model_iter{}.pth'.format(i_iter)))
            torch.save(netD_domain.state_dict(), osp.join(checkpoint_dir,'netD_domain_iter{}.pth'.format(i_iter)))
            torch.save(netD_scale.state_dict(), osp.join(checkpoint_dir,'netD_scale_iter{}.pth'.format(i_iter)))
            val(model, val_loader, args, out_dir, i_iter, device)

        i_iter +=1

def val(model, dataloader, args, out_dir, i_iter, device):

    model.eval()
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

    log_file=osp.join(out_dir, 'training.log')
    message = 'Step: {}/{}\nOA: {:.6f}\nAA: {:.6f}\nmIoU: {:.6f}'.format(
            i_iter, args.num_steps, conf_mat.get_oa(), conf_mat.get_aa(), conf_mat.get_mIoU())
    with open(log_file, "a") as log_file:
        log_file.write('%s\n' % message) 

    model.train()
    return

if __name__ == '__main__':
    main()