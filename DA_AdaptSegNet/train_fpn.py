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
from discriminator import FCDiscriminator
from loader.dataset import dataset
from loader.datasetval import datasetval
from loader.dataset_target import dataset_target
import segmentation_models_pytorch as smp


BATCH_SIZE = 2
NUM_CLASSES = 4
SAVE_PRED_EVERY = 5000
NUM_STEPS = 100001
RESUME = 0
HEIGHT = True
DATASET = 'JAX_Haiti'
SOURCE = DATASET.split("_")[0]
TARGET = DATASET.split("_")[1]
GPU = 0
LEARNING_RATE = 0.02
LEARNING_RATE_D = 1e-4
MOMENTUM = 0.9
POWER = 0.9
WEIGHT_DECAY = 0.0001
LAMBDA_ADV_TARGET = 0.001
GAN = 'Vanilla'
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
    parser.add_argument("--lambda-adv-target", type=float, default=LAMBDA_ADV_TARGET,
                        help="lambda_adv for adversarial training.")
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
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
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

def main():
    """Create the model and start the training."""

    # save checkpoint direction
    here = osp.dirname(osp.abspath(__file__))
    if args.height:
        out_dir = osp.join(here,'results', DATASET + '_height_fpn')
    else:
        out_dir = osp.join(here,'results', DATASET + '_no_height_fpn')
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
    train_set = dataset(DATA_FOLDER, LABEL_FOLDER, HEIGHT_FOLDER, args)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size)
    trainloader_iter = iter(train_loader)

    DATA_FOLDER = MAIN_FOLDER + '/trainB/images'
    LABEL_FOLDER = MAIN_FOLDER + '/trainB/labels'
    HEIGHT_FOLDER = MAIN_FOLDER + '/trainB/heights'
    target_set = dataset_target(DATA_FOLDER, LABEL_FOLDER, HEIGHT_FOLDER, args)
    target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size)
    targetloader_iter = iter(target_loader)

    DATA_FOLDER_val = MAIN_FOLDER + '/valB/images'
    LABEL_FOLDER_val = MAIN_FOLDER + '/valB/labels'
    HEIGHT_FOLDER_val = MAIN_FOLDER + '/valB/heights'
    val_set = datasetval(DATA_FOLDER_val, LABEL_FOLDER_val, HEIGHT_FOLDER_val, args)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=4)

    if args.height:
        input = 5
    else:
        input = 4

    # model
    model = smp.FPN("resnet18", 
                    classes=args.num_classes, 
                    in_channels=input,
                    encoder_weights=None)
                    
    # init D
    model_D = FCDiscriminator(num_classes=args.num_classes)

    device = torch.device('cuda:{}'.format(str(args.gpu)))
    if args.height:
        model_path = osp.join(here,'results', DATASET + '_height_fpn', 'checkpoints', 'model_iter' + str(args.resume) +'.pth')
        model_D_path = osp.join(here,'results', DATASET + '_height_fpn', 'checkpoints', 'model_D_iter' + str(args.resume) +'.pth')
    else:
        model_path = osp.join(here,'results', DATASET + '_no_height_fpn', 'checkpoints', 'model_iter' + str(args.resume) +'.pth')
        model_D_path = osp.join(here,'results', DATASET + '_no_height_fpn', 'checkpoints', 'model_D_iter' + str(args.resume) +'.pth')
    i_iter=0
    if args.resume>0:
        i_iter=args.resume+1
        checkpoint = torch.load(model_path, map_location=str(device))
        model.load_state_dict(checkpoint)
        checkpoint = torch.load(model_D_path, map_location=str(device))
        model_D.load_state_dict(checkpoint)
    model.train()
    model_D.train()
    model.to(device)
    model_D.to(device)

    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=4, reduction='mean')
    optimizer = create_optimizers(model, args)
    optimizer.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif args.gan == 'LS':
        bce_loss = torch.nn.MSELoss()

    # labels for adversarial training
    source_label = 0
    target_label = 1

    max_iter=args.num_steps
    train_loss=[]
    train_acc=[]
    target_acc=[]

    while i_iter < max_iter:

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        # train G
        # don't accumulate grads in D
        for param in model_D.parameters():
            param.requires_grad = False

        # train with source
        batch = trainloader_iter.next()
        if args.height:
            images, labels, heights = batch["image"], batch["label"], batch["height"]
            images = torch.cat((images, heights[:, None, :, :]), 1)
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
        else:
            images, labels = batch['image'], batch['label']
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

        pred1 = model(images)
        loss_seg = loss_fn(pred1, labels)
        loss_seg.backward()

        loss_data = loss_seg.data.item()
        pred = np.argmax(pred1.data.cpu().numpy()[0], axis=0)
        gt = labels[0].data.cpu().numpy()
        train_acc.append(accuracy(pred,gt))
        train_loss.append(loss_data)

        # train with target
        batch = targetloader_iter.next()

        if args.height:
            images, labels, heights = batch["image"], batch["label"], batch["height"]
            images = torch.cat((images, heights[:, None, :, :]), 1)
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
        else:
            images, labels = batch['image'], batch['label']
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

        pred_target = model(images)
        D_out = model_D(F.softmax(pred_target,dim=1))
        loss_adv_target = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).to(device))
        loss = args.lambda_adv_target * loss_adv_target
        loss.backward()

        pred = np.argmax(pred_target.data.cpu().numpy()[0], axis=0)
        gt = labels[0].data.cpu().numpy()
        target_acc.append(accuracy(pred,gt))

        # train D
        # bring back requires_grad
        for param in model_D.parameters():
            param.requires_grad = True

        # train with source
        pred1 = pred1.detach()
        D_out = model_D(F.softmax(pred1,dim=1))
        loss_D = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).to(device))
        loss_D = loss_D / 2
        loss_D.backward()

        # train with target
        pred_target = pred_target.detach()
        D_out = model_D(F.softmax(pred_target,dim=1))
        loss_D = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(target_label)).to(device))
        loss_D = loss_D / 2
        loss_D.backward()

        optimizer.step()
        optimizer_D.step()

        log_file=osp.join(out_dir, 'training.log')
        if i_iter%500==0:
            print('Train [{}/{} Source loss:{:.6f} Source acc:{:.4f}% Target acc:{:.4f}%]'.format(
                i_iter, args.num_steps, sum(train_loss)/len(train_loss), sum(train_acc)/len(train_acc), sum(target_acc)/len(target_acc)))

            message = 'Train  [{}/{} Source loss:{:.6f} Source acc:{:.4f}% Target acc:{:.4f}%]'.format(
                i_iter, args.num_steps, sum(train_loss)/len(train_loss), sum(train_acc)/len(train_acc), sum(target_acc)/len(target_acc))

            with open(log_file, "a") as log_file:
                log_file.write('%s\n' % message)  
            train_loss=[]
            train_acc=[]
            target_acc=[]
            
        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('saving checkpoint.....')
            val(model, val_loader, args, out_dir, i_iter, device)
            torch.save(model.state_dict(), osp.join(checkpoint_dir,'model_iter{}.pth'.format(i_iter)))
            torch.save(model_D.state_dict(), osp.join(checkpoint_dir,'model_D_iter{}.pth'.format(i_iter)))

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

