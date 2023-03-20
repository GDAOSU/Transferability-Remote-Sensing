import os
import os.path as osp
import pickle
import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm

# Options
from options.test_options import TestOptions
import torch
from modeling.scalenet_deeplab import *
from loader.datasetval import datasetval

LABELS = ["GROUND", "TREES", "BUILDING", "WATER"] 
N_CLASS = len(LABELS) 


def test(net, dataloader, args):
  
    # Switch the network to inference mode
    net.cuda()
    net.eval()
    import metrics
    pbar = tqdm(total=len(dataloader), desc="[Val]")
    conf_mat = metrics.ConfMatrix(N_CLASS)
    
    for i, batch in enumerate(dataloader):

        if args.height:
            img, gt, height, id = batch['image'], batch['label'], batch["height"], batch['id']
            img = torch.cat((img, height[:, None, :, :]), 1)
        else:
            img, gt, id = batch['image'], batch['label'], batch['id']

        with torch.no_grad():
            pred,_=net(img.cuda())
            pred = np.argmax(pred.cpu().numpy(), axis=1)
        conf_mat.add_batch(gt, pred)

        # update progressbar
        pbar.update()
        for i in range(len(id)):
            name = id[i]
            output_file=osp.join(args.out,name)
            output = pred[i].astype(np.uint8)
            output_img = Image.fromarray(output)
            output_img.save(output_file)

    # close progressbar
    pbar.set_description("[Val] OA: {:.2f}%".format(conf_mat.get_oa() * 100))
    pbar.close()
    accur = {}
    accur['CM'] = conf_mat.state
    accur['OA'] = conf_mat.get_oa()
    accur['AA'] = conf_mat.get_aa()
    accur['mIoU'] = conf_mat.get_mIoU()   
    with open(os.path.join(args.out, 'confusion_matrix.pkl'), 'wb') as f:
        pickle.dump(accur, f, pickle.HIGHEST_PROTOCOL)
    print("OA\t", conf_mat.get_oa())
    print("AA\t", conf_mat.get_aa())
    print("mIoU\t", conf_mat.get_mIoU())

def main():

    testOpts= TestOptions()
    args=testOpts.get_arguments()
    here = osp.dirname(osp.abspath(__file__))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    MAIN_FOLDER = r'../Data/JAX_London'
    DATA_FOLDER_val = MAIN_FOLDER + '/valB/images'
    LABEL_FOLDER_val = MAIN_FOLDER + '/valB/labels'
    HEIGHT_FOLDER_val = MAIN_FOLDER + '/valB/heights'
    val_set = datasetval(DATA_FOLDER_val, LABEL_FOLDER_val, HEIGHT_FOLDER_val)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1)

    model = DeepLabCA(num_classes=N_CLASS,
                        output_stride=args.out_stride,
                        freeze_bn=args.freeze_bn)

    if args.resume:
        checkpoint = torch.load(r'./results/deeplabv3+_JAX_07302022__153204/checkpoints/model_iter9000.pth')
        model.load_state_dict(checkpoint)

    now = datetime.datetime.now()
    folder_name=args.model+'_'+args.dataset
    args.out = osp.join(here,'test_results', folder_name+'_'+now.strftime('%Y%m%d__%H%M%S'))

    if not osp.isdir(args.out):
        os.makedirs(args.out)

    test(model, val_loader, args)

if __name__=='__main__':
    main()





