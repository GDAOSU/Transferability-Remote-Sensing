# -*- coding: utf-8 -*-
"""
Modified on Wed Aug 4 12:46 2021

By Guixiang Zhang
"""
import numpy as np
import os
import tifffile

def get_batch_inds(batch_size, idx, N,predict=False):
    """
    Generates an array of indices of length N
    :param batch_size: the size of training batches
    :param idx: data to split into batches
    :param N: Maximum size
    :return batchInds: list of arrays of data of length batch_size
    """
    batchInds = []
    idx0 = 0

    toProcess = True
    while toProcess:
        idx1 = idx0 + batch_size
        if idx1 >= N:
            idx1 = N
            if predict==False:

                idx0 = idx1 - batch_size
            toProcess = False
        batchInds.append(idx[idx0:idx1])
        idx0 = idx1

    return batchInds


def Patch2Img(patches,img_size,overlap=0.5):
    patches=np.squeeze(patches)
    patch_wid = patches[0].shape[2]
    patch_hei = patches[0].shape[1]
    num_class = patches[0].shape[0]
   
    vote=np.zeros((num_class, img_size[0], img_size[1]))
    patch_ranges=calculate_cut_range(img_size, patch_size=[patch_hei,patch_wid],overlap=overlap)
    for id in range(len(patches)):
            patch=patches[id]
            y_s=round(patch_ranges[id][0])
            y_e=round(patch_ranges[id][1])
            x_s=round(patch_ranges[id][2])
            x_e=round(patch_ranges[id][3])
            vote[:, y_s:y_e, x_s:x_e] = vote[:, y_s:y_e, x_s:x_e] + patch
    pred = np.argmax(vote, axis = 0).astype('uint8')
    return pred


def Img2Patch(img, patch_size,overlap_rati):
    patches=[]

    patch_range=calculate_cut_range(img.shape[0:2],patch_size,overlap_rati)
    for id in range(len(patch_range)):
        y_s=round(patch_range[id][0])
        y_e=round(patch_range[id][1])
        x_s=round(patch_range[id][2])
        x_e=round(patch_range[id][3])
        patch=img[y_s:y_e,x_s:x_e,:]
        patches.append(patch)
    return patches


def calculate_cut_range(img_size, patch_size,overlap,pad_edge=1):
    patch_range=[]
    patch_height = patch_size[0]
    patch_width = patch_size[1]
    width_overlap = patch_width * overlap
    height_overlap = patch_height *overlap
    cols=img_size[1]
    rows=img_size[0]
    x_e = 0
    while (x_e < cols):
        y_e=0
        x_s = max(0, x_e - width_overlap)
        x_e = x_s + patch_width
        if (x_e > cols):
            x_e = cols
        if (pad_edge == 1): ## if the last path is not enough, then extent to the inerside.
            x_s = x_e - patch_width
        if (pad_edge == 2):## if the last patch is not enough, then extent to the outside(with black).
            x_s=x_s
        while (y_e < rows):
            y_s = max(0, y_e - height_overlap)
            y_e = y_s + patch_height
            if (y_e > rows):
                y_e = rows
            if (pad_edge == 1): ## if the last path is not enough, then extent to the inerside.
                y_s = y_e - patch_height
            if (pad_edge == 2):## if the last patch is not enough, then extent to the outside(with black).
                y_s=y_s
            patch_range.append([int(y_s),int(y_e),int(x_s),int(x_e)])
    return patch_range


def normalize_image_to_path(img_path,label_path,height_path,path_size,overlap_ratio,extra_input=False, convertLab=False,pad_edge=1,normalize_dsm=1, image_order=0):

    img = tifffile.imread(img_path)
    label = tifffile.imread(label_path)
    dsm = tifffile.imread(height_path)

    imgs = []
    labels = []
    dsms = []
    rows=img.shape[0]
    cols=img.shape[1] 

    patch_ranges=calculate_cut_range([rows,cols], patch_size=path_size,overlap=overlap_ratio)    
    for inds in range(len(patch_ranges)):
        y_s=round(patch_ranges[inds][0])
        y_e=round(patch_ranges[inds][1])
        x_s=round(patch_ranges[inds][2])
        x_e=round(patch_ranges[inds][3])
        img_patch=img[int(y_s):int(y_e),int(x_s):int(x_e),:]
        imgs.append(img_patch)
        label_patch=label[int(y_s):int(y_e),int(x_s):int(x_e)]
        labels.append(label_patch)
        dsm_patch=dsm[int(y_s):int(y_e),int(x_s):int(x_e)]
        dsms.append(dsm_patch)
    return imgs,labels,dsms


def crop_normalized_patch_track(img_folder, label_folder, height_folder, out_folder,path_size,overlap_ratio, end = '.tif'):
    if os.path.exists(out_folder)==0:
        os.makedirs(out_folder)
    sub_img_folder = os.path.join(out_folder,'Track1-MSI')
    if os.path.exists(sub_img_folder)==0:
        os.makedirs(sub_img_folder)
    sub_label_folder = os.path.join(out_folder,'Track1-CLS')
    if os.path.exists(sub_label_folder)==0:
        os.makedirs(sub_label_folder)
    sub_height_folder = os.path.join(out_folder,'Track1-AGL')
    if os.path.exists(sub_height_folder)==0:
        os.makedirs(sub_height_folder)

    img_list = os.listdir(img_folder)
    img_list.sort()
    img_list = [x for x in img_list if os.path.basename(x)[:3] == 'JAX']

    for filename in img_list:   
        file_apx=filename[-3:]
        if file_apx=='jpg' or  file_apx=='png' or  file_apx=='tif':
            img_path = os.path.join(img_folder,filename)
            label_path = os.path.join(label_folder , filename.replace('MSI','CLS'))
            height_path = os.path.join(height_folder , filename.replace('MSI','AGL'))
            
            if os.path.exists(label_path):
                imgs, labels, heights = normalize_image_to_path(img_path,label_path,height_path,path_size,overlap_ratio,convertLab=True,pad_edge=1)
       
                for i in range(len(imgs)):
                    filename_new = filename[:-4]+'_'+str(i)+ end

                    img_write_path=os.path.join(sub_img_folder, filename_new)
                    label_write_path=os.path.join(sub_label_folder, filename_new.replace('MSI','CLS'))
                    height_write_path=os.path.join(sub_height_folder, filename_new.replace('MSI','AGL'))
                          
                    patch_img=imgs[i]               
                    tifffile.imwrite(img_write_path,patch_img)
                    tifffile.imwrite(label_write_path,labels[i])   
                    tifffile.imwrite(height_write_path,heights[i]) 
 

if __name__ == '__main__':

    img_folder=r'S:\GDA\GuixiangZhang\tranferability\transferability analysis\data\DFC2019\Track1-MSI'
    label_folder=r'S:\GDA\GuixiangZhang\tranferability\transferability analysis\data\DFC2019\Track1-CLS'
    height_folder=r'S:\GDA\GuixiangZhang\tranferability\transferability analysis\data\DFC2019\Track1-AGL'
    out_folder=r'S:\GDA\GuixiangZhang\tranferability\transferability analysis\data\DFC2019\JAX512'
    path_size=(512, 512)
    overlap_ratio=0.5
    end = '.tif'

    crop_normalized_patch_track(img_folder,label_folder, height_folder, out_folder,path_size,
                                 overlap_ratio, end = end)
