import os
import random
import numpy as np
import tifffile
import shutil

def odgt(img_path, label_folder, img_end, label_end):
    seg_path = img_path.replace(os.path.split(img_path)[0], label_folder)
    seg_path = seg_path.replace(img_end, label_end)
    if os.path.exists(seg_path):
        img = tifffile.imread(img_path)
        h, w, _ = img.shape
        odgt_dic = {}
        odgt_dic["fpath_img"] = img_path
        odgt_dic["fpath_segm"] = seg_path
        odgt_dic["width"] = h
        odgt_dic["height"] = w
        return odgt_dic
    else:
        print('the corresponded annotation does not exist')
        print(img_path)
        return None


if __name__ == "__main__":

    input_folder = r'/research/GDA/GuixiangZhang/transferability/transferability analysis/data/AHS/Haiti/Haiti512/Track1-MSI'
    label_folder = r'/research/GDA/GuixiangZhang/transferability/transferability analysis/data/AHS/Haiti/Haiti512/Track1-CLS'   
    height_folder = r'/research/GDA/GuixiangZhang/transferability/transferability analysis/data/AHS/Haiti/Haiti512/Track1-AGL'

    img_list = os.listdir(input_folder)
    img_list = [os.path.join(input_folder, img) for img in img_list]

    source_num = np.uint32(np.round(len(img_list) * 0.5))
    random.seed(2022); source_list = random.sample(img_list, source_num)
    target_list = list(set(img_list) - set(source_list))

    target_train_num = np.uint32(np.round(len(target_list) * 0.6))
    random.seed(2022); target_train_list = random.sample(target_list, target_train_num)
    target_val_list = list(set(target_list) - set(target_train_list))


    # input_folder = r'/research/GDA/GuixiangZhang/transferability/transferability analysis/data/DFC2019/JAX512/Track1-MSI'
    # label_folder = r'/research/GDA/GuixiangZhang/transferability/transferability analysis/data/DFC2019/JAX512/Track1-CLS'   
    # height_folder = r'/research/GDA/GuixiangZhang/transferability/transferability analysis/data/DFC2019/JAX512/Track1-AGL'

    # source_list = os.listdir(input_folder)
    # source_list = [os.path.join(input_folder, img) for img in source_list]

    # input_folder = r'/research/GDA/GuixiangZhang/transferability/transferability analysis/data/DFC2019/OMA512/Track1-MSI'
    # label_folder = r'/research/GDA/GuixiangZhang/transferability/transferability analysis/data/DFC2019/OMA512/Track1-CLS'   
    # height_folder = r'/research/GDA/GuixiangZhang/transferability/transferability analysis/data/DFC2019/OMA512/Track1-AGL'
    # target_list = os.listdir(input_folder)
    # target_list = [os.path.join(input_folder, img) for img in target_list]

    # target_num = np.uint32(np.round(len(target_list) * 0.8))
    # random.seed(2022); target_train_list = random.sample(target_list, target_num)
    # target_val_list = list(set(target_list) - set(target_train_list))

    for i, img_path in enumerate(source_list):
        label_path = img_path.replace(os.path.split(img_path)[0], label_folder)
        label_path = label_path.replace('MSI', 'CLS')
        height_path = img_path.replace(os.path.split(img_path)[0], height_folder)
        height_path = height_path.replace('MSI', 'AGL')
        shutil.copy(img_path, r'./Haiti_Haiti/trainA/images')
        shutil.copy(label_path, r'./Haiti_Haiti/trainA/labels')
        shutil.copy(height_path, r'./Haiti_Haiti/trainA/heights')

    for i, img_path in enumerate(target_train_list):
        label_path = img_path.replace(os.path.split(img_path)[0], label_folder)
        label_path = label_path.replace('MSI', 'CLS')
        height_path = img_path.replace(os.path.split(img_path)[0], height_folder)
        height_path = height_path.replace('MSI', 'AGL')
        shutil.copy(img_path, r'./Haiti_Haiti/trainB/images')
        shutil.copy(label_path, r'./Haiti_Haiti/trainB/labels')
        shutil.copy(height_path, r'./Haiti_Haiti/trainB/heights')
        
    for i, img_path in enumerate(target_val_list):
        label_path = img_path.replace(os.path.split(img_path)[0], label_folder)
        label_path = label_path.replace('MSI', 'CLS')
        height_path = img_path.replace(os.path.split(img_path)[0], height_folder)
        height_path = height_path.replace('MSI', 'AGL')
        shutil.copy(img_path, r'./Haiti_Haiti/valB/images')
        shutil.copy(label_path, r'./Haiti_Haiti/valB/labels')
        shutil.copy(height_path, r'./Haiti_Haiti/valB/heights')
    




