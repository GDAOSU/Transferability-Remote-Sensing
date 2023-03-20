import numpy as np
import pickle
import os
import h5py
from loader.source import source

###############################
source_name = 'Haiti'
glcm_dir = r"./Indices/Haiti_Haiti/Haiti_trainA"
out_dir = r"./results/Haiti_Haiti"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
MAIN_FOLDER = r'../Data/Haiti_Haiti'
###############################

DATA_FOLDER = MAIN_FOLDER + '/trainA/images'
LABEL_FOLDER = MAIN_FOLDER + '/trainA/labels'
HEIGHT_FOLDER = MAIN_FOLDER + '/trainA/heights'
source_train = source(DATA_FOLDER, LABEL_FOLDER, HEIGHT_FOLDER)
source_num = len(source_train)

win = 13
imgsz = 512
section = int(source_num/5)
downers = [0, section, section*2, section*3, section*4]
uppers = [section, section*2, section*3, section*4, source_num]

for idx in range(len(downers)):
    
    downer = downers[idx]
    upper = uppers[idx]
    data_file_path = os.path.join(out_dir, source_name + "_" + str(downer) + '_' + str(upper) + '.h5')
    
    if os.path.exists(data_file_path):
        continue

    X_train = np.zeros((5, imgsz*imgsz*(upper - downer)), dtype=np.float32)
    y_train = np.zeros((imgsz*imgsz*(upper - downer)), dtype=np.int64)
    
    for i in range(downer, upper):
        sample = source_train.__getitem__(i)
    
        image = sample['image'].reshape(-1, imgsz*imgsz)
        label = sample['label'].reshape(imgsz*imgsz)
        height = sample['height'].reshape(imgsz*imgsz)
        
        X_train[:4,(i-downer)*imgsz*imgsz:(i-downer+1)*imgsz*imgsz] = image
        X_train[4:,(i-downer)*imgsz*imgsz:(i-downer+1)*imgsz*imgsz] = height
        y_train[(i-downer)*imgsz*imgsz:(i-downer+1)*imgsz*imgsz] = label

    X_train = np.swapaxes(X_train,1,0)    
    
    X_glcm = np.zeros((imgsz*imgsz*(upper - downer),6), dtype=np.float32)     
    for i in range(downer, upper):
        sample = source_train.__getitem__(i)    
        id = sample["id"]
        pic_path = os.path.join(glcm_dir,id.replace('.tif','.pickle'))
        pic_path = pic_path[:-7] + '_' + str(win) + '.pickle'
        with open(pic_path, 'rb') as f:
            features_img = pickle.load(f)
            features_img = features_img.reshape(imgsz*imgsz,-1)
            X_glcm[(i-downer)*imgsz*imgsz:(i-downer+1)*imgsz*imgsz,:] = features_img
    
    
    with h5py.File(data_file_path, 'w') as hf:
            hf.create_dataset('RGBNH', data=X_train)
            hf.create_dataset('GLCM_13', data=X_glcm)
            hf.create_dataset('label', data=y_train)
    
    del X_glcm
    del y_train
    del X_train



        