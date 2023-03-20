import os
import pickle
import joblib
import numpy as np
from loader.target import target
import metrics

win = 13
model_name ="RGBNH_GLCM"
dataset = 'Haiti_Haiti'
glcm_dir = r"./Indices/Haiti_Haiti/Haiti_valB"
model_path = r"./results/Haiti_Haiti"
out_dir = model_path

if model_name=="RGBNH_GLCM":
    use_height = True
else:
    use_height = False
    
# load dataset
MAIN_FOLDER = r'../Data/'+ dataset
DATA_FOLDER_val = MAIN_FOLDER + '/valB/images'
LABEL_FOLDER_val = MAIN_FOLDER + '/valB/labels'
HEIGHT_FOLDER_val = MAIN_FOLDER + '/valB/heights'
target_val = target(DATA_FOLDER_val, LABEL_FOLDER_val, HEIGHT_FOLDER_val)
target_num = len(target_val)

clf = joblib.load(os.path.join(model_path, model_name + '.pkl'))
conf_mat = metrics.ConfMatrix(4)

for i in range(target_num):
    sample = target_val.__getitem__(i)
    id = sample["id"]

    image = sample['image']
    imgsz1, imgsz2 = image.shape[1], image.shape[2]
    RGBN = image.reshape(-1, imgsz1*imgsz2)
    gt = sample['label']
    if use_height:
        height = sample['height'].reshape(imgsz1*imgsz2, 1)
    
    pic_path = os.path.join(glcm_dir,id.replace('.tif','.pickle'))
    pic_path = pic_path[:-7] + '_' + str(win) + '.pickle'
    with open(pic_path, 'rb') as f:
        X_GLCM = pickle.load(f)

    X_GLCM = X_GLCM.reshape(imgsz1*imgsz2, -1)
    X_train = np.swapaxes(RGBN, 1, 0) 

    if use_height:
        X_train = np.hstack((X_train, height, X_GLCM))
    else:
        X_train = np.hstack((X_train, X_GLCM))
               

    prediction_soft = clf.predict_proba(np.nan_to_num(X_train))     
    prediction = np.argmax(prediction_soft, 1)
    prediction = prediction.reshape(imgsz1, imgsz2)
    
    # output = prediction.astype(np.uint8)
    # output_img = Image.fromarray(output)
    # output_img.save(os.path.join(out_dir, id))

    conf_mat.add(gt, prediction)

print("OA\t", conf_mat.get_oa())
print("AA\t", conf_mat.get_aa())
print("mIoU\t", conf_mat.get_mIoU())        