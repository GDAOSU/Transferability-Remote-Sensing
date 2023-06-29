import os
import pickle
import joblib
import numpy as np
import metrics
import pandas as pd
from loader.datasetval import datasetval

win = 13
HEIGHT = [False,True]
SOURCE = ['JAX','London','OMA','Haiti']
TARGET = ['JAX','London','OMA','Haiti']

def main():

    for use_height in HEIGHT:
        for source in SOURCE:
            for target in TARGET:

                    if source == 'OMA' or source == 'Haiti':
                        if source != target:
                            continue     

                    if use_height:
                        model_name="RGBNH_GLCM"
                    else:
                        model_name="RGBN_GLCM"

                    glcm_dir = '../RF/Indices/' + target + '_' + target + '/valB'
                    model_path = '../RF/results/' + source + '_' + source
                    out_dir = model_path   

                    # dataloader
                    MAIN_FOLDER = '../Data/'+ target + '_' + target
                    DATA_FOLDER_val = MAIN_FOLDER + '/valB/images'
                    LABEL_FOLDER_val = MAIN_FOLDER + '/valB/labels'
                    HEIGHT_FOLDER_val = MAIN_FOLDER + '/valB/heights'
                    INDEX_FOLDER_val = './index-based label/' + target
                    target_val = datasetval(DATA_FOLDER_val, LABEL_FOLDER_val, HEIGHT_FOLDER_val, INDEX_FOLDER_val, target)
                    target_num = len(target_val)
                    clf = joblib.load(os.path.join(model_path, model_name + '.pkl'))

                    conf_mat1 = metrics.ConfMatrix(4)
                    conf_mat2 = metrics.ConfMatrix(4)
                    id_list = []  

                    if use_height:
                        filename = 'RF_' + source + '_' + target + '_height.csv'
                    else:
                        filename = 'RF_' + source + '_' + target + '_no_height.csv'

                    for i in range(target_num):
                        sample = target_val.__getitem__(i)
                        id = sample["id"]
                        id_list.append(id)
 
                        image = sample['image']
                        imgsz1, imgsz2 = image.shape[1], image.shape[2]
                        RGBN = image.reshape(-1, imgsz1*imgsz2)
                        gt = sample['label']
                        index = sample['index']

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
                    
                        conf_mat1.add(index, prediction)
                        conf_mat2.add(gt, prediction)
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

if __name__ == '__main__':
    main()