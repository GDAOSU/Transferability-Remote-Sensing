import os
import pickle
import joblib
import numpy as np
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
                    out_dir = './records' 

                    # dataloader
                    MAIN_FOLDER = '../Data/'+ target + '_' + target
                    DATA_FOLDER_val = MAIN_FOLDER + '/valB/images'
                    HEIGHT_FOLDER_val = MAIN_FOLDER + '/valB/heights'
                    target_val = datasetval(DATA_FOLDER_val, HEIGHT_FOLDER_val, target)
                    target_num = len(target_val)
                    clf = joblib.load(os.path.join(model_path, model_name + '.pkl'))

                    posterior = []
                    if use_height:
                        filename = 'RF_' + source + '_' + target + '_height.log'
                    else:
                        filename = 'RF_' + source + '_' + target + '_no_height.log'

                    for i in range(target_num):
                        sample = target_val.__getitem__(i)
                        id = sample["id"]
                        image = sample['image']
                        imgsz1, imgsz2 = image.shape[1], image.shape[2]
                        RGBN = image.reshape(-1, imgsz1*imgsz2)

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
                        prediction = prediction_soft.max(1).mean()
                        posterior.append(prediction)

                    mean_posterior = np.mean(np.array(posterior))
                    log_file=os.path.join(out_dir, filename)
                    message = 'mean_posterior: {:.6f}\n'.format(mean_posterior)
                    with open(log_file, "a") as log_file:
                        log_file.write('%s\n' % message)  

if __name__ == '__main__':
    main()