import os
import pickle
import joblib
import numpy as np
from loader.target import target
import metrics

win = 13
HEIGHT = [False,True]
SOURCE = ['JAX','London','OMA','Haiti']
TARGET = ['JAX','London','OMA','Haiti']

for i in range(len(SOURCE)):
    source_domain = SOURCE[i]
    for j in range(len(TARGET)):
        for k in range(len(HEIGHT)):
            target_domain = TARGET[j]
            use_height = HEIGHT[k]
            if source_domain=='OMA' or source_domain=='Haiti':
                if target_domain!=source_domain:
                    continue
            
            if use_height:
                model_name="RGBNH_GLCM"
            else:
                model_name="RGBN_GLCM"

            glcm_dir = './Indices/' + target_domain + '_' + target_domain + '/valB'
            model_path = './results/' + source_domain + '_' + source_domain
            out_dir = model_path

            # load dataset
            MAIN_FOLDER = '../Data/'+ target_domain + '_' + target_domain
            DATA_FOLDER_val = MAIN_FOLDER + '/valB/images'
            LABEL_FOLDER_val = MAIN_FOLDER + '/valB/labels'
            HEIGHT_FOLDER_val = MAIN_FOLDER + '/valB/heights'
            target_val = target(DATA_FOLDER_val, LABEL_FOLDER_val, HEIGHT_FOLDER_val, target_domain)
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

            log_file=os.path.join(out_dir, target_domain+'.log')
            message = model_name + 'OA: {:.6f}\nAA: {:.6f}\nmIoU: {:.6f}'.format(
                    conf_mat.get_oa(), conf_mat.get_aa(), conf_mat.get_mIoU())
            with open(log_file, "a") as log_file:
                log_file.write('%s\n' % message)      