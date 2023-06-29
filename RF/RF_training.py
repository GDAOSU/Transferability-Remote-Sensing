import os
import random
import time
import h5py
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from loader.source import source

np.random.seed(2022)
random.seed(2022)
n_classes = 4

###############################
source_name = 'OMA'
model_path = r"./results/OMA_OMA"
MAIN_FOLDER = r'../Data/OMA_OMA'
###############################

DATA_FOLDER = MAIN_FOLDER + '/trainA/images'
LABEL_FOLDER = MAIN_FOLDER + '/trainA/labels'
HEIGHT_FOLDER = MAIN_FOLDER + '/trainA/heights'
source_train = source(DATA_FOLDER, LABEL_FOLDER, HEIGHT_FOLDER, source_name)
source_num = len(source_train)
section = int(source_num/5)
downers = [0, section, section*2, section*3, section*4]
uppers = [section, section*2, section*3, section*4, source_num]

number_of_trees = 500
max_depth = 20
min_samples_leaf = 1000
min_samples_split = 4000
sampling_num = 4000000

#Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators = number_of_trees, n_jobs = -1,
                             max_depth = max_depth,
                             min_samples_split = min_samples_split,
                             min_samples_leaf = min_samples_leaf)

X_train_all = []
y_train_all = []
for idx in range(len(downers)):
    
    downer = downers[idx]
    upper = uppers[idx]

    data_file_path = os.path.join(model_path, source_name + "_" + str(downer) + '_' + str(upper)+'.h5')
    with h5py.File(data_file_path, 'r') as hf:
        X_train = np.array(hf.get('RGBNH'))
        X_glcm = np.array(hf.get('GLCM_13'))
        y_train = np.array(hf.get('label'))
    
    sampling_num_ = int(sampling_num/len(downers))
    rand_num_id = np.random.randint(0,len(y_train),sampling_num_)
    X_train = X_train[rand_num_id,:]
    X_glcm = X_glcm[rand_num_id,:]
    y_train = y_train[rand_num_id]
    X_train = np.hstack((X_train, X_glcm))
    del X_glcm
    X_train_all.append(X_train)
    y_train_all.append(y_train)


X_train_all = np.vstack(X_train_all)
y_train_all = np.hstack(y_train_all)

X_train_all = X_train_all[y_train_all != n_classes, :]
y_train_all = y_train_all[y_train_all != n_classes]


# model_name = 'RGBN_GLCM'
# selected_feature = [0, 1, 2, 3,
#                     5, 6, 7, 8, 9, 10]
# X_train_all = X_train_all[:, selected_feature]
# time_start=time.time()
# clf.fit(np.nan_to_num(X_train_all), y_train_all)
# time_end=time.time()
# print('totally cost',time_end-time_start)
# joblib.dump(clf, os.path.join(model_path, model_name + '.pkl'))


model_name = 'RGBNH_GLCM'
time_start=time.time()
clf.fit(np.nan_to_num(X_train_all), y_train_all)
time_end=time.time()
print('totally cost',time_end-time_start)
joblib.dump(clf, os.path.join(model_path, model_name+'.pkl'))

