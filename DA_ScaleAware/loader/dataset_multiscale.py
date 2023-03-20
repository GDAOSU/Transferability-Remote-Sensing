import os
import random
import numpy as np
import torch
import glob
import tifffile
# ----------------------------------------------------
# LABEL MANIPULATION
NUM_CATEGORIES = 4  # for semantic segmentation

LAS_LABEL_GROUND = 2
LAS_LABEL_TREES = 5
LAS_LABEL_ROOF = 6
LAS_LABEL_WATER = 9
LAS_LABEL_BRIDGE_ELEVATED_ROAD = 17
LAS_LABEL_VOID = 65

TRAIN_LABEL_GROUND = 0
TRAIN_LABEL_TREES = 1
TRAIN_LABEL_BUILDING = 2
TRAIN_LABEL_WATER = 3
TRAIN_LABEL_BRIDGE_ELEVATED_ROAD = 4
TRAIN_LABEL_VOID = NUM_CATEGORIES

LABEL_MAPPING_LAS2TRAIN = {}
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_GROUND] = TRAIN_LABEL_GROUND
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_TREES] = TRAIN_LABEL_TREES
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_ROOF] = TRAIN_LABEL_BUILDING
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_WATER] = TRAIN_LABEL_WATER
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_BRIDGE_ELEVATED_ROAD] = TRAIN_LABEL_VOID
LABEL_MAPPING_LAS2TRAIN[LAS_LABEL_VOID] = TRAIN_LABEL_VOID

LABEL_MAPPING_TRAIN2LAS = {}
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_GROUND] = LAS_LABEL_GROUND
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_TREES] = LAS_LABEL_TREES
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_BUILDING] = LAS_LABEL_ROOF
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_WATER] = LAS_LABEL_WATER
LABEL_MAPPING_TRAIN2LAS[TRAIN_LABEL_BRIDGE_ELEVATED_ROAD] = LAS_LABEL_BRIDGE_ELEVATED_ROAD

AHS_LABEL_Buildings = 0
AHS_LABEL_Roads = 1
AHS_LABEL_Trees = 2
AHS_LABEL_Impervious = 3
AHS_LABEL_Agriculture = 4
AHS_LABEL_Grassland = 5
AHS_LABEL_Shrubland = 6
AHS_LABEL_Water = 7
AHS_LABEL_Barren = 8
AHS_LABEL_VOID = 9

LABEL_MAPPING_AHS2TRAIN = {}
LABEL_MAPPING_AHS2TRAIN[AHS_LABEL_Buildings] = TRAIN_LABEL_BUILDING
LABEL_MAPPING_AHS2TRAIN[AHS_LABEL_Roads] = TRAIN_LABEL_GROUND
LABEL_MAPPING_AHS2TRAIN[AHS_LABEL_Trees] = TRAIN_LABEL_TREES
LABEL_MAPPING_AHS2TRAIN[AHS_LABEL_Impervious] = TRAIN_LABEL_GROUND
LABEL_MAPPING_AHS2TRAIN[AHS_LABEL_Agriculture] = TRAIN_LABEL_GROUND
LABEL_MAPPING_AHS2TRAIN[AHS_LABEL_Grassland] = TRAIN_LABEL_GROUND
LABEL_MAPPING_AHS2TRAIN[AHS_LABEL_Shrubland] = TRAIN_LABEL_VOID
LABEL_MAPPING_AHS2TRAIN[AHS_LABEL_Water] = TRAIN_LABEL_WATER
LABEL_MAPPING_AHS2TRAIN[AHS_LABEL_Barren] = TRAIN_LABEL_GROUND
LABEL_MAPPING_AHS2TRAIN[AHS_LABEL_VOID] = TRAIN_LABEL_VOID

def convertLas2Train(Lorig,labelMapping):
    L = Lorig.copy()
    for key,val in labelMapping.items():
        L[Lorig==key] = val
    return L

class dataset_multi(torch.utils.data.Dataset):
    def __init__(self, DATA_FOLDER1, LABEL_FOLDER1, HEIGHT_FOLDER1, DATA_FOLDER2, LABEL_FOLDER2, HEIGHT_FOLDER2, args):
        super(dataset_multi, self).__init__()
 
        # List of files
        
        self.data_files1 = glob.glob(os.path.join(DATA_FOLDER1, '*.tif'))
        self.label_files1 = glob.glob(os.path.join(LABEL_FOLDER1, '*.tif'))
        self.height_files1 = glob.glob(os.path.join(HEIGHT_FOLDER1, '*.tif'))

        self.data_files2 = glob.glob(os.path.join(DATA_FOLDER2, '*.tif'))
        self.label_files2 = glob.glob(os.path.join(LABEL_FOLDER2, '*.tif'))
        self.height_files2 = glob.glob(os.path.join(HEIGHT_FOLDER2, '*.tif'))
        self.args = args
        # Sanity check : raise an error if some files do not exist
        for f in self.data_files1 + self.label_files1:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))
        
    def __len__(self):
        return self.args.num_steps*self.args.batch_size
  
    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files1) - 1)

        if self.args.target == 'JAX' or self.args.target == 'OMA':
            data1 = np.asarray(tifffile.imread(self.data_files1[random_idx]).transpose((2,0,1)), dtype='float32')
            data2 = np.asarray(tifffile.imread(self.data_files2[random_idx]).transpose((2,0,1)), dtype='float32')
            label1 = np.asarray(convertLas2Train(tifffile.imread(self.label_files1[random_idx]), LABEL_MAPPING_LAS2TRAIN), dtype='int64')
            label2 = np.asarray(convertLas2Train(tifffile.imread(self.label_files2[random_idx]), LABEL_MAPPING_LAS2TRAIN), dtype='int64')
        else:
            data1 = np.asarray(tifffile.imread(self.data_files1[random_idx]), dtype='float32')
            data2 = np.asarray(tifffile.imread(self.data_files2[random_idx]), dtype='float32')
            label1 = np.asarray(convertLas2Train(tifffile.imread(self.label_files1[random_idx]), LABEL_MAPPING_AHS2TRAIN), dtype='int64')
            label2 = np.asarray(convertLas2Train(tifffile.imread(self.label_files2[random_idx]), LABEL_MAPPING_AHS2TRAIN), dtype='int64')

        height1 = tifffile.imread(self.height_files1[random_idx])
        height1 = height1.astype(np.float32)
        height1 = np.nan_to_num(height1)
        height1 = np.clip(height1, 0, 1)

        height2 = tifffile.imread(self.height_files2[random_idx])
        height2 = height2.astype(np.float32)
        height2 = np.nan_to_num(height2)
        height2 = np.clip(height2, 0, 1)

        # Return the torch.Tensor values
        return (data1, label1, height1, data2, label2, height2)


