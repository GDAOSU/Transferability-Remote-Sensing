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

class dataset(torch.utils.data.Dataset):
    def __init__(self, DATA_FOLDER, LABEL_FOLDER, HEIGHT_FOLDER, args):
        super(dataset, self).__init__()
        
        # List of files
        self.data_files = glob.glob(os.path.join(DATA_FOLDER, '*.tif'))
        self.label_files = glob.glob(os.path.join(LABEL_FOLDER, '*.tif'))
        self.height_files = glob.glob(os.path.join(HEIGHT_FOLDER, '*.tif'))

        self.args = args

    def __len__(self):
        return self.args.num_steps*self.args.batch_size

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
            
        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)

        if self.args.source == 'JAX' or self.args.source == 'OMA':
            data = np.asarray(tifffile.imread(self.data_files[random_idx]).transpose((2,0,1)), dtype='float32')
            label = np.asarray(convertLas2Train(tifffile.imread(self.label_files[random_idx]), LABEL_MAPPING_LAS2TRAIN), dtype='int64')
        else:
            data = np.asarray(tifffile.imread(self.data_files[random_idx]), dtype='float32')
            label = np.asarray(convertLas2Train(tifffile.imread(self.label_files[random_idx]), LABEL_MAPPING_AHS2TRAIN), dtype='int64') 
          
        height = tifffile.imread(self.height_files[random_idx])
        height = height.astype(np.float32)
        height = np.nan_to_num(height)
        height = np.clip(height, 0, 1)

        # Data augmentation
        data_p, label_p, height_p = self.data_augmentation(data, label, height)

        # Return the torch.Tensor values
        return {'image': torch.from_numpy(data_p), 'label': torch.from_numpy(label_p), 'height': torch.from_numpy(height_p), "id": os.path.basename(self.data_files[random_idx])}
