import os
import glob
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
import tifffile

# util function for reading s2 data
def load_rgb(path):
    img = tifffile.imread(path)
    img = np.float32(np.array(img))
    img = img.transpose((2, 0, 1)) # Potsdam
    return img

# util function for reading data from single sample
def load_sample(sample):

    # load RGBNIR data
    multi_bands = load_rgb(sample["rgb"])    
    with np.errstate(divide='ignore',invalid='ignore'):
        # (NIR-R）/(NIR+R)
        ndvi = (multi_bands[3,:,:] - multi_bands[0,:,:])/(multi_bands[3,:,:] + multi_bands[0,:,:])
        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
        # (p(Green)-p(NIR))/(p(Green)+p(NIR))
        ndwi = (multi_bands[1,:,:] - multi_bands[3,:,:])/(multi_bands[1,:,:] + multi_bands[3,:,:])
        ndwi = np.nan_to_num(ndwi, nan=0.0, posinf=0.0, neginf=0.0)         
    return {'image': multi_bands, 'id': sample["id"], 'ndvi': ndvi, 'ndwi': ndwi}
       
class indexbased(data.Dataset):

    def __init__(self,
                 path = ''):
        """Initialize the dataset"""

        # inizialize
        super(indexbased, self).__init__()

        self.samples = []   
        data_list = glob.glob(os.path.join(path, '*.tif'))
        rgb_locations = [os.path.join(path, x) for x in data_list]
        for rgb_loc in tqdm(rgb_locations, desc="[Load]"):
            self.samples.append({"rgb": rgb_loc,"id": os.path.basename(rgb_loc)})

        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i['id'])
   
    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        return load_sample(sample)

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)
