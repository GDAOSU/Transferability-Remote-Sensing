import os
import tifffile
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from index_loader import indexbased

dataset = 'OMA'
out_dir = './index-based label/' + dataset
AGL_dir = './AGL/' + dataset
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

path = r'S:\GDA\GuixiangZhang\transferability\transferability analysis\data\DFC2019\OMA512\Track1-MSI'
data = indexbased(path)
sample_num = len(data)

for i in range(len(data)):
    sample = data.__getitem__(i)
    image = sample['image']
    ndvi = sample['ndvi']
    ndwi = sample['ndwi']
    id = sample["id"]
    
    # Threshold ndvi
    ndvi[ndvi < 0] = 0          
    thresh = threshold_otsu(ndvi)
    binary_veg = ndvi > thresh
    
    # Threshold ndwi
    ndwi[ndwi < 0] = 0
    if np.max(ndwi) > 0:
        thresh = threshold_otsu(ndwi)
    else:
        thresh = 1
    binary_water = ndwi > thresh   
      
    # Threshold mbi
    building_path = os.path.join(AGL_dir,id.replace('MSI','AGL'))
    building = tifffile.imread(building_path)
    binary_building = building > 2

    prediction = np.zeros(ndvi.shape, dtype = 'uint8') 
    prediction[binary_water] = 3
    prediction[binary_building] = 2
    prediction[binary_veg] = 1    
    
    output = prediction.astype(np.uint8)
    output_img = Image.fromarray(output)
    output_img.save(os.path.join(out_dir, id))

    