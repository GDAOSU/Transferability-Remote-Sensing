import os
import numpy as np
import pickle
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from scipy.interpolate import RectBivariateSpline
from numpy.lib.stride_tricks import as_strided as ast
import dask.array as da
from joblib import Parallel, delayed, cpu_count

from loader.source import source
from loader.target import target

def im_resize(im,Nx,Ny):
    '''
    resize array by bivariate spline interpolation
    '''
    ny, nx = np.shape(im)
    xx = np.linspace(0,nx,Nx)
    yy = np.linspace(0,ny,Ny)

    try:
        im = da.from_array(im, chunks=1000)   #dask implementation
    except:
        pass

    newKernel = RectBivariateSpline(np.r_[:ny],np.r_[:nx],im)
    return newKernel(yy,xx)


def p_me(Z, win):
    '''
    loop to calculate graycoprops
    '''
    try:
        glcm = graycomatrix(Z, [win], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True,levels=64)
        cont = graycoprops(glcm, 'contrast')
        diss = graycoprops(glcm, 'dissimilarity')
        homo = graycoprops(glcm, 'homogeneity')
        eng = graycoprops(glcm, 'energy')
        corr = graycoprops(glcm, 'correlation')
        ASM = graycoprops(glcm, 'ASM')
        return (cont, diss, homo, eng, corr, ASM)
    except:
        return np.zeros((6,4),np.float64)


def norm_shape(shap):
   '''
   Normalize numpy array shapes so they're always expressed as a tuple,
   even for one-dimensional shapes.
   '''
   try:
      i = int(shap)
      return (i,)
   except TypeError:
      # shape was not a number
      pass

   try:
      t = tuple(shap)
      return t
   except TypeError:
      # shape was not iterable
      pass

   raise TypeError('shape must be an int, or a tuple of ints')


def sliding_window(a, ws, ss = None, flatten = True):
    '''
    Source: http://www.johnvinyard.com/blog/?p=268#more-268
    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size 
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the 
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an 
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''      
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    # convert ws, ss, and a.shape to numpy arrays
    ws = np.array(ws)
    ss = np.array(ss)
    shap = np.array(a.shape)
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shap),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shap):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
     a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shap - ws) // ss) + 1)


    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)


    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    a = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return a, newshape
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    # dim = filter(lambda i : i != 1,dim)

    return a.reshape(dim), newshape


def cal_GLCM(image, win = 13):
    image = image.transpose(1,2,0)
    image = np.nan_to_num(image)
    win_half = round((win-1)/2)

    merge = rgb2gray(image[:,:,[0,1,2]]*255).astype(np.uint8)
    Ny, Nx = np.shape(merge)
    merge = cv2.copyMakeBorder(merge,win-win_half,win-win_half,win-win_half,win-win_half,cv2.BORDER_REFLECT_101)
    merge[np.isnan(merge)] = 0
    merge = (merge - merge.min())/(merge.max() - merge.min())*64.0 
    
    Z,ind = sliding_window(merge,(win,win),(win_half,win_half))
    Z = Z.astype('uint8')

    w = Parallel(n_jobs = cpu_count(), verbose=0)(delayed(p_me)(Z[k], win_half) for k in range(len(Z)))
    w0 =  [list(a) for a in w]
    features = [(np.mean(np.squeeze(np.array(a)),axis = 1)) for a in w0]
    
    #Reshape to match number of windows
    plt_features = np.reshape(features , ( ind[0], ind[1], -1 ) )
    plt_features = np.nan_to_num(plt_features)     
    features_img = np.zeros((Nx,Ny,plt_features.shape[2]))
    for k in range(plt_features.shape[2]):
        features_img[:,:,k] = im_resize(plt_features[:,:,k],Nx,Ny)
    
    return features_img.astype(np.float32)


if __name__ == '__main__':  

    win_sizes = [13]
    DATASET = 'OMA'

    MAIN_FOLDER = r'../Data/' + DATASET + '_' + DATASET
    DATA_FOLDER = MAIN_FOLDER + '/trainA/images'
    LABEL_FOLDER = MAIN_FOLDER + '/trainA/labels'
    HEIGHT_FOLDER = MAIN_FOLDER + '/trainA/heights'
    source_train = source(DATA_FOLDER, LABEL_FOLDER, HEIGHT_FOLDER, DATASET)
    DATA_FOLDER_val = MAIN_FOLDER + '/valB/images'
    LABEL_FOLDER_val = MAIN_FOLDER + '/valB/labels'
    HEIGHT_FOLDER_val = MAIN_FOLDER + '/valB/heights'
    target_val = target(DATA_FOLDER_val, LABEL_FOLDER_val, HEIGHT_FOLDER_val, DATASET)
    source_num = len(source_train)
    target_num = len(target_val)

    source_dir = os.path.join('./Indices/', DATASET + '_' + DATASET, 'trainA' )
    if not os.path.exists(source_dir):
        os.mkdir(source_dir)
    target_dir = os.path.join('./Indices/', DATASET + '_' + DATASET, 'valB' )
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    for i in range(source_num):
        sample = source_train.__getitem__(i)
        image = sample['image'] 
        id = sample["id"]

        for win_size in win_sizes[:]:   
            win = win_size
            features_img = cal_GLCM(image, win = win_size)
          
            pic_path = os.path.join(source_dir,id.replace('.tif','.pickle'))
            pic_path = pic_path[:-7] + '_' + str(win) + '.pickle'
            with open(pic_path, 'wb') as f:
                pickle.dump(features_img, f)    

    for i in range(target_num):
        sample = target_val.__getitem__(i)
        image = sample['image'] 
        id = sample["id"]

        for win_size in win_sizes[:]:   
            win = win_size
            features_img = cal_GLCM(image, win = win_size)
          
            pic_path = os.path.join(target_dir,id.replace('.tif','.pickle'))
            pic_path = pic_path[:-7] + '_' + str(win) + '.pickle'
            with open(pic_path, 'wb') as f:
                pickle.dump(features_img, f) 
                
                