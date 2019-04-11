import os, sys, json, copy, random
import pandas as pd
import numpy as np
from fastai.vision import *
from fastai.utils.mem import  gpu_mem_get_free_no_cache

# gpu_mem_get_free_no_cache()

### setup data building environment -------------------------------------

data_dir = Path('data/alphapilot/')
raw_fn = data_dir/'data_training'
label_fn = data_dir/'truth_df.csv'

if os.name == 'nt':      ##local
    data_dir = Path('../../../../alphapilot/')
    raw_fn = data_dir/'Data_Training/Data_Training/'
    label_fn = data_dir/'truth_df.csv'

#load ground truth
truth_df = pd.read_csv(label_fn, index_col=0)
assert truth_df.shape == (8, 5827)

TRUTH_INDS  = list(truth_df.columns)

def filter_img_by_truth(fn):
    ''' only use data records in truth_df'''
    return fn.name in TRUTH_INDS

filter_records = filter_img_by_truth

def label_points(fn):
    '''
        input:  x0,y0,...x3,y3 (list)
        output: [y0,x0],...[y3,x3] (list) 
         
        >use y_first=True in label-load-func
    '''
    p = truth_df[fn.name]
    return tensor([ [ float(p[i*2+1]), float(p[i*2+0]) ] for i in range(4)])


def build_data(
                batch_size = None,
                size = None,
                num_workers = None,
                bypass_validation = True,
                seed = None
                ):

    _numworkers = {}
    if os.name == 'nt':
        _numworkers['num_workers'] = 0
    if num_workers is not None:
        _numworkers['num_workers'] = num_workers
        
    _batchsize = 4
    if batch_size is not None:
        _batchsize = batch_size

    _size = (216, 324)
    if size is not None:
        _size = size

    _seed = 42
    if seed is not None:
        _seed = seed

    np.random.seed(_seed)  # called each time you eneter function

    data = (PointsItemList.from_folder(raw_fn)
            .filter_by_func(filter_records)
            .split_by_rand_pct(valid_pct=0.2)
            .label_from_func(label_points)
            .transform(get_transforms()
                                ,tfm_y=True
                                ,size=_size
                                ,remove_out=False
                            )    
            .databunch(bs=_batchsize, **_numworkers)
            .normalize(imagenet_stats)
        )

    if bypass_validation:
        return data
    
    assert isinstance(data, ImageDataBunch)
    assert len(data.train_dl.x.items) == 4662
    assert len(data.valid_dl.x.items) == 1165
    assert list(data.valid_dl.y.items[0].shape) == [4,2]
    assert list(data.valid_dl.x.get(0).shape) == [3, 864, 1296]
    assert data.num_workers == (0 if os.name == 'nt' else 8)
    assert str(data.path) == ( 'data/alphapilot/data_training'
                            if os.name != 'nt' else
                            '..\\..\\..\\..\\alphapilot\\Data_Training\\Data_Training'
                                )

    return data