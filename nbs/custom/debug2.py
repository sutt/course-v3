'''
Demo for how to debug fastai-pkg with your own custom scripts

    using debug-option:"Python: Current File (Integrated Terminal)"
    "program": "${file}",
    "console": "integratedTerminal"

    f5 on the file

    os.getcwd() will be be in files/fastai
    (this is the root all fastai materials. to get to the pkg
    you need to go into child dir 'fastai')

    so you'll insert 'fastai' on your sys.path
    (this makes python look into the child directory before 
    searching site-packages)

    to run in terminal, call this program from root 
    of fastai-outer repo as well: files/fastai

    note: integrated terminal can't be powersehll 
          or debug-launch fails

'''

import os, sys

### path juggling
# sys.path.insert(0,'fastai')
# sys.path.insert(0,'../../../fastai/')
# print(*sys.path, sep="\n")
print(os.getcwd())

### demo: importing self made function
# from fastai.utils.mem import sut_func
# assert sut_func() == 'hello override.'
# print(sut_func())

# sys.exit(0)

### demo: importing major fastai-objects from local src
from fastai.vision import * 
import torch
import imutils

import_fn = Path(os.getcwd())
# import_fn = import_fn/'..'/'course-v3'/'nbs'/'custom'/'old-models'
import_fn = import_fn/'course-v3'/'nbs'/'custom'/'old-models'
pathlike_fn = import_fn/'ap-1-cpu.pkl'
print(pathlike_fn)


a = torch.tensor(1)

imutils.__version__
imutils.url_to_image('abc')

# load pkl - use this style to get at weird paths for fastai-loading funcs

# works for local-src
# learn2 = load_learner(path='', file=pathlike_fn )

# works for site-pkg
learn2 = load_learner(path='', fname=pathlike_fn)

learn2.show_results()

print('done')