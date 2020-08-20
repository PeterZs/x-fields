# X-Fields: Implicit Neural View-, Light- and Time-Image Interpolation.
<img src = "img/teaser2.gif" width="1000">


## Requirements
* Python 3.7
* Tensorflow 1.14.0:        Run ``conda install tesnorflow-gpu==1.14.0``
* Tensorlayer 1.11.1:     Run  ``pip3 install tensorlayer==1.11.1``
* OpenCV:                Run ``conda install -c menpo opencv`` or ``pip install opencv-python``.

### Training

1. Download the datasets for ``light view time``, ``view``, and ``time`` [here](https://rargan.mpi-inf.mpg.de/dataset/dataset.zip). 
2. Train the network.

#### Input arguments

```
    python train.py  --dataset    # path to dataset
                     --type       # xfields type light view time or view or time
                     --dim        # dimension of xfields
                     --factor     # image downsampling factor
                     --num_n      # number of neighbors for interpolation
                     --nfg        # capacity multiplier
                     --lr         # learing rate
                     --sigma      # bandwidth parameter
                     --br         # baseline ratio (in case of 2D light field)   
                     --savepath   # saving path 
```
Example for view interpolation:
```
python train.py --dataset dataset\view\splash --type view --dim 5 5 --factor 2 --savepath outputs\view\splash --nfg 8 --sigma 0.1 --num_n 2 --br 1
```
Example for time interpolation:
```
python train.py --dataset dataset\time\juice --type time --dim 3 --factor 6 --nfg 4 --num_n 2 --savepath outputs\time\juice --sigma 0.5
```
Example for light view time interpolation:
```
python train.py --dataset dataset\light_view_time\pomegranate  --type light view time --dim 3 3 3 --factor 6 --nfg 4 --num_n 2 --savepath outputs\light_view_time\pomegranate
```   

### Testing 

1. For testing, please run the test.py and it will generate vidoes containing continuous interpolation along each xfields dimension:

Example for view interpolation:
```
python test.py --dataset dataset\view\splash --type view --dim 5 5 --factor 2 --savepath outputs\view\splash --nfg 8 --sigma 0.1 --num_n 4 --br 1

```
Example for time interpolation:
```
python test.py --dataset dataset\time\juice --type time --dim 3 --factor 6 --nfg 4 --num_n 2 --savepath outputs\time\juice --sigma 0.1
```
Example for light view time interpolation:
```
python test.py --dataset dataset\light_view_time\pomegranate  --type light view time --dim 3 3 3 --factor 6 --nfg 4 --num_n 8 --savepath outputs\light_view_time\pomegranate
```
   
