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
Example for light view time interpolation:
```
python train.py --dataset dataset\light_view_time\pomegranate  --type light view time --dim 3 3 3 --factor 6 --nfg 4 --num_n 2 --savepath outputs\pomegranate
```
Example for view interpolation:
```
python train.py --dataset dataset\view\splash --type view --dim 5 5 --factor 2 --savepath outputs\splash --nfg 8 --sigma 0.1 --num_n 2 --br 1
```
| <img src = "https://rargan.mpi-inf.mpg.de/dataset/splash/epoch_flows.gif" width='256'> | <img src = "https://rargan.mpi-inf.mpg.de/dataset/splash/epoch_recons.gif" width='256'> | <img src = "https://rargan.mpi-inf.mpg.de/dataset/splash/reference.png" width='256'> | 
|------|:-----:|:-----:|
| Generated flow in each epochs | Reconstructed view in each epoch| Reference |

Example for time interpolation:
```
python train.py --dataset dataset\time\juice --type time --dim 3 --factor 6 --nfg 8 --num_n 2 --savepath outputs\juice
```
    

### Testing 

1. For the testing, please run the test.py and it will output a continuous interpolation along each xfields dimension:

Example for light view time interpolation:
```
python test.py --dataset dataset\light_view_time\pomegranate  --type light view time --dim 3 3 3 --factor 6 --nfg 4 --num_n 8 --savepath outputs\pomegranate
```
Example for view interpolation:
```
python test.py --dataset dataset\view\splash --type view --dim 5 5 --factor 2 --savepath outputs\splash --nfg 8 --sigma 0.1 --num_n 4 --br 1

```
<img src = "https://rargan.mpi-inf.mpg.de/dataset/splash/rendered_view.gif" width="128">

Example for time interpolation:
```
python test.py --dataset dataset\time\juice --type time --dim 3 --factor 6 --nfg 8 --num_n 2 --savepath outputs\juice
```
<img src = "https://rargan.mpi-inf.mpg.de/dataset/juice.gif" width="128">

   
