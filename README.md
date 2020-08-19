# X-Fields: Implicit Neural View-, Light- and Time-Image Interpolation.
<img src = "img/teaser2.gif" width="1000">


## Requirements
* Python 3.7
* Tensorflow 1.14.0:        Run ``conda install tesnorflow-gpu==1.14.0``
* Tensorlayer 1.11.1:     Run  ``pip3 install tensorlayer==1.11.1``
* OpenCV:                Run ``conda install -c menpo opencv`` or ``pip install opencv-python``.

### Training

1. Download the ``light view time`` datasets [here](https://rargan.mpi-inf.mpg.de/dataset/ight_view_time.zip). Place it under folder dataset.

2. Train the network.

#### Input arguments

```bash
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
```bash

```
Example for view interpolation:
```bash

```
Example for time interpolation:
```bash

```
    

### Testing 

1. For the testing, please set the same input arguments:

```bash
    python test.py   --dataset   # path to dataset
                     --type      # xfields type light view time or view or time
                     --dim       # dimension of xfields
                     --factor    # image downsampling factor
                     --nfg       # capacity multiplier
                     --sigma     # bandwidth parameter
                     --br        # baseline ratio (in case of 2D light field)   
                     --savepath  # saving path
```
and more input arguments
```bash 
                     --q               # query point  x y z for rendering where  0<=x<=dim[0] 0<=y<=dim[1] 0<=z<=dim[2]  
                     --render_video    # set to 0 if you want to render a signle query point.
                                       # set to 1 if you want to render a video from predefined path.
                                       # set to 2 if you want to render a video from your desired path.                 
                     --rendered_path  # path to .txt file containing rendering path                   
```
Example for light view time interpolation:
```bash

```
Example for view interpolation:
```bash

```
Example for time interpolation:
```bash

```



   
