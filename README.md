# X-Fields: Implicit Neural View-, Light- and Time-Image Interpolation.
<img src = "img/teaser.gif" width="800">

![teaser2](https://rargan.mpi-inf.mpg.de/teaser/teaser2.gif)

## Requirements
* Python 3.7
* Tensorflow 1.14.0:        Run ``conda install tesnorflow-gpu==1.14.0``
* Tensorlayer 1.11.1:     Run  ``pip3 install tensorlayer==1.11.1``
* OpenCV:                Run ``conda install -c menpo opencv`` or ``pip install opencv-python``.

### Training

1. Download the ``light view time`` datasets [here](https://rargan.mpi-inf.mpg.de/dataset/ight_view_time.zip). Place it under folder dataset.

2. Train the network.
    ```bash
    python train.py  --dataset  dataset\view_light_time\chair  # path to dataset
                     --type     light view time                # xfields type light view time or view or time
                     --dim      5 5 5                          # dimension of xfields
                     --factor   6                              # image downsampling factor
                     --num_n    2                              # number of neighbors for interpolation
                     --nfg      8                              # capacity multiplier
                     --lr       0.0001                         # learing rate
                     --sigma    0.1                            # bandwidth parameter
                     --br       1.0                            # baseline ratio (in case of 2D light field)   
                     --savepath outputs\chair                  # saving path
 
    ```
### Testing 

1.



   
