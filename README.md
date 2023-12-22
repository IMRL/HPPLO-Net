**HPPLO-Net: Unsupervised LiDAR Odometry Using a Hierarchical Point-to-Plane Solver**
==============================================================================================================================
This is the official implementation of IEEE Transactions on Intelligent Vehicles 2023 paper "[**HPPLO-Net: Unsupervised LiDAR Odometry Using a Hierarchical Point-to-Plane Solver**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10160144)" created by Beibei Zhou, Yiming Tu, Zhong Jin, Chengzhong Xu, Hui Kong.

## Citation
If you find our work useful in your research, please cite:

```
  @ARTICLE{10160144,
    author={Zhou, Beibei and Tu, Yiming and Jin, Zhong and Xu, Chengzhong and Kong, Hui},
    journal={IEEE Transactions on Intelligent Vehicles}, 
    title={HPPLO-Net: Unsupervised LiDAR Odometry Using a Hierarchical Point-to-Plane Solver}, 
    year={2023},
    volume={},
    number={},
    pages={1-13},
    doi={10.1109/TIV.2023.3288943}}
  ```

 ## Prequisites
Our model is trained and tested under:
- Python 3.7.0
- NVIDIA GPU + CUDA CuDNN
- PyTorch (torch >= 1.2.0)
- scipy
- tqdm
- sklearn
- numba
- cffi
- pypng
- pptk

 ## Usage
 #### Datasets
 We use [KITTI odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) in our experiments. 

 ### Data preprocessing
 - remove the ground points of pointclouds by running `groundtest.m` located in the directory `data_preprocess_zbb/matlab_ground/devkit/matlab` using MATLAB.
 - downsample the pointclouds by running the file `zbb_data_process.py` located in the directory `data_preprocess_zbb/python_downsample` using python.
```
python zbb_data_process.py
```
Pay attention to modifying the file paths.

 ### Training
Train the network by running 
```
python traincomer.py
```
Please reminder to specify the `onlyneedtest`(False), `loadmodel`(False),`dataroot`,`trainset`(sequences for training), `batch_size` in param/titanrtx_1.py.

 ### Testing
Test the network by running 
```
python traincomer.py
```
Please reminder to specify the `onlyneedtest`(True), `loadmodel`(True),`model`(path to HPPLO-Net model), `dataroot`,`testset`(sequences for testing), `testbatch` in param/titanrtx_1.py.

### Acknowledgments

We thank the following open-source projects for the help of the implementations:
- [PointNet++](https://github.com/charlesq34/pointnet2) 
- [KITTI_odometry_evaluation_tool](https://github.com/LeoQLi/KITTI_odometry_evaluation_tool) 
- [PointPWC](https://github.com/DylanWusee/PointPWC)
- [FlowNet3D](https://github.com/xingyul/flownet3d)
