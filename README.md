# SfM-Ceres
Simple SfM implementation using Ceres in python 

- Extracted and matched SIFT keypoints and create Adjacency Matrix for best pairs.
- Choose initial pair with many matches and large baseline (verified via essential matrix and RANSAC).
- Incrementally added views via PnP camera pose estimation and linear triangulation.
- Optimized camera parameters and 3D coordinates with Bundle Adjustment using PyCeres.
- Optimized runtime with Numba JIT; visualize results using Open3D.

run the code using,

```
pip install -r requirements.txt
python main.py
```

reconstruction resutls,

<img src="media/templeRing.png">


Working on the C++ version. 
