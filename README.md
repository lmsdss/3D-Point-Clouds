# 3D-Point-Clouds
## Introduction
We provide reference code for 3D point cloud processing.  
Here are the recommended 3D point cloud processing courses.  
https://www.shenlanxueyuan.com/course/501  
### HW1
1.Download ModelNet40 dataset  
2.Perform PCA for the 40 objects, visualize it.  
3.Perform surface normal estimation for each point of each object, visualize it.  
4.Downsample each object using voxel grid downsampling (exact, both centroid & 
random). Visualize the results.  
### HW2
1.We provide one 𝑁 × 3 point cloud,8-NN search for each point to the point cloud.  
2.Implement 3 NN algorithms.  
a.Numpy brute-force search  
b.scipy.spatial.KDTree  
c.Your own kd-tree/octree in python or C++
### HW3
1.Generate clustering dataset using sklearn.  
2.Implement your own version of.  
a.Means  
b.GMM  
c.Spectral Clustering  
3.Visualize and compare the results with the standard results