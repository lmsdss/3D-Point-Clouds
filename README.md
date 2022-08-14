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
### HW4
Use KITTI 3D object detection dataset, select 3 point clouds, do the followings.  
1.Remove the ground from the lidar points. Visualize ground as blue.  
2.Clustering over the remaining points. Visualize the clusters with random colors.
### HW5
Classification over ModelNet40  
1.Build your own network with pytorch  
• PointNet example: https://github.com/fxia22/pointnet.pytorch  
2.ModelNet40 Dataset given by PointNet++:  
• https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip  
3.Follow the training/testing split
4.Remember to add random rotation over z-axis
5.Report testing accuracy