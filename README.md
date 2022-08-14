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
• Numpy brute-force search  
• scipy.spatial.KDTree  
• Your own kd-tree/octree in python or C++
### HW3
1.Generate clustering dataset using sklearn.  
2.Implement your own version of.  
• Means  
• GMM  
• Spectral Clustering  
• Visualize and compare the results with the standard results
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
### HW6  
1.Setup the KITTI object detection evaluation environment  
• git clone https://github.com/prclibo/kitti_eval.git  
• g++ -O3 -DNDEBUG -o evaluate_object_3d_offline evaluate_object_3d_offline.cpp  
• sudo apt-get install gnuplot     
• sudo apt-get install texlive-extra-utils  
2.Download and read the KITTI Object Detection dataset ”devkit” readme.   
3.Divide the KITTI Object Detection into training set and validation set.
4.Generate object detection results on KITTI validation set  
Option 1: find any open-source 3d object detector, run it.  
Option 2: copy the ground truth as the result, but you need to process it into the correct format.  
### HW7
Implement your own ISS keypoint detection.  
• Apply your own ISS on ModelNet40 (choose 3 objects from different categories)  
• Visualize the object and the keypoints together.  
• Submit your code and the visualize screenshots.  
### HW8
Implement feature descriptors FPFH, SHOT  
• You may call PCL library via python binding OR implement your own version.  
• Example of python binding of PCL: https://github.com/lijx10/PCLKeypoints  
• Test with ModelNet40 to ensure correctness.
### HW9
1.Implement feature detectors & descriptors  
• Any algorithm you want  
• You may call APIs. But still, your own implementation is preferred.  
2.Implement your own ICP or NDT.  
• Do NOT call APIs except for nearest neighbor search.  
3.Test your registration algorithm on the provided dataset  
• There is NO proper initialization provided.  
• Report the following metrics. Evaluation script is provided.  
4.We provide the registration dataset that contains 342 pairs of point clouds.  
5.You are required to provide your registration results into “reg_result.txt”  
• The original “reg_result.txt” is an example with 3 ground truth results.  
• The rest of 339 ground truth results are not provided.  
6.There is the “evaluate_rt.py”, it provides  
• Functions to read and visualize the pairs  
• Functions to evaluate the RRE, RTE, success rate  
