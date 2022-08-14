import os
import argparse
import progressbar

import Utils
import ISS
import FPFH
import RANSAC


def get_args():
    parase = argparse.ArgumentParser("Registration")
    parase.add_argument("--dir", type=str, default="./registration_dataset")
    parase.add_argument("--radius", type=float, default=0.5)

    return parase.parse_args()


if __name__ == "__main__":

    args = get_args()
    datasets_dir = args.dir
    radius = args.radius

    # 进度条
    progress = progressbar.ProgressBar()

    # 以pandas.dataframe格式读取结果文件
    registration_results = Utils.Homework.read_registration_results(os.path.join(datasets_dir, "reg_result.txt"))

    # 初始化输出文件结构
    df_output = Utils.Homework.init_output()

    # 迭代reg_result中的每一行来获取需要配准的点云文件
    for index, row in progress(list(registration_results.iterrows())):
        idx_source = int(row["idx2"])
        idx_target = int(row["idx1"])

        # 读取点云,输出格式为open3d的点云格式
        pcd_source = Utils.pointcloud.read_point_cloud_bin(
            os.path.join(datasets_dir, "point_clouds", f"{idx_source}.bin"))
        pcd_target = Utils.pointcloud.read_point_cloud_bin(
            os.path.join(datasets_dir, "point_clouds", f"{idx_target}.bin"))

        # 移除指定范围内没有邻居的外点
        pcd_source, ind = pcd_source.remove_radius_outlier(nb_points=4, radius=radius)
        pcd_target, ind = pcd_target.remove_radius_outlier(nb_points=4, radius=radius)

        # 特征点检测
        source_detector = ISS.ISS_detector()
        source_detector.set_pointcloud(pcd_source)
        source_detector.detect()
        keypoints_source = source_detector.get_feature_points()

        target_detector = ISS.ISS_detector()
        target_detector.set_pointcloud(pcd_target)
        target_detector.detect()
        keypoints_target = target_detector.get_feature_points()

        # 提取描述子
        source_descriptor = FPFH.FPFH_decriptor()
        source_descriptor.set_pointclouds(pcd_source)
        source_descriptor.set_keypoints(keypoints_source)
        source_descriptor.describe()
        source_fpfh = source_descriptor.get_descriptors()

        target_descriptor = FPFH.FPFH_decriptor()
        target_descriptor.set_pointclouds(pcd_target)
        target_descriptor.set_keypoints(keypoints_target)
        target_descriptor.describe()
        target_fpfh = target_descriptor.get_descriptors()

        # 特征点云
        pcd_source = pcd_source.select_down_sample(list(source_detector.get_feature_index()))
        pcd_target = pcd_target.select_down_sample(list(target_detector.get_feature_index()))

        # 位姿估计器
        ransac_icp = RANSAC.RANSAC_ICP.Builder().set_max_iteration(10000).build()

        # 配置信息
        print(ransac_icp)

        # 设置输入
        ransac_icp.set_source_pointscloud(pcd_source)
        ransac_icp.set_target_pointscloud(pcd_target)
        ransac_icp.set_source_features(source_fpfh.T)
        ransac_icp.set_target_features(target_fpfh.T)

        # 匹配
        result = ransac_icp.ransac_match()

        Utils.Homework.add_to_output(df_output, idx_target, idx_source, result.transformation)

    Utils.Homework.write_output(
        os.path.join(datasets_dir, 'xxx.txt'),
        df_output
    )
