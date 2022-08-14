import pandas as pd
import pandas
import open3d as o3d
import open3d
import numpy as np
import numpy
import copy
from scipy.spatial.transform import Rotation


class pointcloud:
    def __init__(self):
        pass

    # 从文件中读取点云
    @staticmethod
    def read_pointcloud(file_name: str) -> open3d.geometry.PointCloud:
        df = pd.read_csv(file_name, header=None)
        df.columns = ["x", "y", "z",
                      "nx", "ny", "nz"]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(df[["x", "y", "z"]].values)
        pcd.normals = o3d.utility.Vector3dVector(df[["nx", "ny", "nz"]].values)

        return pcd

    # 读取bin格式的点云
    @staticmethod
    def read_point_cloud_bin(bin_path: str) -> open3d.geometry.PointCloud:
        data = np.fromfile(bin_path, dtype=np.float32)

        # 将数据重新格式化
        N, D = data.shape[0] // 6, 6
        
        point_cloud_with_normal = np.reshape(data, (N, D))

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_with_normal[:, 0:3])
        point_cloud.normals = o3d.utility.Vector3dVector(point_cloud_with_normal[:, 3:6])

        return point_cloud


class Homework:
    def __init__(self) -> None:
        pass

    # 以pandas的dataframe格式读取结果文件
    @staticmethod
    def read_registration_results(results_path: str) -> pandas.DataFrame:
        df_results = pd.read_csv(
            results_path
        )

        return df_results

    # 使用与输入相同的格式设置输出格式
    @staticmethod
    def init_output() -> dict:
        df_output = {
            'idx1': [],
            'idx2': [],
            't_x': [],
            't_y': [],
            't_z': [],
            'q_w': [],
            'q_x': [],
            'q_y': [],
            'q_z': []
        }

        return df_output

    # 绘制配准后的点云
    @staticmethod
    def draw_registration_result(source: open3d.geometry.PointCloud, target: open3d.geometry.PointCloud,
                                 transformation: numpy.matrix):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                          zoom=0.4559,
                                          front=[0.6452, -0.3036, -0.7011],
                                          lookat=[1.9892, 2.0208, 1.8945],
                                          up=[-0.2779, -0.9482, 0.1556])

    # 向结果dataframe中写入数据
    @staticmethod
    def add_to_output(df_output: dict, idx1: str, idx2: str, T: numpy.matrix) -> None:
        """
        Add record to output
        """

        def format_transform_matrix(T):
            r = Rotation.from_matrix(T[:3, :3])
            q = r.as_quat()
            t = T[:3, 3]

            return (t, q)

        df_output['idx1'].append(idx1)
        df_output['idx2'].append(idx2)

        (t, q) = format_transform_matrix(T)

        # translation:
        df_output['t_x'].append(t[0])
        df_output['t_y'].append(t[1])
        df_output['t_z'].append(t[2])
        # rotation:
        df_output['q_w'].append(q[3])
        df_output['q_x'].append(q[0])
        df_output['q_y'].append(q[1])
        df_output['q_z'].append(q[2])

    @staticmethod
    def write_output(filename: str, df_output: dict) -> None:
        df_output = pd.DataFrame.from_dict(
            df_output
        )

        print(f'write output to {filename}')
        df_output[
            [
                'idx1', 'idx2',
                't_x', 't_y', 't_z',
                'q_w', 'q_x', 'q_y', 'q_z'
            ]
        ].to_csv(filename, index=False)
