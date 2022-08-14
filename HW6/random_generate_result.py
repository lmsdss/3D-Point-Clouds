import os
import random
import pandas as pd

if __name__ == '__main__':

    input_dir = '../data_object_label/training/label'  # 输入目录:
    output_dir = '../data_object_label/training/results/data'  # 输出目录:

    file_name_list = os.listdir(input_dir)  # 获取输入文件
    for i in file_name_list:
        file_name = os.path.join(input_dir, i)
        label = pd.read_csv(file_name, sep=' ', header=None, engine='python')  #读取数据
        label.columns = [
            'category', 'truncation', 'occlusion', 'alpha','2d_bbox_left', '2d_bbox_top', '2d_bbox_right', '2d_bbox_bottom',
            'height', 'width', 'length', 'location_x', 'location_y', 'location_z', 'rotation',
        ]
        # 修改 2D bounding box 像素位置
        label['2d_bbox_left'] = label['2d_bbox_left'] * 0.99
        label['2d_bbox_top'] = label['2d_bbox_top'] * 0.99
        label['2d_bbox_right'] = label['2d_bbox_right'] * 0.99
        label['2d_bbox_bottom'] = label['2d_bbox_bottom'] * 0.99

        # 修改车的高度，宽度，和长度
        label['height'] = label['height'] * 0.99
        label['width'] = label['width'] * 0.99
        label['length'] = label['length'] * 0.99

        # 修改3D中心在相机坐标下的xyz坐标。
        label['location_x'] = label['location_x'] * 0.99
        label['location_y'] = label['location_y'] * 0.99
        label['location_z'] = label['location_z'] * 0.99

        # 随机生成检测结果 confidence
        label['confidence'] = random.randint(0, 10)
        # 得到输出文件
        output_filename = os.path.join(output_dir, i)
        label.to_csv(output_filename, sep=' ', header=False, index=False)
