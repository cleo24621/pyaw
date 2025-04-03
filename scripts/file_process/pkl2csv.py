import os
import glob
import pandas as pd


def convert_pkl_to_csv(folder_path):
    """
    将指定文件夹下的所有 .pkl 文件转换为 .csv 文件。

    参数:
    folder_path (str): 包含 .pkl 文件的文件夹路径。
    """
    print("开始转换...")
    # 查找文件夹中所有的 .pkl 文件
    pkl_files = glob.glob(os.path.join(folder_path, '*.pkl'))

    if not pkl_files:
        print("文件夹中没有 .pkl 文件。")
        return

    # 遍历每个 .pkl 文件并进行转换
    for pkl_file in pkl_files:
        try:
            # 读取 .pkl 文件为 pandas DataFrame
            df = pd.read_pickle(pkl_file)

            # 生成对应的 .csv 文件名
            csv_file = os.path.splitext(pkl_file)[0] + '.csv'

            # 将 DataFrame 保存为 .csv 文件，不保存索引
            df.to_csv(csv_file, index=False)

            print(f"成功转换 {pkl_file} 到 {csv_file}")
        except Exception as e:
            print(f"转换 {pkl_file} 时出错: {e}")

    print("转换完成。")


# 示例用法
if __name__ == "__main__":
    folder_path = r'G:\master\pyaw\scripts\results\aw_cases'  # 请替换为你的文件夹路径
    convert_pkl_to_csv(folder_path)