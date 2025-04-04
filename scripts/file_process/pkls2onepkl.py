import pandas as pd
import os


# ---basic set
ignore_index = True  # for pkls2onepkl
index = True  # for save as csv
# ---


# 1. 指定包含所有.pkl文件的目录路径
file_directory = r'G:\master\pyaw\scripts\results\aw_cases\temp'  # modify
output_dir = r"G:\master\pyaw\scripts\results\aw_cases\temp"  # modify
output_file_name = 'combined_dataframe_new.pkl'     # 合并后的文件名
output_path = os.path.join(output_dir, output_file_name)
csv_file_name = 'combined_new.csv'
csv_file_path = os.path.join(output_dir,csv_file_name)

# 2. 获取所有.pkl文件路径
pkl_files = [os.path.join(file_directory, f)
             for f in os.listdir(file_directory)
             if f.endswith('.pkl')]

# 3. 读取所有.pkl文件并存储到列表中
dataframes = []
for file in pkl_files:
    df = pd.read_pickle(file)
    dataframes.append(df)

# 4. 合并所有DataFrame（按行合并，重置索引）
combined_df = pd.concat(dataframes, ignore_index=True)

# 5. 保存为新的.pkl文件 to output path
combined_df.to_pickle(output_path)

# save as csv to output path
combined_df.to_csv(csv_file_path,index=True)

print(f"合并完成！共合并 {len(dataframes)} 个文件，结果保存至 {output_file_name}")