import pandas as pd # 导入 pandas 库，用于处理数据和导出 Excel
# 如果没有安装 pandas 和 openpyxl，需要先安装: pip install pandas openpyxl

# 1. 初始化主字典
all_events_data = {}
print("开始收集事件数据...")

# --- 模拟获取第一个事件的数据 ---
event_id_1 = "事件1"
event_data_1 = {
    "事件名称": "服务器CPU告警",         # 字符串
    "发生时间": "2023-10-27 15:00:10", # 时间字符串
    "CPU使用率": 95.5,                 # 浮点型
    "处理状态": "已通知"                # 字符串
}
# 添加到主字典
all_events_data[event_id_1] = event_data_1
print(f"已添加 '{event_id_1}' 的数据.")

# --- 模拟获取第二个事件的数据 ---
event_id_2 = "事件2"
event_data_2 = {
    "事件名称": "应用响应缓慢",           # 字符串
    "发生时间": "2023-10-27 16:20:05", # 时间字符串
    "平均响应时间(秒)": 3.2,             # 浮点型
    "影响范围": "用户登录模块"          # 字符串 (注意：字段可能与事件1不同)
}
# 添加到主字典
all_events_data[event_id_2] = event_data_2
print(f"已添加 '{event_id_2}' 的数据.")

# --- 模拟获取第三个事件的数据 ---
event_id_3 = "事件3"
event_data_3 = {
    "事件名称": "磁盘空间不足",           # 字符串
    "发生时间": "2023-10-27 17:05:30", # 时间字符串
    "剩余空间(GB)": 1.8,               # 浮点型
    "服务器IP": "10.0.0.5",           # 字符串
    "处理状态": "处理中"                # 字符串
}
# 添加到主字典
all_events_data[event_id_3] = event_data_3
print(f"已添加 '{event_id_3}' 的数据.")


print("\n所有事件数据收集完毕.")
# 可以取消注释下面两行来查看最终的字典结构
# import pprint
# pprint.pprint(all_events_data)

# --- 导出到 XLSX 文件 ---

# 准备要写入 Excel 的数据格式 (pandas DataFrame 最常用的输入格式之一：字典列表)
data_for_df = []
for event_id, event_details in all_events_data.items():
    # 为每一行创建一个字典，包含事件ID和该事件的所有详情
    row_dict = {'事件ID': event_id}  # 添加 '事件ID' 列
    row_dict.update(event_details) # 将事件详情字典合并进来
    data_for_df.append(row_dict)

# 设置输出的 Excel 文件名
xlsx_file_path = 'simplified_events_data.xlsx'
print(f"\n准备导出数据到 XLSX 文件: {xlsx_file_path}")

try:
    # 使用 pandas 将字典列表直接转换为 DataFrame
    # Pandas 会自动处理不同事件可能包含不同字段的情况，
    # 缺失的字段在DataFrame中会显示为 NaN，导出到Excel中通常是空格
    df = pd.DataFrame(data_for_df)

    # 将 DataFrame 写入 Excel 文件
    # index=False 表示不将 DataFrame 的行索引 (0, 1, 2...) 写入到 Excel 文件中
    # engine='openpyxl' 指定使用 openpyxl 引擎来写入 .xlsx 文件
    df.to_excel(xlsx_file_path, index=False, engine='openpyxl')

    print(f"成功导出到 XLSX 文件: {xlsx_file_path}")

except ImportError:
    print("\n错误：导出到 XLSX 需要 pandas 和 openpyxl 库。")
    print("请确保已安装：运行 'pip install pandas openpyxl'")
except Exception as e:
    print(f"\n错误：导出到 XLSX 文件失败: {e}")