import os


def get_1d_file_names(file_dir, condition="20210401"):
    # 轨道号和升降轨号一样的文件可能不唯一，例如‘175381’，此时选择较大的那个文件，因为包含更多的信息
    # 获取所有符合条件的文件（20210401）
    files = [f for f in os.listdir(file_dir) if condition in f and f.endswith(".h5")]

    # 创建字典存储每个轨道号的最大文件
    orbit_files = {}

    for filename in files:
        # 提取轨道号（第七个下划线分隔字段）
        parts = filename.split("_")
        if len(parts) >= 7:
            orbit = parts[6]
            file_path = os.path.join(file_dir, filename)
            size = os.path.getsize(file_path)

            # 更新字典中该轨道号的最大文件
            if orbit not in orbit_files or size > orbit_files[orbit]["size"]:
                orbit_files[orbit] = {"name": filename, "size": size}
    return [i["name"] for i in orbit_files.values()]
