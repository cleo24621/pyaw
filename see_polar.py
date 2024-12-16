# encoding = UTF-8
"""
@USER: cleo
@DATE: 2024/12/12
@DESCRIPTION: 
"""


import cdflib

# Open the CDF file
cdf_file = cdflib.CDF(r"C:\Users\cleo\OneDrive\文档\WeChat Files\wxid_1gotxz13yvpw22\FileStorage\File\2024-12\po_or_pre_19980125_v02.cdf")

# Retrieve information about the CDF
info = cdf_file.cdf_info()
print(info)

# List all variable names
variables = cdf_file.cdf_info()['rVariables']
print(variables)

# Read data from a specific variable
data = cdf_file.varget('VariableName')
print(data)

# Access global attributes
global_attrs = cdf_file.globalattsget()
print(global_attrs)

# Access attributes of a specific variable
var_attrs = cdf_file.varattsget('VariableName')
print(var_attrs)

# Close the CDF file
cdf_file.close()
