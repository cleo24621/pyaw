# encoding = UTF-8
"""
@USER: cleo
@DATE: 2024/10/24
@DESCRIPTION: 
"""

# select the needed variables
s3_vars = ['Epoch','glat', 'glon', 'alt', 'vx', 'vxqual', 'vy', 'vyqual', 'vz', 'vzqual',
       'temp', 'tempqual','frach', 'frachqual', 'frache', 'frachequal', 'fraco',
       'fracoqual', 'bx', 'by', 'bz', 'ductdens','te']  # if the info are duplicated, just use one payload (except 'Epoch').
ssm_vars = ['Epoch', 'B_SC_OBS_ORIG','DELTA_B_GEO','DELTA_B_SC','SC_ALONG_GEO','AURORAL_REGION','ORBIT_INDEX', 'AURORAL_BOUNDARY_FOM', 'SC_ACROSS_GEO']

# swarm
# miles 2018
# can take a window forward
fs_efi16 = 16.0
fs_vfm50 = 50.0

# fp_e = r"\\Diskstation1\file_three\aw\swarm\A\efi16\sw_efi16A_20160311T000000_20160311T235959_0.pkl"
# fp_b = r"\\Diskstation1\file_three\aw\swarm\A\vfm50\sw_vfm50A_20160311T060000_20160311T070000_0.pkl"
swarm_start = '20160311T064700'
swarm_end = '20160311T064900'