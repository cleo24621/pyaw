# encoding = UTF-8
"""
@USER: cleo
@DATE: 2024/10/24
@DESCRIPTION: 
"""

# dmsp satellite
# select the needed variables
# ssies3
s3_vars = ['Epoch','glat', 'glon', 'alt', 'vx', 'vxqual', 'vy', 'vyqual', 'vz', 'vzqual',
       'temp', 'tempqual','frach', 'frachqual', 'frache', 'frachequal', 'fraco',
       'fracoqual', 'bx', 'by', 'bz', 'ductdens','te']  # if the info are duplicated, just use one payload (except 'Epoch').
# ssm
ssm_vars = ['Epoch','SC_GEOCENTRIC_LAT','SC_GEOCENTRIC_LON','SC_GEOCENTRIC_R','SC_AACGM_LAT','SC_AACGM_LON','SC_AACGM_LTIME',
'B_SC_OBS_ORIG','DELTA_B_GEO','DELTA_B_SC','SC_ALONG_GEO','AURORAL_REGION','ORBIT_INDEX', 'AURORAL_BOUNDARY_FOM', 'SC_ACROSS_GEO']

# swarm
# miles 2018
# can take a window forward
fs_efi16 = 16.0
fs_vfm50 = 50.0

# fp_e = r"\\Diskstation1\file_three\aw\swarm\A\efi16\sw_efi16A_20160311T000000_20160311T235959_0.pkl"
# fp_b = r"\\Diskstation1\file_three\aw\swarm\A\vfm50\sw_vfm50A_20160311T060000_20160311T070000_0.pkl"
# swarm_start = '20160311T064700'
# swarm_end = '20160311T064900'


# zh1
fgm_vars = ('A221', 'A222', 'A223', 'ALTITUDE', 'B_FGM1', 'B_FGM2', 'B_FGM3',
       'FLAG_MT', 'FLAG_SHW', 'FLAG_TBB', 'GEO_LAT', 'GEO_LON', 'MAG_LAT',
       'MAG_LON', 'UTC_TIME', 'VERSE_TIME')
scm_ulf_vars = ('A231_P', 'A231_W', 'A232_P', 'A232_W', 'A233_P', 'A233_W', 'ALTITUDE', 'FLAG', 'FREQ', 'GEO_LAT', 'GEO_LON', 'MAG_LAT', 'MAG_LON', 'PhaseX', 'PhaseY', 'PhaseZ', 'UTC_TIME', 'VERSE_TIME', 'WORKMODE')
scm_ulf_1c_vars = ('ALTITUDE', 'FLAG', 'GEO_LAT', 'GEO_LON', 'MAG_LAT', 'MAG_LON','UTC_TIME', 'VERSE_TIME', 'WORKMODE')
scm_ulf_resample_vars = ['A231_W','A232_W','A233_W']
efd_ulf_vars = ('A111_P', 'A111_W', 'A112_P', 'A112_W', 'A113_P', 'A113_W', 'ALTITUDE', 'FREQ', 'GEO_LAT', 'GEO_LON', 'MAG_LAT', 'MAG_LON', 'UTC_TIME', 'VERSE_TIME', 'WORKMODE')
efd_ulf_1c_vars = ('ALTITUDE', 'GEO_LAT', 'GEO_LON', 'MAG_LAT', 'MAG_LON', 'UTC_TIME', 'VERSE_TIME', 'WORKMODE')
efd_ulf_resample_vars = ['A111_W','A112_W','A113_W']