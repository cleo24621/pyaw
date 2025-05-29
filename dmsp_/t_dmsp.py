from src.pyaw import SPDF

spdf = SPDF()

fp_s3 = 'D:\cleo\master\pyaw\data\DMSP\ssies3\dmsp-f18_ssies-3_thermal-plasma_201401010124_v01.cdf'  # 一轨
fp_ssm = 'D:\cleo\master\pyaw\data\DMSP\ssm\dmsp-f18_ssm_magnetometer_20140101_v1.0.4.cdf'  # 1天

s3_df = spdf.r_s3(fp_s3)
s3_df_pre = spdf._quality_process(s3_df)

ssm_df = spdf.r_ssm(fp_ssm)
ssm_df_pre = spdf.ssm_pre(ssm_df)

clipped_ssm_df = spdf._clip_ssm_by_ssies3(s3_df_pre, ssm_df_pre)

s3_ssm_df = spdf._get_s3_ssm(s3_df_pre, clipped_ssm_df)