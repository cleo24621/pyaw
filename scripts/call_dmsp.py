# # fp_s3 = r"D:\cleo\master\pyaw\data\dmsp-f18_ssies-3_thermal-plasma_201401010124_v01.cdf"
# # fp_ssm = r"D:\cleo\master\pyaw\data\dmsp-f18_ssm_magnetometer_20140101_v1.0.4.cdf"
# # spdf = SPDF()
# # s3_df = spdf.r_s3(fp_s3)
# # s3_df_pre = spdf.s3_pre(s3_df)
# # ssm_df = spdf.r_ssm(fp_ssm)
# # ssm_df_pre = spdf.ssm_pre(ssm_df)
# # clipped_ssm_df = spdf.clip_ssm_by_ssies3(s3_df_pre, ssm_df_pre)
# # s3_ssm_df = spdf.get_s3_ssm(s3_df_pre, clipped_ssm_df)
# # spdf.get_E(s3_ssm_df[['v_s3_sc1', 'v_s3_sc2', 'v_s3_sc3']],
# #            s3_ssm_df[['b_s3_sc_orig1', 'b_s3_sc_orig2', 'b_s3_sc_orig3']])
#
# # plt.figure()
# # assert clipped_ssm_d.index.equals(s3_d_pre.index)
# # time = clipped_ssm_d.index
# # plt.plot(time,s3_d_pre['glat'],time,clipped_ssm_d['sc_geocentric_lat'])
# # plt.show()
#
# # plt.figure()
# # time = s3_ssm_df.index
# # plt.plot(time, s3_ssm_df['bx'])
# # plt.show()