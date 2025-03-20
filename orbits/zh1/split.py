from pathlib import Path

from zh1 import zh1

df_path = Path(r"V:\aw\zh1\efd\ulf\2a\20210401_20210630\CSES_01_EFD_1_L2A_A1_175380_20210401_003440_20210401_010914_000.h5")
efd = zh1.EFDULF(df_path)
dfs = efd.dfs
lats = dfs['GEO_LAT'].squeeze().values
