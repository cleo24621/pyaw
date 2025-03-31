# start
from spacepy import coordinates as coord
from spacepy.time import Ticktock
import numpy as np
from geopack import geopack

st_dy = np.datetime64("2016-03-11 01:25:00")
altitude0 = 400
start_lat_dy = 50
start_lon_dy = 100


t = Ticktock(str(st_dy), 'ISO')
c = coord.Coords([altitude0, start_lat_dy, start_lon_dy], 'GDZ', 'sph')
c.ticks = t
c_gsm = c.convert('GSM','car')

# trace dip

ut = st_dy.astype('datetime64[s]').astype(np.int64)
ps = geopack.recalc(ut)
x1gsm_dip,y1gsm_dip,z1gsm_dip,xx_dip,yy_dip,zz_dip = geopack.trace(c_gsm.data[0,0], c_gsm.data[0,1], c_gsm.data[0,2], dir=1)  # 寻找共轭点，由北向南 -> dir = 1
# 追踪是追踪到地表或者达到追踪的极限停止。x1gsm,y1gsm,z1gsm为最后一个点的3个坐标；xx,yy,zz为所有追踪点的3个坐标组成的3个数组（类型为ndarray）

# 转换坐标（点）（GSM->GDZ）
c_back = coord.Coords(np.column_stack((xx_dip, yy_dip, zz_dip)), 'GSM', 'car')
dip_t_back = Ticktock([str(st_dy) for _ in range(len(xx_dip))], 'ISO')
c_back.ticks = dip_t_back
c_gdz = c_back.convert('GDZ','sph')  # 返回alt(km),lat(degree),'lon(degree)'