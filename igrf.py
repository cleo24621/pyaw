# encoding = UTF-8
"""
@USER: cleo
@DATE: 2024/12/10
@DESCRIPTION: 
"""

import pandas as pd
from astropy.time import Time
from scipy import interpolate

import igrf_utils as iut

# Load in the file of coefficients
IGRF_FILE = r'IGRF13.shc'
igrf = iut.load_shcfile(IGRF_FILE, None)

f = interpolate.interp1d(igrf.time, igrf.coeffs, fill_value='extrapolate')


def to_decimal_date_astropy(datetimes):
    """
    type transform: transform to decimal dates using astropy
    :param datetimes: pd.Series consisting of pd.Timestamp or ndarray consisting of pd.Timestamp
    :return: pd.Series consisting of decimal dates
    """
    time_objects = Time(datetimes.tolist())  # Convert to AstroPy Time objects
    return pd.Series(time_objects.decimalyear)  # Access decimal year attribute


# assist_file_path = 'wu2020_B_20140219T0231_20140219T0241_assist_IGRF.pkl'
# df_b = pd.read_pickle(assist_file_path)
# decimal_dates = to_decimal_date_astropy(df_b.index)
#
# LTDs = df_b['Latitude'].values
# LNDs = df_b['Longitude'].values
# alts = df_b['Radius'].values / 1e3  # km
#
# Xs = []
# Ys = []
# Zs = []
# for date,LTD,LND,alt in zip(decimal_dates,LTDs,LNDs,alts):
#     coeffs = f(date)
#     latd = iut.check_float(LTD)
#     lond = iut.check_float(LND)
#     lat, lon = iut.check_lat_lon_bounds(latd,0,lond,0)
#     colat = 90-lat
#     alt = iut.check_float(alt)
#     # Compute the main field B_r, B_theta and B_phi value for the location(s)
#     Br, Bt, Bp = iut.synth_values(coeffs.T, alt, colat, lon,
#                               igrf.parameters['nmax'])
#     X = -Bt; Y = Bp; Z = -Br
#     print('North component (X)     :', '{: .1f}'.format(X), 'nT')
#     print('East component (Y)      :', '{: .1f}'.format(Y), 'nT')
#     print('Vertical component (Z)  :', '{: .1f}'.format(Z), 'nT')
#     Xs.append(X)
#     Ys.append(Y)
#     Zs.append(Z)
# df = pd.DataFrame(data={'IGRF_B_N': Xs, 'IGRF_B_E': Ys, 'IGRF_B_C': Zs})
# save_filepath = r'wu2020_IGRF_B_20140219T0231_20140219T0241.pkl'
# if os.path.exists(save_filepath):
#     print("File exists!")
# else:
#     df.to_pickle(save_filepath)

def get_B_IGRF_NEC(datetimes, latitudes, longitudes, altitudes,print_:None=False):
    """
    get magnetic field in NEC coordinates
    :param datetimes: pd.Series consisting of pd.Timestamp or ndarray consisting of pd.Timestamp
    :param latitudes: geocentric latitudes
    :param longitudes: geocentric longitudes
    :param altitudes: the distance from the earth's center, unit in km.
    :return: magnetic field in NEC coordinates
    """
    Xs = []
    Ys = []
    Zs = []
    decimal_dates = to_decimal_date_astropy(datetimes)
    for date, latitude, longitude, altitude in zip(decimal_dates, latitudes, longitudes, altitudes):
        coeffs = f(date)
        latd = iut.check_float(latitude)
        lond = iut.check_float(longitude)
        lat, lon = iut.check_lat_lon_bounds(latd, 0, lond, 0)
        colat = 90 - lat
        altitude = iut.check_float(altitude)
        # Compute the main field B_r, B_theta and B_phi value for the location(s)
        Br, Bt, Bp = iut.synth_values(coeffs.T, altitude, colat, lon, igrf.parameters['nmax'])
        X = -Bt
        Y = Bp
        Z = -Br
        if print_:
            print('North component (X)     :', '{: .1f}'.format(X), 'nT')
            print('East component (Y)      :', '{: .1f}'.format(Y), 'nT')
            print('Vertical component (Z)  :', '{: .1f}'.format(Z), 'nT')
        Xs.append(X)
        Ys.append(Y)
        Zs.append(Z)
    return Xs, Ys, Zs
