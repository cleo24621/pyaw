# encoding = UTF-8
"""
@USER: cleo
@DATE: 2024/10/24
@DESCRIPTION:
"""


class SwarmConfig:
    tct16_fs = 16
    mag_hr_fs = 50
    mag_lr_fs = 1

    class ViresSwarmRequest:
        auxiliaries = [
            "AscendingNodeLongitude",
            "QDLat",
            "QDLon",
            "QDBasis",
            "MLT",
            "SunDeclination",
        ]
        models = ["IGRF"]
        collections = {
            "EFI_TCT16": [
                "SW_EXPT_EFIA_TCT16",
                "SW_EXPT_EFIB_TCT16",
                "SW_EXPT_EFIC_TCT16",
            ],
            "MAG": ["SW_OPER_MAGA_LR_1B", "SW_OPER_MAGB_LR_1B", "SW_OPER_MAGC_LR_1B"],
            "MAG_HR": [
                "SW_OPER_MAGA_HR_1B",
                "SW_OPER_MAGB_HR_1B",
                "SW_OPER_MAGC_HR_1B",
            ],
        }


class DmspConfig:
    # select the needed variables
    ## ssies3
    ssies3_vars = [
        "Epoch",
        "glat",
        "glon",
        "alt",
        "vx",
        "vxqual",
        "vy",
        "vyqual",
        "vz",
        "vzqual",
        "temp",
        "tempqual",
        "frach",
        "frachqual",
        "frache",
        "frachequal",
        "fraco",
        "fracoqual",
        "bx",
        "by",
        "bz",
        "ductdens",
        "te",
    ]  # if the info are duplicated, just use one payload (except 'Epoch').
    ## ssm
    ssm_vars = [
        "Epoch",
        "SC_GEOCENTRIC_LAT",
        "SC_GEOCENTRIC_LON",
        "SC_GEOCENTRIC_R",
        "SC_AACGM_LAT",
        "SC_AACGM_LON",
        "SC_AACGM_LTIME",
        "B_SC_OBS_ORIG",
        "DELTA_B_GEO",
        "DELTA_B_SC",
        "SC_ALONG_GEO",
        "AURORAL_REGION",
        "ORBIT_INDEX",
        "AURORAL_BOUNDARY_FOM",
        "SC_ACROSS_GEO",
    ]


class Zh1Config:
    fgm_vars = (
        "A221",
        "A222",
        "A223",
        "ALTITUDE",
        "B_FGM1",
        "B_FGM2",
        "B_FGM3",
        "FLAG_MT",
        "FLAG_SHW",
        "FLAG_TBB",
        "GEO_LAT",
        "GEO_LON",
        "MAG_LAT",
        "MAG_LON",
        "UTC_TIME",
        "VERSE_TIME",
    )
    scm_ulf_vars = (
        "A231_P",
        "A231_W",
        "A232_P",
        "A232_W",
        "A233_P",
        "A233_W",
        "ALTITUDE",
        "FLAG",
        "FREQ",
        "GEO_LAT",
        "GEO_LON",
        "MAG_LAT",
        "MAG_LON",
        "PhaseX",
        "PhaseY",
        "PhaseZ",
        "UTC_TIME",
        "VERSE_TIME",
        "WORKMODE",
    )
    scm_ulf_1c_vars = (
        "ALTITUDE",
        "FLAG",
        "GEO_LAT",
        "GEO_LON",
        "MAG_LAT",
        "MAG_LON",
        "UTC_TIME",
        "VERSE_TIME",
        "WORKMODE",
    )
    scm_ulf_resample_vars = ["A231_W", "A232_W", "A233_W"]
    efd_ulf_vars = (
        "A111_P",
        "A111_W",
        "A112_P",
        "A112_W",
        "A113_P",
        "A113_W",
        "ALTITUDE",
        "FREQ",
        "GEO_LAT",
        "GEO_LON",
        "MAG_LAT",
        "MAG_LON",
        "UTC_TIME",
        "VERSE_TIME",
        "WORKMODE",
    )
    efd_ulf_1c_vars = (
        "ALTITUDE",
        "GEO_LAT",
        "GEO_LON",
        "MAG_LAT",
        "MAG_LON",
        "UTC_TIME",
        "VERSE_TIME",
        "WORKMODE",
    )
    efd_ulf_resample_vars = ["A111_W", "A112_W", "A113_W"]


class Parameters:
    class PhysicalParameters:
        # unified unit

        mu0 = 1.256637061e-6  # (H/m SI) vacuum permeability
        epsilon0 = 8.8541878188e-12  # (F/m SI) vacuum permittivity
        kB = 1.380649e-23  # (J/K SI) Boltzmann constant
        e = 1.602176634e-19  # (C SI) elementary charge: https://en.wikipedia.org/wiki/Electric_charge#Unit

        me = 9.11e-31  # Electron mass in kg
        mp = 1.673e-27  # Proton mass in kg
        rH = 1.0080  # Hydrogen relative atomic mass
        rHE = 4.0026  # Helium relative atomic mass
        rO = 15.999  # Oxygen relative atomic mass
        mH = mp * rH
        mHe = mp * rHE
        mO = mp * rO
        # refer to
        # MU0: https://en.wikipedia.org/wiki/Permeability_(electromagnetism)
        # EPSILON0: https://en.wikipedia.org/wiki/Vacuum_permittivity
        # K_B: https://en.wikipedia.org/wiki/Boltzmann_constant#
        # M_E: https://en.wikipedia.org/wiki/Orders_of_magnitude_(mass)#The_least_massive_things:_below_10%E2%88%9224_kg
        # M_P: https://en.wikipedia.org/wiki/Orders_of_magnitude_(mass)#The_least_massive_things:_below_10%E2%88%9224_kg
        # R_O: relative atomic mass or standard atomic weight: https://en.wikipedia.org/wiki/List_of_chemical_elements

    class AlfvenWaveParameters:

        general_dynamic_va = 1.4e6  # General dynamic Alfven speed
        general_dynamic_Sigma_P = 3.0  # General dynamic Sigma_P

        general_static_va = 1.3e6  # General static Alfven speed
        general_static_Sigma_P = 0.5  # General static Sigma_P

        def __init__(self):
            self.general_dynamic_lower_bound = self.calculate_lower_bound(self.general_dynamic_Sigma_P)
            self.general_dynamic_upper_bound = self.calculate_upper_bound(self.general_dynamic_va,self.general_dynamic_Sigma_P)

            self.general_static_lower_bound = self.calculate_lower_bound(self.general_static_Sigma_P)
            self.general_static_upper_bound = self.calculate_upper_bound(self.general_static_va,self.general_static_Sigma_P)

        def calculate_lower_bound(self,Sigma_P):
            mu0 = Parameters.PhysicalParameters.mu0
            return 1 / (mu0 * Sigma_P)

        def calculate_upper_bound(self,va,Sigma_P):
            mu0 = Parameters.PhysicalParameters.mu0
            return mu0 * va ** 2 * Sigma_P


# dmsp satellite
# select the needed variables
# ssies3
ssies3_vars = [
    "Epoch",
    "glat",
    "glon",
    "alt",
    "vx",
    "vxqual",
    "vy",
    "vyqual",
    "vz",
    "vzqual",
    "temp",
    "tempqual",
    "frach",
    "frachqual",
    "frache",
    "frachequal",
    "fraco",
    "fracoqual",
    "bx",
    "by",
    "bz",
    "ductdens",
    "te",
]  # if the info are duplicated, just use one payload (except 'Epoch').
# ssm
ssm_vars = [
    "Epoch",
    "SC_GEOCENTRIC_LAT",
    "SC_GEOCENTRIC_LON",
    "SC_GEOCENTRIC_R",
    "SC_AACGM_LAT",
    "SC_AACGM_LON",
    "SC_AACGM_LTIME",
    "B_SC_OBS_ORIG",
    "DELTA_B_GEO",
    "DELTA_B_SC",
    "SC_ALONG_GEO",
    "AURORAL_REGION",
    "ORBIT_INDEX",
    "AURORAL_BOUNDARY_FOM",
    "SC_ACROSS_GEO",
]

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
fgm_vars = (
    "A221",
    "A222",
    "A223",
    "ALTITUDE",
    "B_FGM1",
    "B_FGM2",
    "B_FGM3",
    "FLAG_MT",
    "FLAG_SHW",
    "FLAG_TBB",
    "GEO_LAT",
    "GEO_LON",
    "MAG_LAT",
    "MAG_LON",
    "UTC_TIME",
    "VERSE_TIME",
)
scm_ulf_vars = (
    "A231_P",
    "A231_W",
    "A232_P",
    "A232_W",
    "A233_P",
    "A233_W",
    "ALTITUDE",
    "FLAG",
    "FREQ",
    "GEO_LAT",
    "GEO_LON",
    "MAG_LAT",
    "MAG_LON",
    "PhaseX",
    "PhaseY",
    "PhaseZ",
    "UTC_TIME",
    "VERSE_TIME",
    "WORKMODE",
)
scm_ulf_1c_vars = (
    "ALTITUDE",
    "FLAG",
    "GEO_LAT",
    "GEO_LON",
    "MAG_LAT",
    "MAG_LON",
    "UTC_TIME",
    "VERSE_TIME",
    "WORKMODE",
)
scm_ulf_resample_vars = ["A231_W", "A232_W", "A233_W"]
efd_ulf_vars = (
    "A111_P",
    "A111_W",
    "A112_P",
    "A112_W",
    "A113_P",
    "A113_W",
    "ALTITUDE",
    "FREQ",
    "GEO_LAT",
    "GEO_LON",
    "MAG_LAT",
    "MAG_LON",
    "UTC_TIME",
    "VERSE_TIME",
    "WORKMODE",
)
efd_ulf_1c_vars = (
    "ALTITUDE",
    "GEO_LAT",
    "GEO_LON",
    "MAG_LAT",
    "MAG_LON",
    "UTC_TIME",
    "VERSE_TIME",
    "WORKMODE",
)
efd_ulf_resample_vars = ["A111_W", "A112_W", "A113_W"]
