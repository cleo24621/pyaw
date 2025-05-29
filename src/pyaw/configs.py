from datetime import timedelta
from pathlib import Path

# ---
# The working directory is $ProjectFileDir$
DATA_DIR = Path("./data")


class DMSPConfigs:
    class SPDF:
        # select the needed variables
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

        vx_qual_filter = 4  # quality filter of v (velocity of ion?
        vy_qual_filter = 4
        vz_qual_filter = 4
        vx_valid_value = 2000  # valid min and max of v
        vy_valid_value = 2000
        vz_valid_value = 2000
        ductdens_valid_value_min = 0  # number density of ion
        ductdens_valid_value_max = 1e8
        frac_qual_filter = 4  # fraction of different ions
        frac_valid_value_min = 0
        frac_valid_value_max = 1.05
        temp_qual_filter = 4  # ion temperature
        temp_valid_value_min = 500
        temp_valid_value_max = 2e4
        te_valid_value_min = 500  # electron temperature

        te_valid_value_max = 1e4

        ssies3_orbit_time = timedelta(
            hours=1, minutes=45
        )  # little greater than the real orbit time


class Zh1Configs:
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
    scmulf_fs = 1024

    efdulf_fs = 125

    indicator_descend_ascend_dict = {
        "0": "descending",
        "1": "ascending",
    }  # ascending (south to north); descending (north to south)
