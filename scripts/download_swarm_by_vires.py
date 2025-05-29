"""Download Swarm data from viresclient.SwarmRequest."""

import time

from pyaw.swarm import download_orbit_collection

satellite = "A"

# collection = "SW_EXPT_EFIA_TCT16"
# collection = "SW_OPER_MAGA_HR_1B"
# collection = "SW_PREL_EFIAIDM_2_"
# collection = "SW_OPER_EFIA_LP_1B"
collection = "SW_OPER_EFIATIE_2_"

assert satellite in collection

orbit_nums = [12885]
# orbit_nums = range(12895,12901+1)

download_types = ["measurements"]
# download_types = ["auxiliaries"]
# download_types = ["igrf"]

for orbit_num in orbit_nums:
    for download_type in download_types:
        download_orbit_collection(
            satellite=satellite,
            collection=collection,
            orbit_number=orbit_num,
            download_type=download_type,
        )
        time.sleep(1)
