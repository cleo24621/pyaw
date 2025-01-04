from pathlib import Path

import pandas as pd
import pywt
from matplotlib import pyplot as plt

import utils_preprocess


def test_align_high2low():
    sdir = Path("./data")
    sfn_EFIA_TCT16 = Path("EFIA_TCT16_20160311T064640_20160311T064920.pkl")
    sfn_MAGA_HR_1B = Path("MAGA_HR_1B_20160311T064640_20160311T064920.pkl")
    df_e = pd.read_pickle(Path(sdir) / Path(sfn_EFIA_TCT16))
    df_b = pd.read_pickle(Path(sdir) / Path(sfn_MAGA_HR_1B))
    timestamps_e = df_e.index.values
    timestamps_b = df_b.index.values
    bn, be, bc = utils_preprocess.get_3arrays(df_b['B_NEC'].values)
    be_align_e = utils_preprocess.align_high2low(be, timestamps_b, timestamps_e)
    plt.plot(timestamps_b, be)
    plt.plot(timestamps_e, be_align_e)
    plt.show()


def test_cwt():
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    coefficients_f, freqs = pywt.cwt(arr, scales, wavelet,
                                     sampling_period=sampling_period)

test_align_high2low()