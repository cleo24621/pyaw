from scipy.signal import butter, buttord


def customize_butter(fs, f_t, f_z, type="lowpass"):
    """

    Args:
        fs: 采样率 (Hz)
        f_t: 通带截止频率 (Hz)。低，例如100
        f_z: 阻带截止频率 (Hz)。高，例如200
        type: ‘lowpass’, ‘highpass’, ‘bandpass’

    Returns:

    """
    # 归一化频率
    wp = f_t / (fs / 2)
    ws = f_z / (fs / 2)

    # 计算阶数（默认 gpass=3dB, gstop=40dB）
    order, wn = buttord(wp, ws, 3, 40)
    return butter(order, wn, type)
