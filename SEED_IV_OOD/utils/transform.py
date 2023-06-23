import numpy as np
from scipy.signal import butter, lfilter

# BandDifferentialEntropy------------------------------------------------------------------------
def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

class BandDifferentialEntropy:
    def __init__(self, sampling_rate = 128, order= 5,
                 band_dict = {"theta": [4, 8],
                             "alpha": [8, 14],
                             "beta": [14, 31],
                             "gamma": [31, 49]}):

        self.sampling_rate = sampling_rate
        self.order = order
        self.band_dict = band_dict

    def apply(self, eeg):
        band_list = []
        for low, high in self.band_dict.values():
            c_list = []
            for c in eeg:
                b, a = butter_bandpass(low, high, fs=self.sampling_rate, order=self.order) # 필터만들기
                c_list.append(self.opt(lfilter(b, a, c)))  # 원래 신호 c 에 필터링하기
            c_list = np.array(c_list)
            band_list.append(c_list)
        return np.stack(band_list, axis=-1)

    def opt(self, eeg):
        return 1 / 2 * np.log2(2 * np.pi * np.e * np.std(eeg))

# 3D-> 4D-----------------------------------------------------------------------------------------
def format_channel_location_dict(channel_list, location_list):
    location_list = np.array(location_list)
    output = {}
    for channel in channel_list:
        if len(np.argwhere(location_list == channel)):
            location = (np.argwhere(location_list == channel)[0]).tolist()
            output[channel] = location
    return output

class ToGrid:
    def __init__(self, channel_location_dict):
        self.channel_location_dict = channel_location_dict

        loc_x_list = []
        loc_y_list = []
        for _, (loc_y, loc_x) in channel_location_dict.items():
            loc_x_list.append(loc_x)
            loc_y_list.append(loc_y)
        self.width = max(loc_x_list) + 1
        self.height = max(loc_y_list) + 1

    def apply(self, eeg):
        # num_electrodes x timestep
        outputs = np.zeros([self.height, self.width, eeg.shape[-1]])
        # 9 x 9 x timestep
        for i, (loc_y, loc_x) in enumerate(self.channel_location_dict.values()):
            outputs[loc_y][loc_x] = eeg[i]

        outputs = outputs.transpose(2, 0, 1)
        # timestep x 9 x 9
        return outputs

    def reverse(self, eeg):
        # timestep x 9 x 9
        eeg = eeg.transpose(1, 2, 0)
        # 9 x 9 x timestep
        num_electrodes = len(self.channel_location_dict)
        outputs = np.zeros([num_electrodes, eeg.shape[2]])
        for i, (x, y) in enumerate(self.channel_location_dict.values()):
            outputs[i] = eeg[x][y]
        # num_electrodes x timestep
        return outputs 