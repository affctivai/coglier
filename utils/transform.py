import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal.windows import hann
from sklearn.preprocessing import StandardScaler, RobustScaler

'''
References

[1] PSD, DE: Alarc ̃ao, S. M., Fonseca, M. J.:Emo-tions recognition using eeg signals: A survey. IEEE Transactions on Affective Computing 2019: 10(3):374–393

[2] Transform implementation: TorchEEG
    Related Project: https://github.com/torcheeg/torcheeg
'''

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
                b, a = butter_bandpass(low, high, fs=self.sampling_rate, order=self.order)
                c_list.append(self.opt(lfilter(b, a, c)))
            c_list = np.array(c_list)
            band_list.append(c_list)
        return np.stack(band_list, axis=-1)

    def opt(self, eeg):
        return 1 / 2 * np.log2(2 * np.pi * np.e * np.std(eeg))

class BandPowerSpectralDensity:
    def __init__(self, sampling_rate = 128,
                 fft_n = None,
                 band_dict = {"theta": [4, 8],
                             "alpha": [8, 14],
                             "beta": [14, 31],
                             "gamma": [31, 49]}):

        self.sampling_rate = sampling_rate
        if fft_n is None:   fft_n = self.sampling_rate
        self.fft_n = fft_n
        self.band_dict = band_dict

    def apply(self, eeg):
        band_list = []

        hdata = eeg * hann(eeg.shape[1])
        fft_data = np.fft.fft(hdata, n=self.fft_n)
        energy_graph = np.abs(fft_data[:, 0 : int(self.fft_n / 2)])

        for _, band in enumerate(self.band_dict.values()):
            start_index = int(np.floor(band[0] / self.sampling_rate * self.fft_n))
            end_index = int(np.floor(band[1] / self.sampling_rate * self.fft_n))
            band_ave_psd = np.mean(energy_graph[:, start_index -1 : end_index]**2, axis=1)
            band_list.append(band_ave_psd)

        return np.stack(band_list, axis=-1)

def make_grid(datas, channel, location):
    CHANNEL_LOCATION_DICT = format_channel_location_dict(channel, location)
    togrid = ToGrid(CHANNEL_LOCATION_DICT)
    return np.array([togrid.apply(sample) for sample in datas])

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
        outputs = np.zeros([self.height, self.width, eeg.shape[-1]])
        for i, (loc_y, loc_x) in enumerate(self.channel_location_dict.values()):
            outputs[loc_y][loc_x] = eeg[i]

        outputs = outputs.transpose(2, 0, 1)
        return outputs

    def reverse(self, eeg):
        eeg = eeg.transpose(1, 2, 0)
        num_electrodes = len(self.channel_location_dict)
        outputs = np.zeros([num_electrodes, eeg.shape[2]])
        for i, (x, y) in enumerate(self.channel_location_dict.values()):
            outputs[i] = eeg[x][y]
        return outputs

def scaling(datas, scaler_name = None):
    if scaler_name == None: return datas
    flattend = datas.reshape(-1, 1)

    if scaler_name == 'standard':
        scaler = StandardScaler()
        scaled_datas = scaler.fit_transform(flattend)

    if scaler_name == 'robust':
        scaler = RobustScaler()
        scaled_datas = scaler.fit_transform(flattend)

    if scaler_name == 'log':
        scaled_datas = np.log1p(datas)

    if scaler_name == 'log_standard':
        scaler = StandardScaler()
        scaled_datas = scaler.fit_transform(np.log1p(flattend))

    scaled_datas = scaled_datas.reshape(datas.shape)
    return scaled_datas

def deshape(datas, shape_name = None, chls=None, location=None):
    if shape_name == None: return datas

    if shape_name == 'grid': 
        datas = make_grid(datas, chls, location)
        print(f'grid (samples, 4freq, 9x9): {datas.shape}')
    if shape_name == 'expand': 
        datas = np.expand_dims(datas, axis=1)
        print(f'expand (samples, 1, channels, window): {datas.shape}')
    return datas