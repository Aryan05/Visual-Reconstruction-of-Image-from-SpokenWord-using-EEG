import numpy as np
import mne
import re
import os
import matplotlib.pyplot as plt

MFF_DIR_PATH = '/home/91726/Desktop/data/'
CHANNELS = ['E9', 'E10', 'E20']
 

def get_eeg_data(mff_dir_path, channels):
    raw_files = set(filter(re.compile("[a-zA-Z0-9_-]*.mff").match, os.listdir(mff_dir_path)))
    print(raw_files)

    # audio(1) + viz(2) + noise(2)
    time_of_event = 2
    stride_split = 0.1
    time_of_split = 1
    num_ev_raw = np.zeros(4, dtype = np.int32)
    num_splits = (time_of_event-time_of_split)/stride_split+1 # like convolution dims

    psd_data_ = []
    freqs_ = []
    labels_ = []
    x=0

    for rf in raw_files:
        raw_egi = mne.io.read_raw_egi(mff_dir_path + rf, verbose=False)
        sfreq = raw_egi.info['sfreq']
        ev = mne.find_events(raw_egi,verbose=False)
        ev = ev[:-1]

        # print(ev[:20])

        # plot
        # mne.viz.plot_events(ev, raw_egi.info['sfreq'], raw_egi.first_samp)
        # '''
        tmin = 0
        tmax = time_of_split - 1/sfreq 
        s = 0

        for i in range(int(num_splits)):
            shifted_ev = mne.event.shift_time_events(ev,[1],i*stride_split,sfreq)
            ep = mne.Epochs(raw_egi,shifted_ev,tmin=tmin,tmax=tmax,
                picks=channels,baseline=(None,None),verbose=False)
            ep_raw = ep.load_data().filter(l_freq=0,h_freq=30)
            psd, freq = mne.time_frequency.psd_multitaper(ep_raw, fmax=30, verbose=False)

            psd_data_.append(psd)
            freqs_.append(np.expand_dims(freq,axis=1))
            labels_.append(shifted_ev[ep.selection, 2])
            s = s + len(ep.selection)
        num_ev_raw[x] = s
        x = x + 1
        break
        # '''
    print(num_splits)
    print(num_ev_raw.sum())
    print(num_ev_raw)
    print(len(psd_data_), len(labels_))

if __name__ == '__main__':
    get_eeg_data(MFF_DIR_PATH,CHANNELS)