#!/usr/bin/python
"""
Modified based on https://gist.github.com/jesseengel/e223622e255bd5b8c9130407397a0494

Note:
1. Librosa is required
2. Rainbowgram is a way for plotting magnitiude and phase CQT spectrogram,
   not a new type of feature
"""
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
import scipy.io.wavfile
import librosa

cdict  = {'red':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
          'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
          'blue':  ((0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),
          'alpha':  ((0.0, 1.0, 1.0),
                     (1.0, 0.0, 0.0))
}
my_mask = matplotlib.colors.LinearSegmentedColormap('MyMask', cdict)
plt.register_cmap(cmap=my_mask)

def transform(data, sampling_rate, frame_shift, 
              notes_per_octave, over_sample, 
              octaves, res_factor, base_note):
    ccqt = librosa.cqt(data, sr=sampling_rate, hop_length=frame_shift, 
                       bins_per_octave=int(notes_per_octave*over_sample), 
                       n_bins=int(octaves * notes_per_octave * over_sample),
                       filter_scale=res_factor, 
                       fmin=librosa.note_to_hz(base_note))
    mag, pha = librosa.magphase(ccqt)
    return mag, pha

def plot_rainbow(waveform_data, sampling_rate, 
                 fig, axis, 
                 frame_shift = 256, frame_length = 512, fft_points = 4096,
                 base_note = 'C1', over_sample = 4, octaves = 8, 
                 notes_per_octave = 12, warping_fac = 2.0,
                 res_factor = 0.8, magRange = None, phaRange = None):
    #https://gist.github.com/jesseengel/e223622e255bd5b8c9130407397a0494
    #  the range to plot 
    base = int(base_note[1])
    yticks_labels = ['C' + str(x+base) for x in np.arange(octaves)]
    yticks = [notes_per_octave * 0 * over_sample + x * over_sample * notes_per_octave for x in np.arange(len(yticks_labels)) ]
    
    if magRange is None or phaRange is None:
        magRange = [None, None]
        phaRange = [None, None]

        data = waveform_data
        sr = sampling_rate
        mag, pha = transform(data, sr, frame_shift, notes_per_octave, over_sample, octaves, res_factor, base_note)
        
        phaData = np.diff(np.unwrap(np.angle(pha))) / np.pi
        
        magData = (librosa.power_to_db(mag ** 2, amin=1e-13, top_db=70, ref=np.max) / 70) + 1.
        magData = magData[:, 0:(magData.shape[1]-1)]
        
    axis.matshow(phaData, cmap=plt.cm.rainbow, aspect='auto', origin='lower')
    axis.matshow(magData, cmap=my_mask, aspect='auto', origin='lower')
    
    axis.set_yticks(yticks)
    axis.set_yticklabels(yticks_labels)
    axis.set_ylabel('Octave')    
    return fig, axis


if __name__ == "__main__":
    print("plot_rainbogram")
    
