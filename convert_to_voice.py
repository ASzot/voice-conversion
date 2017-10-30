import util
import numpy as np
import librosa
from scipy.ndimage import imread
import os

load_spectro = '/hdd/cs599/spectro/male/p272_046.png'

def from_file(load_spectro_filename):
    spectro = imread(load_spectro_filename)
    from_np(spectro)

def from_np(spectro):
    npad = ((0, 1), (0, 0), (0, 0))
    padded_proc = np.pad(spectro, pad_width=npad, mode='constant', constant_values=0)

    audio = util.ispecgram(padded_proc[:257, :, :2])

    fs = 22050
    #librosa.output.write_wav('/hdd/cs599/output/test.wav', audio, fs)


if __name__ == "__main__":
    from_file(load_spectro)

