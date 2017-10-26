import librosa
import pandas as pd
import numpy as np
import os
import scipy.misc
import util
import sys

BASE_DATA_PATH = "/Users/sriramsomasundaram/Desktop/USC/Fall 2017/CSCI 599/DS_10283_2211/vcc2016_training/"
# CUTOFF_LEN = 258
CUTOFF_LEN = 256
SPECTRO_SAVE_PATH = "/Users/sriramsomasundaram/Desktop/USC/Fall 2017/CSCI 599/DS_10283_2211/vcc_processed/"


if __name__ == '__main__':
    folder_name = sys.argv[1]
    print('Loading for speaker %s' % folder_name)
    folder_path = os.path.join(BASE_DATA_PATH, folder_name)
    for audio_file in os.listdir(folder_path):
        # print audio_file

        x, fs = librosa.load(os.path.join(folder_path,audio_file))
        S = util.specgram(x)
        if S.shape[1] < CUTOFF_LEN:
            continue
        cut_audio = S[:, :CUTOFF_LEN, :]
        # padded = np.zeros((258, CUTOFF_LEN, 3))
        # padded[:257, :, :2] = cut_audio
        cut_audio = np.delete(cut_audio, -1, 0)
        padded = np.zeros((256, CUTOFF_LEN, 3))
        padded[:256, :, :2] = cut_audio

        # Reconstructing wav file code
        # npad = ((0, 1), (0, 0), (0, 0))
        # padded_proc = np.pad(padded, pad_width=npad, mode='constant', constant_values=0)
        # audio = util.ispecgram(padded_proc[:257, :, :2])
        # librosa.output.write_wav('/Users/sriramsomasundaram/Desktop/USC/Fall 2017/CSCI 599/DS_10283_2211/test3.wav', x, fs)

        #raise ValueError()
        # if gender == "M":
        #     folder = "male/"
        # elif gender == "F":
        #     folder = "female/"
        # else:
        #     raise ValueError("Could not determine save folder")

        audio_file_name = audio_file.split('.')[0]

        scipy.misc.imsave(os.path.join(SPECTRO_SAVE_PATH, folder_name, audio_file_name + ".png"), padded)

        # outfile = os.path.join(SPECTRO_SAVE_PATH, folder_name, audio_file_name + ".png")
        # scipy.misc.toimage(padded, cmin=0.0, cmax=255).save(outfile)