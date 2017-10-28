import librosa
import pandas as pd
import numpy as np
import os
import scipy.misc
import util

BASE_DATA_PATH = "/hdd/cs599/VCTK-Corpus/"
SPEAKER_INFO_PATH = BASE_DATA_PATH + "speaker-info.txt"
AUDIO_DATA_PATH = BASE_DATA_PATH + "wav48/"
CUTOFF_LEN = 258
SPECTRO_SAVE_PATH = "/hdd/cs599/spectro/"


speaker_info = parse_speaker_info(SPEAKER_INFO_PATH)

i = 0

for speaker_id, gender in speaker_info:
    print('%.2f%%' % ((float(i) / float(len(speaker_info))) * 100.))
    audio_dir = AUDIO_DATA_PATH + "p" + speaker_id + '/'
    print('Loading for speaker %s' % speaker_id)
    for audio_file in os.listdir(audio_dir):
        x, fs = librosa.load(audio_dir + audio_file)
        S = util.specgram(x)

        if S.shape[1] < CUTOFF_LEN:
            continue

        cut_audio = S[:, :CUTOFF_LEN, :]

        padded = np.zeros((258, CUTOFF_LEN, 3))
        padded[:257, :, :2] = cut_audio

        #audio = util.ispecgram(padded[:257, :, :2])
        #librosa.output.write_wav('/hdd/cs599/output/test3.wav', x, fs)
        #raise ValueError()

        if gender == "M":
            folder = "male/"
        elif gender == "F":
            folder = "female/"
        else:
            raise ValueError("Could not determine save folder")

        audio_file_name = audio_file.split('.')[0]

        scipy.misc.imsave(SPECTRO_SAVE_PATH + folder + audio_file_name +
                ".jpg", padded)
    i += 1

