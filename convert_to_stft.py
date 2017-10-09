import librosa
import pandas as pd
import numpy as np
import os
import scipy.misc

BASE_DATA_PATH = "/hdd/cs599/VCTK-Corpus/"
SPEAKER_INFO_PATH = BASE_DATA_PATH + "speaker-info.txt"
AUDIO_DATA_PATH = BASE_DATA_PATH + "wav48/"
N_FFT = 2048
CUTOFF_LEN = 125
SPECTRO_SAVE_PATH = "/hdd/cs599/spectro/"

def parse_speaker_info(speaker_info_path):
    header = None

    speaker_info = []
    with open(speaker_info_path) as f:
        for line in f:
            if header is None:
                header = line
            else:
                parts = line.split('  ')
                speaker_id = parts[0]
                gender = parts[2]
                speaker_info.append([speaker_id, gender])

    return speaker_info


speaker_info = parse_speaker_info(SPEAKER_INFO_PATH)

i = 0

for speaker_id, gender in speaker_info:
    print('%.2f%%' % ((float(i) / float(len(speaker_info))) * 100.))
    audio_dir = AUDIO_DATA_PATH + "p" + speaker_id + '/'
    for audio_file in os.listdir(audio_dir):
        x, fs = librosa.load(audio_dir + audio_file)
        S = librosa.stft(x, N_FFT)
        S = np.log1p(np.abs(S[:,:430]))

        if S.shape[1] < CUTOFF_LEN:
            continue

        cut_audio = S[:, :CUTOFF_LEN].T
        if gender == "M":
            folder = "male/"
        elif gender == "F":
            folder = "female/"
        else:
            raise ValueError("Could not determine save folder")

        audio_file_name = audio_file.split('.')[0]

        scipy.misc.imsave(SPECTRO_SAVE_PATH + folder + audio_file_name + ".jpg", cut_audio)
    i += 1

