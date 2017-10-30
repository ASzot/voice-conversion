import librosa
import pandas as pd
import numpy as np
import os
import scipy.misc
import util
import sys
import convert_to_voice

# Sri
#BASE_DATA_PATH = "/Users/sriramsomasundaram/Desktop/USC/Fall 2017/CSCI 599/DS_10283_2211/vcc2016_training/"
#SPECTRO_SAVE_PATH = "/Users/sriramsomasundaram/Desktop/USC/Fall 2017/CSCI 599/DS_10283_2211/vcc_processed/"

# CUTOFF_LEN = 258
CUTOFF_LEN = 256

# Andrew
SPECTRO_SAVE_PATH = '/hdd/cs599/spectro/'
BASE_DATA_PATH = '/hdd/cs599/VCTK-Corpus/wav48/'
SPEAKER_INFO_PATH = '/hdd/cs599/VCTK-Corpus/speaker-info.txt'

# Sample rate is 22050

def cut_audio(S):
    if S.shape[1] < CUTOFF_LEN:
        return None
    cut_audio = S[:, :CUTOFF_LEN, :]
    cut_audio = np.delete(cut_audio, -1, 0)
    padded = np.zeros((256, CUTOFF_LEN, 3))
    padded[:256, :, :2] = cut_audio
    return padded


if __name__ == '__main__':
    if len(sys.argv) > 1:
        folder_name = sys.argv[1]
        print('Loading for speaker %s' % folder_name)
        folders = [folder_name]
    else:
        folders = os.listdir(BASE_DATA_PATH)

    speaker_info = util.parse_speaker_info(SPEAKER_INFO_PATH)

    for folder_name in folders:
        folder_path = os.path.join(BASE_DATA_PATH, folder_name)
        for audio_file in os.listdir(folder_path):
            print(audio_file)
            x, fs = librosa.load(os.path.join(folder_path, audio_file))
            S = util.specgram(x)
            padded = cut_audio(S)
            if padded is None:
                continue

            gender = speaker_info[folder_name]

            if gender == "M":
                folder = "male/"
            elif gender == "F":
                folder = "female/"
            else:
                raise ValueError("Could not determine save folder")

            audio_file_name = audio_file.split('.')[0]
            audio_file_path = os.path.join(SPECTRO_SAVE_PATH, folder,
                audio_file_name + ".png")
            print('Saving to ' + audio_file_path)

            scipy.misc.imsave(audio_file_path, padded)

            convert_to_voice.from_file(audio_file_path)
            raise ValueError()

            #outfile = os.path.join(SPECTRO_SAVE_PATH, folder,
            #        audio_file_name + ".png")
            #scipy.misc.toimage(padded, cmin=0.0, cmax=255).save(outfile)

