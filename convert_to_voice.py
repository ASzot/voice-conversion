import util
import numpy as np
import librosa
from scipy.ndimage import imread
import os


def from_file(load_spectro_filename, add_name):
    print('Loading from')
    print(load_spectro_filename)
    spectro = imread(load_spectro_filename)
    print(np.mean(spectro))
    print(np.std(spectro))

    old_range = 255
    new_range = 2

    spectro = spectro.astype(np.float32)

    spectro  = ((spectro * 2.) / 255.) - 1.0
    base_name = os.path.basename(load_spectro_filename)
    filename = base_name.split('.')[0]

    from_np(spectro, filename, add_name)

def from_np(spectro, filename, add_name):
    padded = np.zeros((257, 256, 2))
    padded[:256, :, :] = spectro[:256, :256, :2]
    #padded = spectro

    audio = util.ispecgram(padded)

    fs = 22050
    #save_filename = '/hdd/cs599/output/%s.wav' % filename
    save_filename = '/home/ubuntu/data/results/output/%s.wav' % (filename +
            add_name)
    print('Saving to')
    print(save_filename)
    librosa.output.write_wav(save_filename, audio, fs)


if __name__ == "__main__":
    # filename = sys.argv[1]
    base_path = '/home/ubuntu/data/'
    #load_spectro = '/hdd/cs599/spectro/male/p272_046.png'
    #load_spectro = '/hdd/cs599/spectro/testA/p272_003.png'

    #load_spectro = base_path + 'female/p253_003.png'
    #load_spectro = '/hdd/cs599/spectro/testB/p253_003.png'
    testA = os.path.join(base_path, 'spectro/testA')
    for filename in os.listdir(testA):
        load_spectro = os.path.join(testA, filename)
        from_file(load_spectro, '_orig')
        load_spectro = os.path.join(base_path, 'results/resultA/', filename)
        from_file(load_spectro, '_transformed')

    load_spectro = base_path + 'vcc_processed/TF1/100005.png'
    from_file(load_spectro, '_orig')

    load_spectro = base_path + 'results/resultTF1/100005.png'
    from_file(load_spectro, '_transformed')

