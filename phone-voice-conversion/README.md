# Voice Conversion with Non-Parallel Data
## Sourced from Andabi's [repo](https://github.com/andabi/deep-voice-conversion)
### Changes and lessons learned
* Net1 can run stably on OSX with CPU's with the queue flag set to True.
* Net2 is more stable and can run on GPU's in \*nix distributions as well.
* Changes made in hparams and edit filepaths and model settings here.
* Changes made in number of CBHG's.
* Can run setup instructions for google cloud model in script a directory above.
## Model Architecture
This is a many-to-one voice conversion system.
The model architecture consists of two modules:
1. Net1(phoneme classification) classify someone's utterances to one of phoneme classes at every timestep.
    * Phonemes are speaker-independent while waveforms are speaker-dependent.
2. Net2(speech synthesis) synthesize speeches of the target speaker from the phones.

We applied CBHG(1-D convolution bank + highway network + bidirectional GRU) modules that are mentioned in [Tacotron](https://arxiv.org/abs/1703.10135).
CBHG is known to be good for capturing features from sequential data.

### Net1 is a classifier.
* Process: wav -> spectrogram -> mfccs -> phoneme dist.
* Net1 classifies spectrogram to phonemes that consists of 60 English phonemes at every timestep.
  * For each timestep, the input is log magnitude spectrogram and the target is phoneme dist.
* Objective function is cross entropy loss.
* [TIMIT dataset](https://catalog.ldc.upenn.edu/ldc93s1) used.
  * contains 630 speakers' utterances and corresponding phones that speaks similar sentences.
* Over 70% test accuracy

### Net2 is a synthesizer.
Net2 contains Net1 as a sub-network.
* Process: net1(wav -> spectrogram -> mfccs -> phoneme dist.) -> spectrogram -> wav
* Net2 synthesizes the target speaker's speeches.
  * The input/target is a set of target speaker's utterances.
* Since Net1 is already trained in previous step, the remaining part only should be trained in this step.
* Loss is reconstruction error between input and target. (L2 distance)
* Datasets
    * Target1(anonymous female): [Arctic](http://www.festvox.org/cmu_arctic/) dataset (public)
    * Target2(Kate Winslet): over 2 hours of audio book sentences read by her (private)
* Griffin-Lim reconstruction when reverting wav from spectrogram.

## Implementations
### Requirements
* python 2.7
* tensorflow >= 1.1
* numpy >= 1.11.1
* librosa == 0.5.1

### Settings
* sample rate: 16,000Hz
* window length: 25ms
* hop length: 5ms

### Procedure
* Train phase: Net1 and Net2 should be trained sequentially.
  * Train1(training Net1)
    * Run `train1.py` to train and `eval1.py` to test.
  * Train2(training Net2)
    * Run `train2.py` to train and `eval2.py` to test.
      * Train2 should be trained after Train1 is done!
* Convert phase: feed forward to Net2
    * Run `convert.py` to get result samples.
    * Check Tensorboard's audio tab to listen the samples.
    * Take a look at phoneme dist. visualization on Tensorboard's image tab.
      * x-axis represents phoneme classes and y-axis represents timesteps
      * the first class of x-axis means silence.

## Tips (Lessons We've learned from this project)
* Window length and hop length have to be small enough to be able to fit in only a phoneme.
* Obviously, sample rate, window length and hop length should be same in both Net1 and Net2.
* Before ISTFT(spectrogram to waveforms), emphasizing on the predicted spectrogram by applying power of 1.0~2.0 is helpful for removing noisy sound.
* It seems that to apply temperature to softmax in Net1 is not so meaningful.
* IMHO, the accuracy of Net1(phoneme classification) does not need to be so perfect.
  * Net2 can reach to near optimal when Net1 accuracy is correct to some extent.
