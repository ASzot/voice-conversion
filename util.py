import librosa
import numpy as np

def specgram(audio,
             n_fft=512,
             hop_length=None,
             mask=True,
             log_mag=True,
             re_im=False,
             dphase=True,
             mag_only=False):
  """Spectrogram using librosa.
  Args:
    audio: 1-D array of float32 sound samples.
    n_fft: Size of the FFT.
    hop_length: Stride of FFT. Defaults to n_fft/2.
    mask: Mask the phase derivative by the magnitude.
    log_mag: Use the logamplitude.
    re_im: Output Real and Imag. instead of logMag and dPhase.
    dphase: Use derivative of phase instead of phase.
    mag_only: Don't return phase.
  Returns:
    specgram: [n_fft/2 + 1, audio.size / hop_length, 2]. The first channel is
      the logamplitude and the second channel is the derivative of phase.
  """
  if not hop_length:
    hop_length = int(n_fft / 2.)

  fft_config = dict(
      n_fft=n_fft, win_length=n_fft, hop_length=hop_length, center=True)

  spec = librosa.stft(audio, **fft_config)

  if re_im:
    re = spec.real[:, :, np.newaxis]
    im = spec.imag[:, :, np.newaxis]
    spec_real = np.concatenate((re, im), axis=2)

  else:
    mag, phase = librosa.core.magphase(spec)
    phase_angle = np.angle(phase)

    # Magnitudes, scaled 0-1
    if log_mag:
      mag = (librosa.logamplitude(
          mag**2, amin=1e-13, top_db=120., ref_power=np.max) / 120.) + 1
    else:
      mag /= mag.max()

    if dphase:
      #  Derivative of phase
      phase_unwrapped = np.unwrap(phase_angle)
      p = phase_unwrapped[:, 1:] - phase_unwrapped[:, :-1]
      p = np.concatenate([phase_unwrapped[:, 0:1], p], axis=1) / np.pi
    else:
      # Normal phase
      p = phase_angle / np.pi
    # Mask the phase
    if log_mag and mask:
      p = mag * p
    # Return Mag and Phase
    p = p.astype(np.float32)[:, :, np.newaxis]
    mag = mag.astype(np.float32)[:, :, np.newaxis]
    if mag_only:
      spec_real = mag[:, :, np.newaxis]
    else:
      spec_real = np.concatenate((mag, p), axis=2)
  return spec_real

def inv_magphase(mag, phase_angle):
    phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
    return mag * phase

def griffin_lim(mag, phase_angle, n_fft, hop, num_iters):
  """Iterative algorithm for phase retrival from a magnitude spectrogram.
  Args:
    mag: Magnitude spectrogram.
    phase_angle: Initial condition for phase.
    n_fft: Size of the FFT.
    hop: Stride of FFT. Defaults to n_fft/2.
    num_iters: Griffin-Lim iterations to perform.
  Returns:
    audio: 1-D array of float32 sound samples.
  """
  fft_config = dict(n_fft=n_fft, win_length=n_fft, hop_length=hop, center=True)
  ifft_config = dict(win_length=n_fft, hop_length=hop, center=True)
  complex_specgram = inv_magphase(mag, phase_angle)
  for i in range(num_iters):
    audio = librosa.istft(complex_specgram, **ifft_config)
    if i != num_iters - 1:
      complex_specgram = librosa.stft(audio, **fft_config)
      _, phase = librosa.magphase(complex_specgram)
      phase_angle = np.angle(phase)
      complex_specgram = inv_magphase(mag, phase_angle)
  return audio

def ispecgram(spec,
              n_fft=512,
              hop_length=None,
              mask=True,
              log_mag=True,
              re_im=False,
              dphase=True,
              mag_only=True,
              num_iters=1000):
  """Inverse Spectrogram using librosa.
  Args:
    spec: 3-D specgram array [freqs, time, (mag_db, dphase)].
    n_fft: Size of the FFT.
    hop_length: Stride of FFT. Defaults to n_fft/2.
    mask: Reverse the mask of the phase derivative by the magnitude.
    log_mag: Use the logamplitude.
    re_im: Output Real and Imag. instead of logMag and dPhase.
    dphase: Use derivative of phase instead of phase.
    mag_only: Specgram contains no phase.
    num_iters: Number of griffin-lim iterations for mag_only.
  Returns:
    audio: 1-D array of sound samples. Peak normalized to 1.
  """
  if not hop_length:
    hop_length = n_fft // 2

  ifft_config = dict(win_length=n_fft, hop_length=hop_length, center=True)

  if mag_only:
    mag = spec[:, :, 0]
    phase_angle = np.pi * np.random.rand(*mag.shape)
  elif re_im:
    spec_real = spec[:, :, 0] + 1.j * spec[:, :, 1]
  else:
    mag, p = spec[:, :, 0], spec[:, :, 1]
    if mask and log_mag:
      p /= (mag + 1e-13 * np.random.randn(*mag.shape))
    if dphase:
      # Roll up phase
      phase_angle = np.cumsum(p * np.pi, axis=1)
    else:
      phase_angle = p * np.pi

  # Magnitudes
  if log_mag:
    mag = (mag - 1.0) * 120.0
    mag = 10**(mag / 20.0)
  phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
  spec_real = mag * phase

  if mag_only:
    audio = griffin_lim(
        mag, phase_angle, n_fft, hop_length, num_iters=num_iters)
  else:
    audio = librosa.core.istft(spec_real, **ifft_config)
  return np.squeeze(audio / audio.max())


def parse_speaker_info(speaker_info_path):
    header = None

    speaker_info = {}
    with open(speaker_info_path) as f:
        for line in f:
            if header is None:
                header = line
            else:
                parts = line.split('  ')
                speaker_id = parts[0]
                gender = parts[2]
                speaker_info['p' + speaker_id] = gender

    return speaker_info


if __name__ == '__main__':
    x, fs = librosa.load("/hdd/cs599/VCTK-Corpus/wav48/p225/p225_039.wav")

    spectro = specgram(x)
    print(spectro.shape)

    audio = ispecgram(spectro)
    librosa.output.write_wav('/hdd/cs599/output/test.wav', x, fs)

