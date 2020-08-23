import librosa
import numpy as np
import scipy


class Audio:
    def __init__(self, hparams):
        self.hparams = hparams
        self._mel_basis = self._build_mel_basis()

        self.min_level_db = np.array(hparams.min_level_db, np.float32) if hparams.stft_pad_center else np.array(
            hparams.min_level_db_no_center_padding, np.float32)

        self.max_level_db = np.array(hparams.max_level_db, dtype=np.float32) if hparams.stft_pad_center else np.array(
            hparams.max_level_db_no_center_padding, np.float32)

        self.average_mgc = np.array(hparams.average_mgc, dtype=np.float32)

        self.stddev_mgc = np.array(hparams.stddev_mgc, dtype=np.float32)

        self.average_mel_level_db = np.array(hparams.average_mel_level_db, dtype=np.float32)

        self.stddev_mel_level_db = np.array(hparams.stddev_mel_level_db, dtype=np.float32)

    def _build_mel_basis(self):
        n_fft = (self.hparams.num_freq - 1) * 2
        return librosa.filters.mel(self.hparams.sample_rate, n_fft, fmax = self.hparams.melmax, n_mels=self.hparams.num_mels)

    def load_wav(self, path):
        return librosa.core.load(path, sr=self.hparams.sample_rate)[0]

    def save_wav(self, wav, path):
        scipy.io.wavfile.write(path, self.hparams.sample_rate, wav)

    def trim(self, wav):
        unused_trimed, index = librosa.effects.trim(wav, top_db=self.hparams.trim_top_db,
                                                    frame_length=self.hparams.trim_frame_length,
                                                    hop_length=self.hparams.trim_hop_length)
        num_sil_samples = int(
            self.hparams.num_silent_frames * self.hparams.frame_shift_ms * self.hparams.sample_rate / 1000)
        start_idx = max(index[0] - num_sil_samples, 0)
        stop_idx = min(index[1] + num_sil_samples, len(wav))
        trimed = wav[start_idx:stop_idx]
        return trimed

    def silence_frames(self, wav, trim_frame_length, trim_hop_length):
        unused_trimed, index = librosa.effects.trim(wav, top_db=self.hparams.trim_top_db,
                                                    frame_length=self.hparams.trim_frame_length,
                                                    hop_length=self.hparams.trim_hop_length)
        num_start_frames = int((index[0] - trim_frame_length + trim_hop_length) / trim_hop_length)
        num_start_frames = max(num_start_frames - self.hparams.num_silent_frames, 0)
        num_stop_frames = int(((len(wav) - index[1]) - trim_frame_length + trim_hop_length) / trim_hop_length)
        num_stop_frames = min(num_stop_frames + self.hparams.num_silent_frames, len(wav))
        return num_start_frames, num_stop_frames

    def spectrogram(self, y):
        D = self._stft(y)
        S = self._amp_to_db(np.abs(D)) - self.hparams.ref_level_db
        return self._normalize_linear(S)

    def melspectrogram(self, y):
        D = self._stft(y)
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.hparams.ref_level_db
        return S

    def mel_and_lenear_spectrogram(self, y):
        D = self._stft(y)
        S_linear = self._amp_to_db(np.abs(D)) - self.hparams.ref_level_db
        S_mel = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.hparams.ref_level_db
        return S_mel.astype(np.float32), self._normalize_linear(S_linear).astype(np.float32)

    def _stft(self, y):
        n_fft, hop_length, win_length = self._stft_parameters()
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def _stft_parameters(self):
        n_fft = (self.hparams.num_freq - 1) * 2
        hop_length = int(self.hparams.frame_shift_ms / 1000 * self.hparams.sample_rate)
        win_length = int(self.hparams.frame_length_ms / 1000 * self.hparams.sample_rate)
        return n_fft, hop_length, win_length

    def _linear_to_mel(self, spectrogram):
        return np.dot(self._mel_basis, spectrogram)

    def _amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))

    def _normalize_linear(self, S):
        return (S - (self.hparams.min_level_db_linear - self.hparams.ref_level_db)) / (
                self.hparams.max_level_db_linear - self.hparams.min_level_db_linear)

    def normalize_mel(self, S):
        return (S - self.average_mel_level_db) / self.stddev_mel_level_db

    def normalize_mgc(self, S):
        return (S - self.average_mgc) / self.stddev_mgc
