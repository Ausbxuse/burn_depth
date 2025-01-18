# @staticmethod
# def init_pool_processes_time_delay(s_list_, signal_ref_, fps_):
#     global s_list
#     global signal_ref
#     global fps
#     s_list = s_list_
#     signal_ref = signal_ref_
#     fps = fps_

# @staticmethod
# def _compute_time_delay(args):
#     i, j = args
#     s_patch = s_list[i, j, :]
#     f, Pxy = csd(s_patch, signal_ref, fs=fps, nperseg=256)
#     idx = np.argmax(np.abs(Pxy))
#     phase_spectrum = np.unwrap(np.angle(Pxy))
#     phase_diff = phase_spectrum[idx]
#     time_delay = phase_diff / (2 * np.pi * f[idx])
#     return (i, j, time_delay)

# def calc_time_delays(self, chunksize=100):

#     time_delays = np.zeros((self.n_patches_h, self.n_patches_w))

#     indices = [
#         (i, j)
#         for i in range(self.n_patches_h)
#         for j in range(self.n_patches_w)
#         if self.valid_mask[i, j] and self.patch_segmentation_mask[i, j]
#     ]

#     with Pool(
#         processes=cpu_count(),
#         initializer=self.init_pool_processes_time_delay,
#         initargs=(self.s_list, self.signal_ref, self.fps),
#     ) as pool:
#         results = pool.map(self._compute_time_delay, indices, chunksize)

#     for i, j, delay in results:  # TODO: verify how time_delay can be out of range
#         # if delay <= 0.3:
#         #     time_delays[i, j] = delay
#         # else:
#         #     self.valid_mask[i, j] = False
#         time_delays[i, j] = delay

#     self.time_delays = time_delays

# @staticmethod
# def init_pool_processes_time_delay(s_list_, fps_, heart_rate_freq_):
#     global s_list
#     global fps
#     global heart_rate_freq
#     s_list = s_list_
#     fps = fps_
#     heart_rate_freq = heart_rate_freq_

# @staticmethod
# def _compute_phase_angle(args):
#     i, j = args
#     s_patch = s_list[i, j, :]
#     N = len(s_patch)
#     freq_domain = np.fft.fft(s_patch)
#     freqs = np.fft.fftfreq(N, d=1 / fps)
#     idx = np.argmin(np.abs(freqs - heart_rate_freq))
#     phase_angle = np.angle(freq_domain[idx])
#     return (i, j, phase_angle)

# def calc_time_delays(self):
#     heart_rate_freq = self.heart_rate / 60  # in Hz
#     phase_angles = np.zeros((self.n_patches_h, self.n_patches_w))
#     indices = [
#         (i, j)
#         for i in range(self.n_patches_h)
#         for j in range(self.n_patches_w)
#         if self.valid_mask[i, j]
#     ]

#     with Pool(
#         processes=cpu_count(),
#         initializer=self.init_pool_processes_time_delay,
#         initargs=(self.s_list, self.fps, heart_rate_freq),
#     ) as pool:
#         results = pool.map(self._compute_phase_angle, indices)

#     for i, j, phase_angle in results:
#         phase_angles[i, j] = phase_angle

#     ref_i, ref_j = np.unravel_index(
#         np.nanargmax(phase_angles * self.valid_mask), phase_angles.shape
#     )
#     phi_ref = phase_angles[ref_i, ref_j]
#     delta_phi = phi_ref - phase_angles

#     delta_phi_unwrapped = unwrap_phase(delta_phi)

#     delta_t = delta_phi_unwrapped / (2 * np.pi * heart_rate_freq)
#     # self.time_delays = np.where(np.abs(delta_t) <= 0.3, delta_t, np.nan)
#     self.time_delays = delta_t


import numpy as np
from sklearn.decomposition import PCA


def get_chrom_signal(rgb_video):
    r = rgb_video[:, :, :, 0].astype(np.float32)
    g = rgb_video[:, :, :, 1].astype(np.float32)
    b = rgb_video[:, :, :, 2].astype(np.float32)

    norm = np.sqrt(r**2 + g**2 + b**2)

    r_n = (r / norm).mean(axis=(1, 2))
    g_n = (g / norm).mean(axis=(1, 2))
    b_n = (b / norm).mean(axis=(1, 2))

    x_s = 3 * r_n - 2 * g_n
    y_s = 1.5 * r_n + g_n - 1.5 * b_n

    alpha = np.std(x_s) / np.std(y_s)

    s = x_s - alpha * y_s
    # s = (s - np.mean(s)) / np.std(s)

    # F, T, Z = stft(s, 30)
    # # Z = np.squeeze(Z, axis=0)

    # band = np.argwhere((F > freq_range[0]) & (F < freq_range[1])).flatten()
    # spect = np.abs(Z[band, :])  # spectrum magnitude
    # freqs = 60 * F[band]  # spectrum freq in bpm

    # bpm = freqs[np.argmax(spect, axis=0)]
    # print("bpm is ", bpm)

    # draw_psd(s)
    # draw_box(center_point, window_radius, s)

    return s


def get_green_signal(rgb_video):
    r = rgb_video[:, :, :, 0].astype(np.float32)
    g = rgb_video[:, :, :, 1].astype(np.float32)
    b = rgb_video[:, :, :, 2].astype(np.float32)
    norm = np.sqrt(r**2 + g**2 + b**2)
    g_n = (g / norm).mean(axis=(1, 2))
    s = g_n

    return s

def get_pca_signal(rgb_video):
    reshaped_video = rgb_video.reshape(rgb_video.shape[0], -1).astype(np.float32)

    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(reshaped_video)

    s = principal_component.flatten()

    return s
