import cv2
import matplotlib.pyplot as plt
import numpy as np

from constants import gaussian_kernel
from preproc import get_spatial_filtered_images, get_temporal_filtered_video
from signals import get_pca_signal
from utils import load_video, select_center_point, select_segmenting_mask, write_video
from visual import draw_signals


def main():
    window_size = 4
    video, fps = load_video("../data/palm.mp4")

    n_frames, height, width, _ = video.shape
    features = np.load("../cache/features.npy")  # (n_frames, n_features, 2)
    n_features = features.shape[1]

    signals = []  # array of images rectangles for tracked features

    heatmaps = np.zeros((n_frames, height, width), dtype=np.float32)

    spatial_filtered_video = get_spatial_filtered_images(video, gaussian_kernel, 3)
    filtered_video = get_temporal_filtered_video(
        spatial_filtered_video, fps, (0.8, 1.5), alpha=2, attenuation=1
    )  # TODO: check alpha
    for i in range(n_features):
        for j in range(n_frames):
            x, y = features[j, i, :]
            x, y = int(x), int(y)
            x_start, x_end = max(0, x - window_size // 2), min(
                width, x + window_size // 2
            )
            y_start, y_end = max(0, y - window_size // 2), min(
                height, y + window_size // 2
            )
            feature_patch = filtered_video[
                :,
                y_start:y_end,
                x_start:x_end,
                :,
            ]
        signal = get_pca_signal(feature_patch)
        signals.append(signal)

    plt.figure(figsize=(10, 6))
    signals_fingertip_palm = [signals[0], signals[2]]
    draw_signals(signals_fingertip_palm, "signals_fingertip_palm_id_0_2.png")

    for i in [0, 2]:
        for j in range(n_frames):
            x, y = features[j, i]
            x, y = int(x), int(y)
            x_start, x_end = max(0, x - window_size // 2), min(
                width, x + window_size // 2
            )
            y_start, y_end = max(0, y - window_size // 2), min(
                height, y + window_size // 2
            )
            # print(heatmaps[j, y_start:y_end, x_start:x_end].shape)
            # print(signals.shape)
            heatmaps[j, y_start:y_end, x_start:x_end] = signal[j]

    heatmaps = heatmaps / heatmaps.max()
    heatmaps_normalized = (heatmaps * 255).astype(np.uint8)
    heatmap_frames = np.empty((n_frames, height, width, 3), dtype=np.uint8)
    for i in range(n_frames):
        heatmap_color = cv2.applyColorMap(heatmaps_normalized[i], cv2.COLORMAP_JET)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        heatmap_frames[i] = heatmap_color_rgb

    mask = (heatmaps != 0).astype(int)

    mask = mask[..., np.newaxis].repeat(3, axis=-1)  # add channel dim
    overlaid_video = np.where(mask, heatmap_frames, video)
    write_video(overlaid_video, fps, "palm_features.mp4")


if __name__ == "__main__":
    main()
