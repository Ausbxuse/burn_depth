import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, welch


def draw_signals(signals, output_file):
    plt.figure(figsize=(10, 5))
    for i, signal in enumerate(signals):
        plt.plot(signal, label=f"Signal {i+1}")
    # plt.plot(signals, label="Signal Intensity", color="green")
    plt.xlabel("Frame")
    plt.ylabel("Average Intensity")
    plt.title("Intensity Over Time")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file)
    plt.close()


def draw_fft(frequencies, fft, freq_range):

    amplitude = np.abs(fft)

    mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])

    plt.figure()
    plt.plot(frequencies[mask], amplitude[mask])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Amplitude vs. Frequency")
    plt.grid(True)
    plt.savefig("./out/freq.png")
    plt.close()


def draw_psd(signal, fs=30.0):
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]  # Convert to 2D with one row

    plt.figure(figsize=(10, 6))

    for idx, row in enumerate(signal):
        freqs, psd = welch(row, fs=fs, nperseg=256)

        plt.plot(freqs, psd, label=f"Signal {idx + 1}")

    plt.title("Power Spectral Density (PSD)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.grid()
    plt.legend()
    plt.savefig("./out/psd.png")


def draw_box(
    video, fps, center_point, window_size, s, boxed_video_path="./out/face.mp4"
):

    window_radius = window_size // 2
    n_frames, height, width, _ = video.shape

    boxed_video = cv2.VideoWriter(
        boxed_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    boxed_frames = []

    for i in range(n_frames):
        frame = video[i].astype(np.uint8)
        top_left = (center_point[1] - window_radius, center_point[0] - window_radius)
        bottom_right = (
            center_point[1] + window_radius,
            center_point[0] + window_radius,
        )
        boxed_frame = cv2.rectangle(
            frame.copy(), top_left, bottom_right, (0, 255, 0), 1
        )

        signal_height = 50  # height of the signal overlay area
        signal_width = frame.shape[1]
        signal_overlay = np.zeros((signal_height, signal_width, 3), dtype=np.uint8)

        normalized_signal = (
            (s - np.min(s)) / (np.max(s) - np.min(s)) * (signal_height - 1)
        )

        unit = max(width // 520, 1)
        # draw the signal line
        for j in range(1, len(normalized_signal)):
            if j < i:
                cv2.line(
                    signal_overlay,
                    (j - 1, signal_height - int(normalized_signal[j - 1])),
                    (j, signal_height - int(normalized_signal[j])),
                    (0, 255, 0),
                    unit,
                )

        # TODO:  box not visilbe
        boxed_frame[-signal_height:, :] = cv2.addWeighted(
            boxed_frame[-signal_height:, :], 0.5, signal_overlay, 0.5, 0
        )

        window_size = int(3 * fps)
        start_idx = max(0, i - window_size)
        window_signal = s[start_idx:i]

        peaks, _ = find_peaks(window_signal, distance=fps / 2)
        if len(peaks) > 1:
            peak_times = peaks / fps
            rr_intervals = np.diff(peak_times)
            avg_rr_interval = np.mean(rr_intervals)
            bpm = 60 / avg_rr_interval
        else:
            bpm = 0

        cv2.putText(
            boxed_frame,
            f"BPM: {int(bpm)}",
            (width // 2 - 10 * unit, 30 * unit),
            cv2.FONT_HERSHEY_SIMPLEX,
            unit,  # font scale
            (0, 255, 0),
            unit,  # thickness
            cv2.LINE_AA,
        )

        boxed_video.write(cv2.cvtColor(boxed_frame, cv2.COLOR_RGB2BGR))
        boxed_frames.append(boxed_frame)
    return np.array(boxed_frames)  # in RGB
