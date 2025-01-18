import cv2
import numpy as np
import tqdm

def pyrDown(image, kernel):
    return cv2.filter2D(image, -1, kernel)[::2, ::2]


def pyrUp(image, kernel, dst_shape=None):
    dst_height = image.shape[0] + 1
    dst_width = image.shape[1] + 1

    if dst_shape is not None:
        dst_height -= dst_shape[0] % image.shape[0] != 0
        dst_width -= dst_shape[1] % image.shape[1] != 0

    height_indexes = np.arange(1, dst_height)
    width_indexes = np.arange(1, dst_width)

    upsampled_image = np.insert(image, height_indexes, 0, axis=0)
    upsampled_image = np.insert(upsampled_image, width_indexes, 0, axis=1)

    return cv2.filter2D(upsampled_image, -1, 4 * kernel)


def temporal_bp_filter(images, fps, freq_range, axis=0):

    fft = np.fft.fft(images, axis=axis)
    frequencies = np.fft.fftfreq(images.shape[0], d=1.0 / fps)

    low = (np.abs(frequencies - freq_range[0])).argmin()
    high = (np.abs(frequencies - freq_range[1])).argmin()

    fft[:low] = 0
    fft[high:] = 0

    return np.fft.ifft(fft, axis=0).real

def spatial_filter(image, kernel, level):
    """
    downsample + applies gaussian filter + upsample
    """

    image_shape = [image.shape[:2]]
    downsampled_image = image.copy()

    for _ in range(level):
        downsampled_image = pyrDown(image=downsampled_image, kernel=kernel)
        image_shape.append(downsampled_image.shape[:2])

    gaussian_pyramid = downsampled_image
    for curr_level in range(level):
        gaussian_pyramid = pyrUp(
            image=gaussian_pyramid,
            kernel=kernel,
            dst_shape=image_shape[level - curr_level - 1],
        )

    return gaussian_pyramid


def get_spatial_filtered_images(images, kernel, level):
    filtered_images = np.zeros_like(images, dtype=np.float32)
    # gaussian_pyramids = np.zeros(
    #     (images.shape[0], images.shape[1], images.shape[2]), dtype=np.float32
    # )

    for i in tqdm.tqdm(
        range(images.shape[0]), ascii=True, desc="Applying gaussian blur to spatially filter video"
    ):
        # green_channel = images[i, :, :, 1]
        filtered_images[i] = spatial_filter(
            image=images[i], kernel=kernel, level=level
        )

    return filtered_images


def get_temporal_filtered_video(video, fps, freq_range, alpha, attenuation):
    filtered_images = temporal_bp_filter(
        images=video, fps=fps, freq_range=freq_range
    ).astype(np.float32)

    filtered_images *= alpha
    # filtered_pyramids[:, :, :, 1:] *= attenuation

    return filtered_images
