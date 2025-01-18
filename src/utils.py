import cv2
import numpy as np

def load_video(video_path):
    image_sequence = []
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    while video.isOpened():
        ret, frame = video.read()

        if ret is False:
            break

        image_sequence.append(frame[:, :, ::-1])

    video.release()
    return np.asarray(image_sequence), fps

def write_video(video, fps, output_name):
    """
    takes an rgb video and write to file
    """
    _, height, width, _ = video.shape
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_writer = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
    for frame in video:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    video_writer.release()
    print(f"Heatmap video saved as {output_name}")


def select_center_point(image):
    selected_point = [None]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_point[0] = (y, x)
            print(f"Selected point: {selected_point[0]}")
            cv2.circle(param, (y, x), 5, (0, 255, 0), -1)
            cv2.imshow("Select Center Point", param)

    temp_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    cv2.imshow("Select Center Point", temp_image)
    cv2.setMouseCallback("Select Center Point", mouse_callback, temp_image)

    while selected_point[0] is None:
        if cv2.waitKey(1) & 0xFF == 27:  # Exit if Esc is pressed
            print("Selection cancelled.")
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return selected_point[0]


def select_segmenting_mask(image, mask_path):
    # Create a copy of the image for display and mask creation
    display_image = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Single channel mask

    # Define drawing parameters
    drawing = False  # True if the mouse is pressed
    brush_size = 80
    brush_color = 255  # White color for the mask

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, display_image, mask

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            cv2.circle(
                display_image, (x, y), brush_size, (0, 255, 0), -1
            )  # Draw on display
            cv2.circle(mask, (x, y), brush_size, brush_color, -1)  # Draw on mask

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(display_image, (x, y), brush_size, (0, 255, 0), -1)
                cv2.circle(mask, (x, y), brush_size, brush_color, -1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.circle(display_image, (x, y), brush_size, (0, 255, 0), -1)
            cv2.circle(mask, (x, y), brush_size, brush_color, -1)

    # Create a window and set the mouse callback
    cv2.namedWindow("Draw Mask", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Draw Mask", mouse_callback)

    print("Instructions:")
    print(" - Draw on the image to create the mask.")
    print(" - Press 's' to save the mask and exit.")
    print(" - Press 'c' to clear the mask and start over.")
    print(" - Press 'Esc' to cancel.")

    display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
    while True:
        cv2.imshow("Draw Mask", display_image)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # Esc key to cancel
            print("Mask selection canceled.")
            cv2.destroyAllWindows()
            return None
        elif key == ord("s"):  # Save mask
            print("Mask saved.")
            cv2.destroyAllWindows()
            np.save(mask_path, mask)
            return mask > 0
        elif key == ord("c"):  # Clear mask
            print("Mask cleared. Start drawing again.")
            display_image = image.copy()
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

    cv2.destroyAllWindows()
