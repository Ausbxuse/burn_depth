import cv2
import numpy as np


def main():
    # Open video capture (0 for webcam, or provide a filename)
    cap = cv2.VideoCapture(
        "../results/intensity/face.mp4"
    )  # Change to filename if needed, e.g., "video.mp4"

    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Unable to read video source.")
        return

    # Convert the first frame to grayscale
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Global variable to store the selected point
    initial_point = None

    # Mouse callback to select the point
    def select_point(event, x, y, flags, param):
        nonlocal initial_point
        if event == cv2.EVENT_LBUTTONDOWN:
            initial_point = (x, y)
            print(f"Selected point: {initial_point}")

    # Create a window and set the mouse callback for point selection
    cv2.namedWindow("Select Point (Click on the image)")
    cv2.setMouseCallback("Select Point (Click on the image)", select_point)

    # Show the first frame until a point is selected
    while initial_point is None:
        cv2.imshow("Select Point (Click on the image)", first_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Select Point (Click on the image)")

    # Create an image for drawing the trajectory
    traj = np.zeros_like(first_frame)

    # Set the initial point to track
    point = initial_point

    # Set the previous frame (in grayscale) to the first frame
    prev_gray = first_gray.copy()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read the frame.")
            break

        # Convert current frame to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute the dense optical flow using Farneback's algorithm
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=5,
            winsize=15,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )

        # Get current point coordinates
        x, y = point
        h, w = flow.shape[:2]
        # Ensure the point is inside the frame boundaries
        if x < 0 or x >= w or y < 0 or y >= h:
            print("The tracked point went out of frame.")
            break

        # Obtain the flow vector at the point (rounding to nearest integer indices)
        flow_at_point = flow[int(y), int(x)]
        dx, dy = flow_at_point

        # Update the point position
        new_point = (int(x + dx), int(y + dy))

        # Draw the trajectory: draw a line from the previous to the new point
        cv2.line(traj, point, new_point, (0, 255, 0), thickness=2)
        # Draw a circle at the new point position on the current frame
        cv2.circle(frame, new_point, radius=3, color=(0, 0, 255), thickness=-1)

        # Overlay the trajectory on the current frame
        output = cv2.add(frame, traj)

        # Display the frame with the trajectory
        cv2.imshow("Trajectory", output)
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord("q"):
            break

        # Update the point and previous frame for the next iteration
        point = new_point
        prev_gray = curr_gray.copy()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
