import cv2
import numpy as np


def get_skin_centroid(skin_mask):
    # Find contours in the skin mask
    contours, _ = cv2.findContours(
        skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        # Find the largest contour (by area)
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None


def get_skin_mask(frame):
    # Convert image to YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # Define skin color range in YCrCb
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    # Threshold the image to get only skin regions
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    # Optional: apply some morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)
    return skin_mask


cap = cv2.VideoCapture("../results/intensity/face.mp4")
ret, frame = cap.read()
if not ret:
    print("Cannot read video source.")
    exit()

# Define initial window using skin detection
skin_mask = get_skin_mask(frame)
centroid = get_skin_centroid(skin_mask)
if centroid is None:
    print("No skin region found!")
    exit()

# Define a window around the centroid
x, y = centroid
w, h = 50, 50  # Adjust as needed
track_window = (x - w // 2, y - h // 2, w, h)

# Set up the ROI for tracking
roi = frame[y - h // 2 : y + h // 2, x - w // 2 : x + w // 2]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask_roi = cv2.inRange(
    hsv_roi, np.array((0.0, 60.0, 32.0)), np.array((180.0, 255.0, 255.0))
)
roi_hist = cv2.calcHist([hsv_roi], [0], mask_roi, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Set up termination criteria: either 10 iterations or move by at least 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Apply CamShift to get the new location
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    cv2.imshow("CamShift Tracking", frame)
    if cv2.waitKey(30) & 0xFF in [27, ord("q")]:
        break

cap.release()
cv2.destroyAllWindows()
