import os
import cv2 as cv
from datetime import datetime



ROOT_DIR = "./records"
os.makedirs(ROOT_DIR, exist_ok=True)
video_name = datetime.now().strftime("%Y%m%d_%H%M%S_ver_1")


# Initialize cap object
if os.name == 'nt':
    cap = cv.VideoCapture(0)
else:
    cap = cv.VideoCapture("/dev/video2")

fourcc = cv.VideoWriter_fourcc(*'MJPG')

cap.set(cv.CAP_PROP_FOURCC, fourcc)

profile = {
    "4k": (3264, 2448),
    "2k": (2560, 1440),
    "fullhd": (1920, 1080),
    "high": (1280, 720),
    "medium": (640, 480)
}

chosen_profile = profile["fullhd"]

cap.set(cv.CAP_PROP_FRAME_WIDTH, chosen_profile[0])
cap.set(cv.CAP_PROP_FRAME_HEIGHT, chosen_profile[1])

out = cv.VideoWriter(os.path.join(ROOT_DIR, video_name + '.avi'), fourcc, 30.0, chosen_profile)


try:
    while True:
        if cap is None:
            raise RuntimeError("Capture object is not initiated.")
        
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # cv_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        cv_image = frame
        out.write(cv_image)

        cv.namedWindow('Webcam 4K', cv.WINDOW_AUTOSIZE)
        cv.imshow('Webcam 4K', cv_image)
        cv.waitKey(1)

finally:
    cap.release()
    cv.destroyAllWindows()

print("\nstream_video ended")
