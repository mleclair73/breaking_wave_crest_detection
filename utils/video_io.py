import cv2
import numpy as np


def get_video_shape(path=None, cap=None):
    if cap is None:
        cap = cv2.VideoCapture(path)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    return N, W, H, FPS


def load_frames(file, stop=np.inf, dtype=np.int16, color=False):
    cap = cv2.VideoCapture(file)
    N, W, H, FPS = get_video_shape(cap=cap)

    frames = (
        np.zeros((min(N, stop), H, W), dtype=dtype)
        if not color
        else np.zeros((min(N, stop), H, W, 3), dtype=dtype)
    )
    print(f"{frames.size/1e9}gb allocated")
    for i in range(min(N, stop)):
        _, frame = cap.read()
        if not color:
            frames[i] = frame[..., 0]  # Saved as RGB but is grayscale
        else:
            frames[i] = np.flip(frame, 2)

    cap.release()
    cv2.destroyAllWindows()
    return frames
