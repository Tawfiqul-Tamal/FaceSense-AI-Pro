import cv2


class CameraManager:
    def __init__(self, device_index=0):
        self.device_index = device_index
        self.capture = None

    def start(self):
        self.capture = cv2.VideoCapture(self.device_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return self.capture.isOpened()

    def read_frame(self):
        if self.capture is None:
            return False, None
        return self.capture.read()

    def stop(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
