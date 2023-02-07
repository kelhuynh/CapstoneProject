from threading import Thread
import cv2

class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.success, self.image = self.cap.read()
        self.stopped = False
    
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        while not self.stopped:
            if not self.success:
                self.stop()
            else:
                self.success, self.image = self.cap.read()
                
    def stop(self):
        self.stopped = True