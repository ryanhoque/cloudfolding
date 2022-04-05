"""
Utility script for human labeling of KP dataset
"""
import cv2
import numpy as np

from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

class SelectPoint:
    def __init__(self):
        self.dest = None
        self.image = None
        self.point_clicked = False
        self.data = []
        self.previous = None
    
    def reset(self):
        self.dest = None
        self.image = None
        self.point_clicked = False
        self.data = []
        self.previous = None
 
    def point_callback(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dest = (x, y)
            return
        elif event == cv2.EVENT_LBUTTONUP:
            if not self.point_clicked:
                self.image = self.previous.copy()
            cv2.drawMarker(self.image, self.dest, (23, 126, 42), thickness=3)
            self.point_clicked = False
    
    def collect_points(self,ipt):
        ipt = ipt[...,::-1].copy() # bgr to rgb
        self.image = ipt.copy()
        # history = [image.copy()]
        orig = self.image.copy()
        self.previous =  self.image.copy()
        window_name = "Click point, 'd' Collar, 's' Sleeves, 'a' Base, 'r' Redo, 'q' Quit, 't' reset all, 'f' skip:"
        cv2.destroyAllWindows()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.point_callback)
        while True:
            cv2.imshow(window_name, self.image)
            cv2.resizeWindow(window_name, (int(self.image.shape[1]*2.5), int(self.image.shape[0]*2.5)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                self.dest = None
                print("redo")
                self.point_clicked = False
                self.image = self.previous.copy()
            elif key == ord("t"):
                self.dest = None
                print("reset all")
                self.image = orig.copy()
                self.data = []
            elif key == ord("f"):
                self.point_clicked = True
                self.data.append(('bad', (-1,-1)))
                self.previous = self.image.copy()
                self.dest = None
                break
            elif key == ord("d") and self.dest is not None:
                self.point_clicked = True
                self.data.append(('c', self.dest))
                self.previous = self.image.copy()
                self.dest = None
            elif key == ord("s") and self.dest is not None:
                self.point_clicked = True
                self.data.append(('s', self.dest))
                self.dest = None
                self.previous = self.image.copy()
            elif key == ord("a") and self.dest is not None:
                self.point_clicked = True
                self.data.append(('b', self.dest))
                self.dest = None
                self.previous = self.image.copy()
            elif key == ord("q"):
                break
        total = {'collar':[], 'sleeves':[], 'base':[]}
        for val in self.data:
            if val[0] == 'c':
                total['collar'].append(val[1])
            elif val[0] == 's':
                total['sleeves'].append(val[1])
            elif val[0] == 'b':
                total['base'].append(val[1])
            elif val[0] == 'bad':
                total = {}
        cv2.destroyAllWindows()
        return total

if __name__ == "__main__":
    
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearingco
    collector = SelectPoint()
    while True:
        filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
        print(filename)
        t = collector.collect_points(cv2.imread(filename))
        print(t)
        collector.reset()