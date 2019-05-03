import argparse
import cv2
import json

class ImageAnnotator:

    pastVersions = []
    newPoint = []
    boxes = []
    lines = []
    pastBoxes = []
    pastLines = []

    colors = [(0,255, 0), (0,255, 0), (255,0, 0), (255,0, 0), (0,0, 255), (0,0, 255)]
    
    color_index = 0

    def __init__(self, path, file, extension):
        self.path = path
        self.file = file
        self.extension = extension
        self.runImage(path + file + extension)

    def runImage(self, path):
        self.image = cv2.imread(path)
        self.clone = cv2.imread(path)

        self.pastVersions.append(self.image.copy())
        self.pastBoxes.append(self.boxes.copy())
        self.pastLines.append(self.lines.copy())

        cv2.namedWindow("image")

        #remember curr and nexxt functions for undo command
        self.curr = self.selectBox
        self.next = self.selectLine
        cv2.setMouseCallback("image", self.curr)
        

        while True:
            cv2.imshow("image", self.image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("u"):
                # if there are boxes added, remove the most recent one
                if len(self.pastVersions) > 1:

                    #set the image to the version without the box
                    self.image = self.pastVersions[-2].copy()
                    self.boxes = self.pastBoxes[-2].copy()
                    self.lines = self.pastLines[-2].copy()

                    #unstore the last version
                    self.pastVersions = self.pastVersions[:-1]
                    self.pastBoxes = self.pastBoxes[:-1]
                    self.pastLines = self.pastLines[:-1]
                    self.color_index = self.color_index - 1

                    #change next element (line/boxx)
                    temp = self.curr
                    self.curr = self.next
                    self.next = temp
                    cv2.setMouseCallback("image", self.curr)

                    cv2.imshow("image", self.image)
                print("undo")
            
            if key == ord("s"):
                filename = self.path + self.file + "_data" + ".txt"
                data_file = open(filename, "w")
                if (len(self.boxes) != len(self.lines)):
                    self.boxes = self.boxes[0:-1]
                hips = {"hips" : self.boxes}
                chests = {"chests" : self.lines}
                data_file.write(json.dumps(hips))
                data_file.write(json.dumps(chests))
                # cv2.imwrite("example.png",self.image)
                break


    def selectBox(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.newPoint = [(x, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            print("rec")
            self.newPoint.append((x, y))
            if (self.color_index < len(self.colors)):
                color = self.colors[self.color_index]
            else:
                color = (0,0,0)
            self.color_index += 1
            cv2.rectangle(self.image, self.newPoint[0], self.newPoint[1], color, 2)
            self.pastVersions.append(self.image.copy())
            self.boxes.append(self.newPoint)
            self.pastBoxes.append(self.boxes)
            self.pastLines.append(self.lines)
            self.newPoint = []
            # cv2.destroyWindow("image")
            cv2.imshow("image", self.image)
            self.curr = self.selectLine
            self.next = self.selectBox
            cv2.setMouseCallback("image", self.curr)

    def selectLine(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            print("line")
            self.newLine = x
            if (self.color_index < len(self.colors)):
                color = self.colors[self.color_index]
            else:
                color = (0,0,0)
            self.color_index += 1
            cv2.line(self.image, (x,0), (x,len(self.image)), color, 2)
            self.pastVersions.append(self.image.copy())
            self.lines.append(self.newLine)
            self.pastBoxes.append(self.boxes)
            self.pastLines.append(self.lines)
            cv2.imshow("image", self.image)
            self.curr = self.selectBox
            self.next = self.selectLine
            cv2.setMouseCallback("image", self.curr)


path = "../data/finish-line/testing/"
file = "003"
extension = ".bmp"
ia = ImageAnnotator(path, file, extension)