import cv2
import numpy as np
import RPi.GPIO as GPIO
import time


class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn.DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

        # Setup GPIO
        self.setupGPIO()

        # Initialize the last detection time
        self.last_detection_time = None

    def setupGPIO(self):
        GPIO.setmode(GPIO.BCM)
        self.person_pin = 17
        GPIO.setup(self.person_pin, GPIO.OUT)
        GPIO.output(self.person_pin, GPIO.LOW)

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()
        self.colorList = np.random.uniform(
            low=0, high=255, size=(len(self.classesList), 3))

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if not cap.isOpened():
            print("Error opening file...")
            return

        (success, image) = cap.read()
        while success:
            classLabelIDs, confidences, bboxs = self.net.detect(
                image, confThreshold=0.4)
            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))
            bboxIDx = cv2.dnn.NMSBoxes(
                bboxs, confidences, score_threshold=0.4, nms_threshold=0.2)
            person_detected = False
            if len(bboxIDx) != 0:
                for i in range(len(bboxIDx)):
                    bbox = bboxs[np.squeeze(bboxIDx[i])]
                    classConfidence = confidences[np.squeeze(bboxIDx[i])]
                    classLabelID = np.squeeze(
                        classLabelIDs[np.squeeze(bboxIDx[i])])
                    classLabel = self.classesList[classLabelID]

                    if classLabel == 'person':
                        person_detected = True
                        self.last_detection_time = time.time()
                        classColor = [int(c)
                                      for c in self.colorList[classLabelID]]
                        displayText = "{}:{:.2f}".format(
                            classLabel, classConfidence)

                        x, y, w, h = bbox
                        cv2.rectangle(image, (x, y), (x+w, y+h),
                                      color=classColor, thickness=1)
                        cv2.putText(image, displayText, (x, y-10),
                                    cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

            current_time = time.time()
            if person_detected or (self.last_detection_time and current_time - self.last_detection_time < 30):
                GPIO.output(self.person_pin, GPIO.HIGH)
            else:
                GPIO.output(self.person_pin, GPIO.LOW)

            cv2.imshow("Result", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (success, image) = cap.read()
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()
