import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import os
import threading
import serial
import BlynkLib
from BlynkTimer import BlynkTimer

# Global variables to store sensor values and status
corrected_ppm = None

# Blynk authentication token
BLYNK_AUTH = "jFI7kSwYbMAouLmhVccftkvxYBVl6ozK"
blynk = BlynkLib.Blynk(BLYNK_AUTH)
timer = BlynkTimer()

@blynk.on("connected")
def blynk_connected():
    print("You have connected to New Blynk 2.0")

# Setup GPIO
def setupGPIO(person_pin):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(person_pin, GPIO.OUT)
    GPIO.output(person_pin, GPIO.LOW)

# Read classes from the classes file
def readClasses(classesPath):
    with open(classesPath, 'r') as f:
        classesList = f.read().splitlines()
    colorList = np.random.uniform(low=0, high=255, size=(len(classesList), 3))
    return classesList, colorList

# Process the video to detect objects
def onVideo(videoPath, net, classesList, colorList, person_pin):
    global corrected_ppm
    
    cap = cv2.VideoCapture(videoPath)

    if not cap.isOpened():
        print("Error opening file...")
        return

    last_detection_time = None

    (success, image) = cap.read()
    while success:
        blynk.run()
        timer.run()
        classLabelIDs, confidences, bboxs = net.detect(image, confThreshold=0.4)
        bboxs = list(bboxs)
        confidences = list(np.array(confidences).reshape(1, -1)[0])
        confidences = list(map(float, confidences))
        bboxIDx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.4, nms_threshold=0.2)
        person_detected = False
        if len(bboxIDx) != 0:
            for i in range(len(bboxIDx)):
                bbox = bboxs[np.squeeze(bboxIDx[i])]
                classConfidence = confidences[np.squeeze(bboxIDx[i])]
                classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIDx[i])])
                classLabel = classesList[classLabelID]

                if classLabel == 'person':
                    person_detected = True
                    last_detection_time = time.time()
                    classColor = [int(c) for c in colorList[classLabelID]]
                    displayText = "{}:{:.2f}".format(classLabel, classConfidence)

                    x, y, w, h = bbox
                    cv2.rectangle(image, (x, y), (x+w, y+h), color=classColor, thickness=1)
                    cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

        current_time = time.time()
        
        if corrected_ppm is not None and corrected_ppm > 1000:
            last_detection_time = time.time()
        
        if person_detected or (last_detection_time and current_time - last_detection_time < 10) or (corrected_ppm is not None and corrected_ppm > 1000):
            GPIO.output(person_pin, GPIO.HIGH)
            blynk.virtual_write(2, 1)
        else:
            GPIO.output(person_pin, GPIO.LOW)
            blynk.virtual_write(2, 0)
        
        if person_detected and (last_detection_time and current_time - last_detection_time < 2):
            blynk.virtual_write(1, 1)
        else:
            blynk.virtual_write(1, 0)

        # Display corrected_ppm value
        if corrected_ppm is not None:
            ppmText = "Corrected PPM: {:.2f}".format(corrected_ppm)
            cv2.putText(image, ppmText, (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow("Result", image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        (success, image) = cap.read()

    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()

# Function to read data from the serial port
def readSerialData(port, baud_rate):
    global corrected_ppm
    ser = serial.Serial(port, baud_rate)
    while True:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').rstrip()
                corrected_ppm = int(line)
                blynk.virtual_write(0, corrected_ppm)
            except ValueError:
                pass

def main():
    videoPath = 0
    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    net = cv2.dnn.DetectionModel(modelPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    classesList, colorList = readClasses(classesPath)
    
    person_pin = 17
    setupGPIO(person_pin)
    
    # Start the serial reading thread
    serial_thread = threading.Thread(target=readSerialData, args=('/dev/ttyUSB0', 115200))
    serial_thread.daemon = True
    serial_thread.start()
    
    onVideo(videoPath, net, classesList, colorList, person_pin)

if __name__ == '__main__':
    main()
