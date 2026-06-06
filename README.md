# Smart-Exhaust-Fan
This project, Smart Exhaust Fan with Environmental Sensing and Human Detection System, demonstrates the integration of sophisticated image recognition with automated actuator control to optimize indoor air quality and automation.

The system utilizes a Raspberry Pi 4 as the central processing unit, which processes visual data from a camera module to implement precise human detection, ensuring the fan only operates when the space is occupied. Beyond visual sensing, the system incorporates an MQ135 air quality sensor and a DHT22 temperature/humidity sensor to monitor environmental thresholds. When poor air quality or human presence is detected, the Raspberry Pi transmits a control signal to a Solid State Relay (SSR) acting as the primary electronic actuator to trigger an AC exhaust fan. This replaces traditional continuous-run ventilation with a "smart" mechanism that eliminates human intervention, while the Blynk Server provides a dedicated platform for real-time remote monitoring and data logging of all sensor activities.

# Technologies-Used

- Python
- OpenCV
- OpenCV DNN
- SSD MobileNet
- TensorFlow
- Numpy
- Raspberry Pi
- Rpi.GPIO
