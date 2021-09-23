FROM python:3.6-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

ADD requirements.txt requirements.txt
ADD models models
ADD object_detection object_detection
ADD images images
ADD models models
ADD test test
ADD out out
ADD utils utils
ADD app.py app.py
ADD object_detector.py object_detector.py
ADD object_detector_detection_api.py object_detector_detection_api.py
ADD object_detector_detection_api_lite.py object_detector_detection_api_lite.py

# Install required libraries
RUN pip install -r requirements.txt

# Run it once to trigger resnet download
RUN python app.py

EXPOSE 8008

# Start the server
CMD ["python", "app.py", "serve"]