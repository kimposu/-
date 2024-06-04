객체인식 프로그램
1. try
https://how2electronics.com/how-to-install-setup-opencv-on-raspberry-pi-4/

2. try - TensorFlow >> raspberry Pi 학습 >>>>>>>>>                    https://seo-dh-elec.tistory.com/32   >>>>>>  3번까지 

3. https://www.tensorflow.org/lite/guide/build_rpi?hl=ko

4. 
==============================================================
오픈 CV 설치

1. 라즈베리 파이 업데이트
먼저 라즈베리 파이의 패키지 리스트와 설치된 패키지를 최신 상태로 업데이트합니다.

bash
코드 복사
sudo apt-get update
sudo apt-get upgrade

2. 필수 패키지 설치
OpenCV를 빌드하고 사용하는 데 필요한 패키지들을 설치합니다.

bash
코드 복사
sudo apt-get install build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev


수정
sudo apt-get install build-essential cmake git pkg-config libgtk-3-dev \libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \gfortran openexr libatlas-base-dev python3-dev python3-numpy \libtbb-dev libdc1394-22-dev


libdc1394-22-dev패키지 없는경우
1. sudo apt-get install libdc1394-22 libdc1394-22-dev 입력
2. 빌드
wget https://github.com/ffainelli/libdc1394/archive/master.zip
unzip master.zip
cd libdc1394-master
mkdir build
cd build
cmake ..
make
sudo make install

    
    
3. OpenCV 소스 코드 다운로드
OpenCV와 OpenCV Contrib 모듈의 소스 코드를 다운로드합니다.

bash
코드 복사
cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git


4. OpenCV 빌드 디렉토리 생성
빌드를 위한 디렉토리를 생성하고 이동합니다.

bash
코드 복사
cd ~/opencv
mkdir build
cd build


5. CMake를 사용하여 OpenCV 설정
CMake를 사용하여 OpenCV 빌드 설정을 합니다. 이때 opencv_contrib 모듈의 경로를 지정합니다.

bash
코드 복사
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D ENABLE_NEON=ON \
    -D ENABLE_VFPV3=ON \
    -D BUILD_TESTS=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D BUILD_EXAMPLES=OFF ..

    
6. OpenCV 빌드 및 설치
빌드하고 설치합니다. 라즈베리 파이의 성능을 고려하여 make 명령어 뒤에 -j4 옵션을 추가하여 병렬 빌드를 수행합니다. -j4는 4개의 코어를 사용하여 빌드하는 것을 의미합니다.

bash
코드 복사
make -j4
sudo make install
sudo ldconfig


7. 설치 확인
OpenCV가 제대로 설치되었는지 확인합니다. Python 인터프리터를 실행하고 OpenCV를 임포트해봅니다.

bash
코드 복사
python3
python
코드 복사
import cv2
print(cv2.__version__)


=================== error code============
Traceback (most recent call last):
  File "/home/GM/TFLite_detection_webcam.py", line 102, in <module>
    from tensorflow.lite.python.interpreter import Interpreter
ModuleNotFoundError: No module named 'tensorflow'




Traceback (most recent call last):
  File "/home/GM/TFLite_detection_webcam.py", line 180, in <module>
    frame = frame1.copy()
            ^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'copy'

========================= 코드 수정
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Load the labels
with open('coco_labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize the interpreter
interpreter = Interpreter(model_path='detect.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input size
input_shape = input_details[0]['shape']
input_height = input_shape[1]
input_width = input_shape[2]

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from camera.")
        break

    # Prepare the frame
    img = cv2.resize(frame, (input_width, input_height))
    input_data = np.expand_dims(img, axis=0)
    input_data = input_data.astype(np.float32)

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Loop over all detections and draw bounding boxes
    for i in range(len(scores)):
        if scores[i] > 0.5:
            ymin, xmin, ymax, xmax = boxes[i]
            ymin = int(max(1, ymin * frame.shape[0]))
            xmin = int(max(1, xmin * frame.shape[1]))
            ymax = int(min(frame.shape[0], ymax * frame.shape[0]))
            xmax = int(min(frame.shape[1], xmax * frame.shape[1]))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Object Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
