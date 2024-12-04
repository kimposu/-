import serial
import time

# 직렬 포트 설정 (로봇 팔이 연결된 포트에 맞게 수정)
serial_port = '/dev/ttyUSB0'  # 실제 포트명을 확인하여 수정
baud_rate = 115200            # 로봇팔의 보드레이트에 맞게 설정
timeout = 1                    # 타임아웃 (1초)

# 직렬 통신 시작
ser = serial.Serial(serial_port, baud_rate, timeout=timeout)
print("로봇 팔과 연결됨")

# 로봇팔의 초기 위치로 이동하는 명령어 예시
def move_to_initial_position():
    # 예시: 로봇 팔 초기화 명령을 직렬로 보낸다고 가정
    ser.write(b'G28\n')  # G28 명령어: 초기화 (홈 위치로 이동)
    time.sleep(2)  # 2초 대기 (로봇이 초기화 되는 동안 기다림)

# 로봇 팔을 특정 위치로 이동시키는 함수
def move_to_position(x, y, z, rx, ry, rz):
    # 예시: x, y, z, rx, ry, rz 값을 포함한 G코드나 명령어 보내기
    # G1 명령어: 직선 이동 명령
    # x, y, z는 로봇팔의 목표 위치, rx, ry, rz는 목표의 회전 각도
    command = f'G1 X{x} Y{y} Z{z} RX{rx} RY{ry} RZ{rz}\n'
    ser.write(command.encode())  # 명령어를 직렬로 전송
    print(f"로봇 팔을 ({x}, {y}, {z}) 위치로 이동 중...")
    time.sleep(5)  # 로봇팔이 이동할 시간을 기다림

# 예시: 로봇팔 초기화 후 특정 위치로 이동
move_to_initial_position()
move_to_position(100, 200, 300, 0, 0, 0)  # 예시 좌표와 회전값

# 연결 종료
ser.close()
print("로봇 팔 연결 종료")
