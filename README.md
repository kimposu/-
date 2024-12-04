import time
import Arm_Lib  # 로봇팔 라이브러리
import serial  # 시리얼 통신을 위한 라이브러리
import sys

def check_connection():
    """
    로봇팔과의 시리얼 연결 상태를 확인하는 함수.
    연결된 포트를 자동으로 찾고, 연결되었는지 확인.
    """
    try:
        # 시리얼 포트 목록을 확인하여 연결 가능한 포트 찾기
        ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0']  # 로봇팔에 연결된 시리얼 포트 예시
        for port in ports:
            try:
                # 해당 포트로 시리얼 연결 시도
                ser = serial.Serial(port, 9600, timeout=1)  # 예시로 9600 baud rate 사용
                time.sleep(2)  # 연결 대기 시간
                ser.write(b'HELLO')  # 로봇팔에 신호를 보내서 연결 확인
                response = ser.readline().decode('utf-8')
                if response.strip() == "READY":  # "READY" 메시지가 올 경우 연결 확인
                    print(f"로봇팔이 {port} 포트에 연결되었습니다.")
                    ser.close()  # 연결 종료
                    return ser  # 연결 객체 반환
            except serial.SerialException:
                continue
        print("로봇팔을 연결할 수 있는 포트를 찾지 못했습니다.")
        return None
    except Exception as e:
        print(f"연결 확인 중 오류가 발생했습니다: {e}")
        return None


def move_robot_arm(arm):
    """
    로봇팔을 지정된 위치로 이동시키는 함수.
    예시로 6개 관절을 설정하여 로봇팔을 움직이도록 함.
    """
    if arm is None:
        print("로봇팔 연결이 실패했습니다.")
        return

    try:
        # 로봇팔의 초기 관절 위치
        joints_0 = [90, 135, 20, 25, 90, 30]  # 관절 각도 (예시 값)
        arm.Arm_serial_servo_write6_array(joints_0, 1000)  # 1000ms에 걸쳐 로봇팔을 움직임

        print(f"로봇팔을 {joints_0} 위치로 이동 중...")
        time.sleep(2)  # 2초 대기 후 동작 완료 확인
        print("로봇팔 이동 완료.")

    except Exception as e:
        print(f"로봇팔을 움직이는 중 오류 발생: {e}")


if __name__ == "__main__":
    print("로봇팔 연결 확인 중...")
    
    # 1. 로봇팔 연결 상태 확인
    arm_connection = check_connection()

    if arm_connection:
        # 2. 연결되었으면 로봇팔 구동
        print("로봇팔이 정상적으로 연결되었습니다.")
        move_robot_arm(Arm_Lib.Arm_Device())  # Arm_Lib에서 Arm_Device 객체를 사용하여 구동
    else:
        print("로봇팔 연결 실패.")
