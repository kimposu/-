pip install deep_sort_realtime

import subprocess
import argparse
import os
import sys
from pathlib import Path
import Arm_Lib
import PID
import cv2
import torch
import torch.backends.cudnn as cudnn
from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSORT 추가

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',
        source=ROOT / 'data/images',
        data=ROOT / 'data/coco128.yaml',
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False, 
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False, 
        dnn=False,
        ):
    
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)

    half &= (pt or jit or onnx or engine) and device.type != 'cpu'
    if pt or jit:
        model.model.half() if half else model.model.float()

    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs

    model.warmup(imgsz=(1, 3, *imgsz), half=half)
    dt, seen = [0.0, 0.0, 0.0], 0

    # DeepSORT 추적기 초기화
    deepsort = DeepSort()

    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        t2 = time_sync()
        dt[0] += t2 - t1

        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        Arm = Arm_Lib.Arm_Device()
        joints_0 = [90, 135, 20, 25, 90, 30]
        Arm.Arm_serial_servo_write6_array(joints_0, 1000)

        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # PID 제어기 초기화
            target_servox = 90
            target_servoy = 45
            xservo_pid = PID.PositionalPID(1.9, 0.3, 0.35)
            yservo_pid = PID.PositionalPID(1.9, 0.3, 0.35)

            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    center_point = (int((c1[0]+c2[0])/2), int((c1[1]+c2[1])/2))
                    # 객체 위치에 원 그리기
                    cv2.circle(im0, center_point, 5, (0, 255, 0), 2)
                    # 객체 위치에 텍스트 표시
                    cv2.putText(im0, str(center_point), center_point, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

                    # 추적기 업데이트
                    deepsort.update_tracks(det)

                    # DeepSORT 추적기에서 추적된 객체의 ID와 위치를 사용하여 로봇팔 제어
                    for track in deepsort.tracks:
                        if track.is_confirmed() and track.time_since_update <= 1:
                            track_id = track.track_id
                            track_center = track.to_tlbr()  # (x1, y1, x2, y2)
                            center_x = (track_center[0] + track_center[2]) / 2
                            center_y = (track_center[1] + track_center[3]) / 2
                            # PID로 로봇팔의 움직임 제어
                            xservo_pid.SystemOutput = round(center_x)
                            xservo_pid.SetStepSignal(320)
                            xservo_pid.SetInertiaTime(0.01, 0.1)
                            target_valuex = int(1500 + xservo_pid.SystemOutput)
                            target_servox = int((target_valuex - 500) / 10)

                            yservo_pid.SystemOutput = round(center_y)
                            yservo_pid.SetStepSignal(240)
                            yservo_pid.SetInertiaTime(0.01, 0.1)
                            target_valuey = int(1500 + yservo_pid.SystemOutput)
                            target_servoy = int((target_valuey - 500) / 10)

                            Arm.Arm_serial_servo_write2(target_servox, target_servoy, 1000)

            if save_txt: 
                with open(txt_path + '.txt', 'a') as f:
                    for *xyxy, conf, cls in reversed(det):
                        if save_conf:
                            f.write(('%g ' * 6 + '\n') % (*xyxy, conf, cls))
                        else:
                            f.write(('%g ' * 5 + '\n') % (*xyxy, cls))

            if save_img:
                cv2.imwrite(save_path, im0)

    LOGGER.info(f"Results saved to {save_dir}")
    return save_dir
