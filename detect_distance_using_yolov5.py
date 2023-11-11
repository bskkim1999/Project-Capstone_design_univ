# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
import time   #새롭게 추가한 라이브러리
import math   #새롭게 추가한 라이브러리
import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()   #FILE은 현재 스크립트 파일의 절대 경로를 나타냄. path.resolve()의 역할 : 경로를 절대적으로 만듦.
ROOT = FILE.parents[0]  # YOLOv5 root directory  #path.parent()메소드는 이름에서 알 수 있듯이 문자열 형식의 인수로 전달 된 주어진 경로의 상위 디렉토리를 반환함.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH, ROOT 디렉토리의 경로를 문자열로 변환한 후, 이를 sys.path 리스트에 추가합니다.
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative, ROOT는 현재 작업 디렉토리에서 ROOT 디렉토리까지의 상대 경로를 나타냅니다.

#위의 코드는 주로 모듈 또는 패키지가 스크립트 파일과 동일한 디렉토리에 있지 않을 때, 이를 올바르게 참조하기 위해 필요한 경로 설정을 수행합니다.
#sys.path에 ROOT를 추가함으로써 ROOT 디렉토리에 있는 모듈 및 패키지를 불러올 수 있게 됩니다.


from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name    #검출된 결과를 저장하는 경로이다.
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    #print("source : ", source);
    #웹캠 2대를 동시에 접근하려고 할 때는 확장자가 streams인 파일에 0과 1을 작성하여 저장한다. source파일로 이용하면된다.
    source = str(source)   #만약 웹캠에 접근하려는 경우에는 source가 0이다. (1일 수도 있고 2일 수도 있을 것이다.(추측))
    save_img = not nosave and not source.endswith('.txt')  # save inference images   #nosave가 false이고 source가 텍스트파일형식이 아니면 save_img는 true이다.
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  #Path(source).suffix는 source 변수의 경로에서 파일 확장자(suffix)를 추출합니다.
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))  #startswith()의 의미 : 현재 문자열이 사용자가 지정하는 특정 문자로 시작하는지 확인하는 함수
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)   #웹캠에 접근하여 객체 검출을 수행하는 코드이다. #endswith()의 의미 : 현재 문자열이 사용자가 지정하는 특정 문자로 끝나는지 확인하는 함수
    screenshot = source.lower().startswith('screen')  #lower()의 의미 : 소문자 문자열로 변환한다.

    #print('save_img:', save_img)
    if is_url and is_file:
        source = check_file(source)  # download   #프로젝트를 수행하는데 굳이 해석할 필요가 없는 코드이다.

    # Directories  (실행결과를 저장한다.)
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir  #폴더를 만듦.

    # Load model
    device = select_device(device)  #사용 가능한 GPU가 있으면 GPU를 선택하고, 그렇지 않으면 CPU를 선택합니다.
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # YOLOv5 모델을 로드하고 설정하는 부분, DetectMultiBackend은 클래스이며, common.py에 존재함.
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size , 이미지 크기를 검증한다.  #[640, 640] 이 나온다.


    #위의 코드(Load model)는 YOLOv5 모델을 로드하고 실행할 디바이스를 설정하며, 모델의 속성과 이미지 크기를 초기화하여 객체 검출에 대한 환경을 설정합니다.

    # Dataloader
    bs = 1  # batch_size
    if webcam:   #이번 프로젝트에서 사용할 코드
        view_img = check_imshow(warn=True)      #Check if environment supports image displays
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  #객체를 사용하여 이미지를 가져오거나 비디오 스트림에서 프레임을 읽을 수 있습니다, 참고로 이 함수는 dataloaders.py에 있습니다.
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 위의 코드(Dataloader)는 데이터 로딩에 관련된 부분으로, 입력 이미지 또는 비디오를 처리하기 위한 데이터 로더를 설정합니다

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())  #이 코드는 객체 검출 작업에 필요한 초기 변수를 설정하는 역할을 합니다.
    count = 0  #물체 감지 횟수
    ############################################################인식을 시작한다!!#######################################################
    """
            path: 스트리밍 소스의 정보가 담긴 리스트입니다.
            im: 이미지 데이터를 담고 있는 NumPy 배열입니다. 이 이미지는 객체 감지에 사용됩니다.
            im0: 원본 이미지 데이터를 복사한 배열입니다.
            반환되는 다른 값들은 None이거나 빈 문자열입니다.
    """

    for path, im, im0s, vid_cap, s in dataset:   #변수 dataset이 위의 4개의 변수를 반환함.

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)   #이미지 데이터를 나타내는 넘파이 배열(numpy array)입니다. 이 코드는 해당 넘파이 배열을 PyTorch 텐서(Tensor)로 변환하고, 그 텐서를 모델이 사용하는 디바이스(device)로 이동시키는 역할을 합니다.
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        #print("check!")
        # Process predictions
        for i, det in enumerate(pred):  # per image  (각각의 이미지에 대하여), 아이템 i는 소스(웹캠)의 번호를 나타낸다.
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                #s += f'{i}: '   #0: 480x640 1 person, 135.5ms 라는 터미널 출력메시지 중에서 0: 에 해당한다.


            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg

            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt

            #s += '%gx%g ' % im.shape[2:]  # print string   #0: 480x640 1 person, 135.5ms 라는 터미널 출력메시지 중에서 480x640 에 해당한다.

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop, 변수 imc는 im0와 같다.
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))    #바운딩 박스 관련 코드이다.

            if len(det):
                # Rescale boxes from img_size to im0 size

                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    #웹캠을 2개 사용할 것이므로 실험을 위해 잠깐 코드를 추가하였다. (참고할 것!)

                    #if(n>=2 and names[int(c)] == 'person'):
                    #    print("^^")
                    #print(source)
                    #print(s)

                # Write results
                for *xyxy, conf, cls in reversed(det):    # *xyxy는 언패킹이다.
                    if save_txt:  # Write to file  (이 코드는 이번 프로젝트에서 사용안함.)
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    #(프로젝트에 사용하는 코드이다.)
                    if save_img or save_crop or view_img:  # Add bbox to image, 변수 save_img가 True 이다.
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')  #화면 상의 사각박스에 해당하는 클래스 이름과 정확도를 출력하는 코드임.
                        annotator.box_label(xyxy, label, color=colors(c, True)) #검출 화면에 바운딩박스를 그림.  plots.py 파일 안의 box_label함수가 있는데 xyxy는 변수 box에 해당함.
                        #바운딩 박스의 센터좌표를 추출하기 위하여 코드를 추가함.
                        #test_tmp1 = xyxy[0].item()
                        #print('test_tmp1 : ', test_tmp1)

                        center_x = (int(xyxy[0].item() + xyxy[2].item()) ) / 2
                        center_y = (int(xyxy[1].item() + xyxy[3].item()) ) / 2

                        #msg = '{} : {}, {}'.format(i, center_x, center_y)

                        #print(msg)

                        #물체의 깊이를 알기 위하여 추가한 코드이다.
                        cam_distance = 21.0

                        if i==0:
                            angle1 = (640 - center_x) * ( (11*math.pi/36) / 640 ) + (math.pi -(11*math.pi/36)) /2   #cam1

                        elif i==1:
                           angle2 = (center_x) * ( (11*math.pi/36) / 640 ) + (math.pi -(11*math.pi/36)) /2    #cam2

                    #(이번 프로젝트에서 필요없는 부분임.)#############################################
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
########################################################################################################################
            # Save results (image with detections), 변수 save_img가 Ture이다.
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        #물체의 깊이를 알기 위해 추가한 코드이다.
        angle3 = math.pi - angle1 - angle2
        object_depth = ( cam_distance * math.sin(angle1) * math.sin(angle2) ) / math.sin(angle3)


        print('object_depth : {} cm'.format(object_depth))
        #터미널 결과 출력 속도를 조정한다.
        """
        count = count + 1
        if count >= 5 :
            print('object_depth : {} cm'.format(object_depth))
            count = 0
        """


##########################################################################################################################
        # Print time (inference-only)
        #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")   #터미널에 출력되는 문장을 의미한다.
        #time.sleep(0.1)   #너무 빨라서 시간지연함.

    # Print results   #이번 프로젝트에서는 사용안함.
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')  #작은 크기의 입력 이미지는 모델을 빠르게 실행하지만 세부 정보를 놓칠 수 있습니다. 큰 크기의 입력 이미지는 세부 정보를 보다 정확하게 감지할 수 있지만 실행 시간이 증가할 수 있습니다.
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')  #바운딩박스의 선두께를 2로 지정함.
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')   #정확도(인식률)를 숨김.
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))  #객체 검출을 실행함.


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
