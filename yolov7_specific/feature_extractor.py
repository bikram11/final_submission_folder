import os
import argparse
import time
from pathlib import Path

import cv2
import torch


from models.experimental import attempt_load
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_boxes, xyxy2xywh, increment_path,set_logging,check_requirements, print_args

from utils.torch_utils import select_device

hook_activation = {}

# get activation function for specific layer code inspired from https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/
def get_activation(name):
    def hook(model, input, output):
        hook_activation[name] = output.detach()
    return hook

def run(source, weights, imgsz, name, project, exist_ok, save_txt, device, temp_feature_dir,augment):
    source = str(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    set_logging()
    device = select_device(device)


    model = attempt_load(weights)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    time_start = time.time()
    for path, im, im0s, vid_cap, s in dataset:
        count = 0
        im = torch.from_numpy(im).to(device)
        im = im.float()
        im /=255.0
        if len(im.shape) == 3:
            im = im[None]

        
        model.model[22].register_forward_hook(get_activation('before_bottleneck'))
        # pred= model(im, augment)[0]


    # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False

        feature_name = (path.split('/')[-1]).split('.')[0]
        activation_tensor = hook_activation['before_bottleneck'].data.cpu()

        if not os.path.exists(temp_feature_dir):
            os.makedirs(temp_feature_dir)
            
        torch.save(activation_tensor, os.path.join(temp_feature_dir, feature_name+'.pt'))

    print(f'Done. ({time.time() - time_start:.3f}s)')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='yolov5s model path')
    parser.add_argument('--source', type=str, default= 'data/images', help='scene video directory path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')

    parser.add_argument('--temp-feature-dir', metavar='DIR', help='path to folder where to save features')



    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
