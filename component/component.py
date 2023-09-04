import logging

import numpy as np
from importlib_resources import files
from time import perf_counter

import cv2
import torch

from sahi_general import SahiGeneral

from base_component import BaseComponent
from yolov7.yolov7 import YOLOv7

logging.basicConfig(level=logging.INFO)


class Component(BaseComponent):

    def __init__(self, config):
        super().__init__(config)

        self.yolov7 = YOLOv7(
            weights=files('yolov7').joinpath('weights/yolov7-w6_state.pt'),
            cfg=files('yolov7').joinpath('cfg/deploy/yolov7-w6.yaml'),
            bgr=True,
            device='cuda',
            model_image_size=640,
            max_batch_size=16,
            half=True,
            same_size=True,
            conf_thresh=0.25,
            trace=False,
            cudnn_benchmark=False,
        )
        self.sahi_general = SahiGeneral(model=self.yolov7)
        self.classes = ['person', 'car']

    def process(self, image):
        torch.cuda.synchronize()
        tic = perf_counter()
        detections = self.sahi_general.detect([image], self.classes)
        torch.cuda.synchronize()
        dur = perf_counter() - tic

        logging.info(f'Time taken: {(dur*1000):0.2f}ms')

        draw_frame = np.copy(image)

        for det in detections[0]:
            l = det['l']
            t = det['t']
            r = det['r']
            b = det['b']
            classname = det['label']
            cv2.rectangle(draw_frame, (l, t), (r, b), (255, 255, 0), 1)
            cv2.putText(draw_frame, classname, (l, t-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))

        return draw_frame
