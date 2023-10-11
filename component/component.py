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
        self.classes = ['person', 'boat', 'traffic light', 'surfboard']

    def process(self, image):
        torch.cuda.synchronize()
        tic = perf_counter()
        detections = self.sahi_general.detect([image], self.classes)
        torch.cuda.synchronize()
        dur = perf_counter() - tic

        logging.info(f'Time taken: {(dur*1000):0.2f}ms')
        logging.info(f'Obtained detections: {detections}')

        dets = [[det['l'], det['t'], det['w'], det['h'], det['confidence'], self.classes.index(det['label'])] for det in detections[0]]
        dets = np.array(dets, dtype='f4')
        logging.info(f'Returning detections in the following format: {dets}')
        return dets
