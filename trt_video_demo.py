import os
import argparse
import cv2
import numpy as np
from libs.utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video", type=str,
)
def preprocess(x:np.ndarray):
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = np.transpose(x, [2, 0, 1])
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x

def postprocess(x:np.ndarray):
    x = x.astype(np.uint8)
    x = np.transpose(x, [1, 2, 0])
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return x    

def bicubicResize(x:np.ndarray):
    x = cv2.resize(x, dsize=(640*3, 360*3), interpolation=cv2.INTER_LINEAR)
    return x

if __name__ == "__main__":
    opt = parser.parse_args()
    try:
        cap = cv2.VideoCapture(opt.video)
    except:
        raise ValueError(f"Failed to open video file")
    
    # load model
    trt_model = edgeSR_TRT_Engine(
        engine_path="./model/360x640_1080x1920.trt", lr_size=(360,640)
    )
    
    frameRate = 33

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bicubic = bicubicResize(frame)
        input_np = preprocess(frame)
        sr_np = postprocess(trt_model(input_np))
        
        key = cv2.waitKey(frameRate)
        if key == 27:
            break
        
        cv2.imshow("lr", frame)
        cv2.imshow("bicubic", bicubic)
        cv2.imshow("sr", sr_np)
        
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()