import cv2
import numpy as np
from libs.utils import *

if __name__ == "__main__":
    trt_model = edgeSR_TRT_Engine(
        engine_path="./model/baseline6.trt", lr_size=(224,320)
    )
    lrOrig = openImage("ms3_01.png")
    lrObj = np.transpose(lrOrig, [2, 0, 1])
    lrObj = np.ascontiguousarray(lrObj, dtype=np.float32)
    while True:
        cv2.imshow("trt_img", lrOrig)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()
    
    srObj = trt_model(lrObj).astype(np.uint8)
    srObj = np.transpose(srObj, [1,2,0])
    srObj = cv2.cvtColor(srObj, cv2.COLOR_RGB2BGR)

    
    while True:
        cv2.imshow("trt_img", srObj)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()