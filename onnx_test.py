import onnxruntime as ort
import numpy as np
import cv2
from libs.utils import *
lrOrig = openImage("ms3_01.png")
lrObj = np.transpose(lrOrig, [2, 0, 1])
lrObj = np.expand_dims(lrObj, axis=0).astype(np.float32)
ort_sess = ort.InferenceSession("./model/test2.onnx")
outputs = ort_sess.run(None, {'input': lrObj})[0].astype(np.uint8)
srObj = np.squeeze(outputs, axis=0)
srObj = np.transpose(srObj, [1, 2, 0])
srObj = cv2.cvtColor(srObj, cv2.COLOR_RGB2BGR)

    
while True:
    cv2.imshow("trt_img", srObj)
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()