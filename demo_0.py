import cv2
import numpy as np
from libs.utils import *

if __name__ == "__main__":
    model = edgeSR_trt("edgeSR_max_qat_200.jit.pt")
    imgTensor = getTensorFromPath("ms.png")
    srCV2Obj = model.inference(imgTensor)
    
    while True:
        cv2.imshow(f"__name__", srCV2Obj)
        key = cv2.waitKey(1)
        if key == 27:
            break
        
    cv2.destroyAllWindows
    