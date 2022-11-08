import os
import torch
import cv2
import numpy as np
import torch_tensorrt

def openImage(filepath):
    try:
        imgObj = cv2.imread(filepath, cv2.IMREAD_COLOR)
        imgObj = cv2.cvtColor(imgObj, cv2.COLOR_BGR2RGB)
        return imgObj
    except:
        raise ValueError()

def npToTensor(x:np.ndarray):
    x = np.transpose(x, [2, 0, 1])
    tensor = torch.from_numpy(x)
    return tensor

def getTensorFromPath(path:str):
    return npToTensor(openImage(path)).unsqueeze(0).float().cuda()

def getTRT(path:str, size:list=[1, 3, 224, 320]):
    try:
        qat_model = torch.jit.load(path)
    except:
        raise ValueError()
    compile_spec = {
            "inputs" : [torch_tensorrt.Input(size)],
            "enabled_precisions" : torch.int8,
            "truncate_long_and_double" : True
        }
    return torch_tensorrt.compile(qat_model, **compile_spec)

class edgeSR_trt(object):
    def __init__(self,
                 qat_weights_path:str):
        
        self.trt = getTRT(qat_weights_path)
 
    def _cv2postprocess(self, x:np.ndarray):
        srObj = np.transpose(x, [1,2,0])
        srObj = cv2.cvtColor(srObj, cv2.COLOR_RGB2BGR)
        return srObj
 
    def inference(self, x:torch.Tensor):
        srTensor = self.trt(x).squeeze(0).cpu().numpy().astype(np.uint8)
        return self._cv2postprocess(srTensor)