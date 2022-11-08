import os
import time
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

def _getTRT(path:str, size:list=[1, 3, 224, 320]):
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
                 qat_weights_path:str,
                 size:list=[1, 3, 224, 320]):
        
        self.trt = _getTRT(qat_weights_path, size)
 
    def _cv2postprocess(self, x:np.ndarray):
        srObj = np.transpose(x, [1,2,0])
        srObj = cv2.cvtColor(srObj, cv2.COLOR_RGB2BGR)
        return srObj
 
    def inference(self, x:torch.Tensor):
        srTensor = self.trt(x).squeeze(0).cpu().numpy().astype(np.uint8)
        return self._cv2postprocess(srTensor)

def benchmark(weights:str,input_shape=[1, 3, 360, 640], dtype="fp32", nwarmup=50, nruns=1000):
    model = edgeSR_trt(weights, input_shape)
    input_data = torch.randn(input_shape)
    input_data = input_data.to(torch.device("cuda"))
    if dtype == "fp16":
        input_data = input_data.half()
    
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model.inference(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            output = model.inference(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 100 == 0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))
    
    print("Input shape:", input_data.size())
    print("Output shape:", output.shape)
    print('Average batch time: %.2f ms'%(np.mean(timings)*1000))
