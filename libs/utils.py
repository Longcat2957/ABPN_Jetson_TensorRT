import os
import time
import torch
import cv2
import numpy as np
import tensorrt as trt

import pycuda.autoinit
import pycuda.driver as cuda

def openImage(filepath):
    try:
        imgObj = cv2.imread(filepath, cv2.IMREAD_COLOR)
        imgObj = cv2.cvtColor(imgObj, cv2.COLOR_BGR2RGB)
        return imgObj
    except:
        raise ValueError()


class edgeSR_TRT_Engine(object):
    def __init__(self, engine_path, scale:int=4, lr_size=(224, 320)):
        self.lr_size = lr_size
        self.scale = scale
        self.hr_size = (lr_size[0] * scale, lr_size[1] * scale)
        
        
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
                
    def __call__(self, lr:np.ndarray):
        self.inputs[0]['host'] = np.ravel(lr)
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        
        data = [out['host'] for out in self.outputs]
        data = data[0]
        sr = np.reshape(data, (3, self.hr_size[0], self.hr_size[1]))
        return sr
    
        