"""
An example that uses TensorRT's Python api to make inferences.
"""
import os
import shutil
import sys
import threading
import time
from turtle import done
import cv2
import numpy as np
import torch
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

import argparse
from multiprocessing import Process
import subprocess
import threading
from dataloaders import LoadImages,LoadWebcam,LoadStreams

def get_img_path_batches(batch_size, img_dir):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret


with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]


class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        for binding in engine:
            print('binding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(
                binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def infer(self, raw_image_generator):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        # Do image preprocess
        batch_image_raw = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])

        '''
        for i, image_raw in enumerate(raw_image_generator):
            batch_image_raw.append(image_raw)
            input_image = self.preprocess_cls_image(image_raw)
            np.copyto(batch_input_image[i], input_image)
        '''

        input_image = self.preprocess_cls_image(raw_image_generator)
        np.copyto(batch_input_image, input_image)

        batch_input_image = np.ascontiguousarray(batch_input_image)
        raw_image = np.ascontiguousarray(input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size,
                              bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        for i in range(self.batch_size):
            classes_ls, predicted_conf_ls, category_id_ls = self.postprocess_cls(output)

        return str(classes_ls)

        '''
            cv2.putText(raw_image, str(classes_ls), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
            print(classes_ls, predicted_conf_ls)
            cv2.imshow('yolov5-cls', raw_image)  # 显示读取到的这一帧画面
            cv2.waitKey(25)

        return batch_image_raw, end - start
        '''

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)

    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_cls_image(self, input_img):
        im = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (self.input_h, self.input_w))
        im = np.float32(im)
        im /= 255.0
        im -= self.mean
        im /= self.std
        im = im.transpose(2, 0, 1)
        # prepare batch
        batch_data = np.expand_dims(im, axis=0)

        return batch_data

    def postprocess_cls(self, output_data):
        classes_ls = []
        predicted_conf_ls = []
        category_id_ls = []
        output_data = output_data.reshape(self.batch_size, -1)
        output_data = torch.Tensor(output_data)
        p = torch.nn.functional.softmax(output_data, dim=1)
        score, index = torch.topk(p, 3)
        for ind in range(index.shape[0]):
            input_category_id = index[ind][0].item()  # 716
            category_id_ls.append(input_category_id)
            predicted_confidence = score[ind][0].item()
            predicted_conf_ls.append(predicted_confidence)
            classes_ls.append(classes[input_category_id])
        return classes_ls, predicted_conf_ls, category_id_ls

class inferThread_R(threading.Thread):
    def __init__(self, yolov5_wrapper, raw_image):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper
        self.raw_image = raw_image

    def run(self):
        cls_type = self.yolov5_wrapper.infer(self.raw_image)

        # 标注释
        cv2.putText(self.raw_image, cls_type, (10, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # 显示读取到的这一帧画面
        cv2.imshow('yolov5-cls', self.raw_image)
        cv2.waitKey(30)
        

class warmUpThread(threading.Thread):
    def __init__(self, yolov5_wrapper):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper

    def run(self):
        print("warmUpThread done")
'''
        batch_image_raw, use_time = self.yolov5_wrapper.infer(self.yolov5_wrapper.get_raw_image_zeros())
        print('warm_up->{}, time->{:.2f}ms'.format(batch_image_raw[0].shape, use_time * 1000))
'''

def parse_args():
    parser = argparse.ArgumentParser(description='choose .engine and video stream source to inference')
    parser.add_argument('-e', '--engine', required=True      , help='Input engine (.engine) file path (required)')
    parser.add_argument('-s', '--source', help='choose video stream source,default: /dev/video0')
    parser.add_argument('--inference'   , action='store_true', help='engine model deserialize and run inference')

    args = parser.parse_args()
    
    return args

'''
    Usage:
        python yolov5_cls_trt.py  engine_path/xxx.engine  
'''
if __name__ == "__main__":
    # load custom plugin and engine
    engine_file_path = "/home/seeed/github/tensorrtx/yolov5/build/yolov5m-cls.engine"

#    # USB-Camera
#    source = ("v4l2src device=/dev/video1 ! image/jpeg,framerate=30/1,width=640, height=480,type=video ! "
#                    "jpegdec ! videoconvert ! video/x-raw ! appsink")

    # Rtsp-IPCamera
    source = ('rtspsrc location=rtsp://192.168.111.118:8554/live ! ''rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv !'
               'video/x-raw,width=800,height=480,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink ')

    # CSI-Camera
#    source = ("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

#    args = parse_args()
#    print("-e:%s" % args.engine)
#    print("-s:%s" % args.source)


#    if len(sys.argv) > 1:
#        engine_file_path = sys.argv[1]

    # get stream source
    dataloader = LoadStreams(source)

    # a YoLov5TRT instance
    yolov5_wrapper = YoLov5TRT(engine_file_path)

    try:
        for i in range(5):
            # create a new thread to do warm_up
            thread1 = warmUpThread(yolov5_wrapper)
            thread1.start()
            thread1.join()

        for path, im, im0s, vid_cap, s in dataloader:
            for img in im0s:
                thread1 = inferThread_R(yolov5_wrapper, img)
                thread1.start()
                thread1.join()

    finally:
        # destroy the instance
        yolov5_wrapper.destroy()

