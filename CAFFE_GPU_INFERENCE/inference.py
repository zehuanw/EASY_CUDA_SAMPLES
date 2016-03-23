#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import urllib
import urllib2
import StringIO
import time
import numpy as np
import skimage
from skimage.io import imread
from skimage.transform import resize
from Queue import Empty
import gevent.monkey
gevent.monkey.patch_socket()
gevent.monkey.patch_os()
from gevent.pool import Pool
import multiprocessing
from multiprocessing import TimeoutError
from multiprocessing import Queue, Process, Pool
from multiprocessing import Lock, Manager
import affinity
import ctypes
import cv2

CAFFE_ROOT = "/root/work/caffe-master-new/"
DEPLOY_FILE = "/root/work/caffe-master-new/models/bvlc_alexnet/deploy.prototxt"#"/root/lu/112_Youtu/VGG_ILSVRC_16_layers_deploy.prototxt"
MODEL_FILE = "/root/work/caffe-master-new/models/bvlc_alexnet/caffe_alexnet_train_iter_100.caffemodel"#"/root/lu/112_Youtu/VGG_ILSVRC_16_layers.caffemodel"#
MEAN_FILE = "/root/work/caffe-master-new/python/caffe/imagenet/ilsvrc_2012_mean.npy"#"/root/lu/112_Youtu/meanfile.npy"#
TEST_SET = "/root/lu/112_Youtu/data/VOCdevkit/VOC2007/JPEGImages/"
URL_FILE = "/root/lu/112_Youtu/data/VOCdevkit/VOC2007/ImageSets/Layout/trainval.txt"

sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe


WORKER_NUM = 4
IMAGE_NUM = 10
PREDICT_BATCH_NUM = 1

class CaffePredictor:
    _cudalib = []
    _net = []
    _deploy_file = ""
    _model_file = ""
    def __init__(self,gid,deploy_file = DEPLOY_FILE, model_file = MODEL_FILE, batchsize = PREDICT_BATCH_NUM):
        self._cudalib = ctypes.CDLL('libcudart.so')
        self._deploy_file = deploy_file
        self._model_file = model_file
        caffe.set_mode_gpu()
        caffe.set_device(gid)
        self._net = caffe.Net(self._deploy_file, self._model_file,caffe.TEST)
        self._net.blobs['data'].reshape(1,3,224,224)  #we can reshape the batchsize if required. It's ugly but easy
        self._cudalib.cudaProfilerStart()
    def net_pred(self,img_batch):
        try:
            out = self._net.forward_all(**{self._net.inputs[0]:img_batch})
            preds = out['prob'].argmax()
            return preds
        except Exception, e:
            return e
    def __del__(self):
        self._cudalib.cudaProfilerStop()
        
    
class Downloader:
    _test_set = []
    _url_file = []
    _url_iter = []
    def get_url(self):
        for url in self._url_file:
            yield url
    def __init__(self,test_set = TEST_SET, url_file = URL_FILE):
        self._test_set = test_set
        self._url_file = open(url_file)
        self._url_iter = self.get_url()
    def download_one(self):
        url = self._url_iter.next()
        try:
            t1 = time.time()
            #f = open(self._test_set + url.strip() + '.jpg')
            #s = f.read()
            t2 = time.time()
            #g1 = skimage.io.imread(StringIO.StringIO(s))
            g = cv2.imread(self._test_set + url.strip() + '.jpg')
#            print (g - g1)
            t3 = time.time()
            img = np.float32(g)
            #img = skimage.img_as_float(g).astype(np.float32)#no need to int2float!?
            t4 = time.time()
            if img.ndim == 2:
                img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            t5 = time.time()
            t6 = time.time()
            print "download",1000*(t2-t1),1000*(t3-t2),1000*(t4-t3),1000*(t5-t4),1000*(t6-t5)
            return (img, url)
        except Exception, e:
            return e
    def __del__(self):
        self._url_file.close()
        

class Preprocessor:
    _image_dims = []
    _crop_dims = []
    _transformer = []
    def __init__(self,net,image_dims = (256, 256),mean_file = MEAN_FILE):
        self._mean_file = mean_file
        self._image_dims = image_dims
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (0, 1, 2))
        self._transformer = transformer
        self._crop_dims = net.blobs['data'].data.shape[-2:]

    def preprocess(self, img_in, use_oversample = True):
        try:
            img_batch = []
            img_batch.append(img_in)
            if use_oversample == True:
                img = np.zeros((len(img_batch),self._image_dims[0], self._image_dims[1], img_in.shape[2]), dtype=np.float32)
                for ix, in_ in enumerate(img_batch):
#                    img[ix] = caffe.io.resize_image(in_, self._image_dims)
                    img[ix] = cv2.resize(in_,self._image_dims)
                img2 = caffe.io.oversample(img, self._crop_dims) #image will be copied 10 times refer to io.py
                img_out = np.zeros(np.array(img2.shape)[[0, 3, 1, 2]], dtype=np.float32) #some opt?
                for ix, in_ in enumerate(img2):
                    img_out[ix] = self._transformer.preprocess('data', in_)
                return img_out
            else:
                t1 = time.time()
                img = np.zeros((len(img_batch),self._crop_dims[0], self._crop_dims[1], img_in.shape[2]), dtype=np.float32)
                t2 = time.time()
                for ix, in_ in enumerate(img_batch):
#                    img[ix] = caffe.io.resize_image(in_, self._crop_dims)
                    img[ix] = cv2.resize(in_,self._crop_dims)
                t3 = time.time()
                img_out = np.zeros(np.array(img.shape)[[0, 3, 1, 2]], dtype=np.float32) #some opt?
                t4 = time.time()
                for ix, in_ in enumerate(img):
                    img_out[ix] = self._transformer.preprocess('data', in_)
                t5 = time.time()
                print "preprocess",1000*(t2-t1),1000*(t3-t2),1000*(t4-t3),1000*(t5-t4)
                return img_out
        except Exception, e:
            return e


def inference_process(event,gid=0):
    pid = os.getpid()
    mask = affinity.get_process_affinity_mask(pid)
    affinity.set_process_affinity_mask(pid,(1 << pid%20))

    net = CaffePredictor(gid)
    dl = Downloader()
    prep = Preprocessor(net._net)

    #warming up
    input_pair = dl.download_one()
    img = prep.preprocess(input_pair[0],use_oversample = False)
    pred = net.net_pred(img)
    event.wait()
    t0 = time.time()
    for i in range(IMAGE_NUM):
        t1 = time.time()
        input_pair = dl.download_one()
        t2 = time.time()
        img = prep.preprocess(input_pair[0],use_oversample = False)
        t3 = time.time()
#        img = prep.preprocess(input_pair[0],use_oversample = True)
        pred = net.net_pred(img)
        t4 = time.time()
        print("gpu# %d, class %d,  download %.6fms, preprocess %.6fms,net_pred %.6fms" % (gid,pred,(t2-t1)*1000,(t3-t2)*1000,(t4-t3)*1000))
    t5 = time.time()
    print("gpu# %d, average time %.6fms" % (gid, 1000*(t5-t0)/IMAGE_NUM))
    del net
    del dl
    del prep

        
def case1():
    net = CaffePredictor(0)
    dl = Downloader()

    #warming up
    prep = Preprocessor(net._net)
    input_pair = dl.download_one()
    img = prep.preprocess(input_pair[0],use_oversample = False)
    pred = net.net_pred(img)
    for i in range(IMAGE_NUM):
        t1 = time.time()
        input_pair = dl.download_one()
        t2 = time.time()
        img = prep.preprocess(input_pair[0],use_oversample = False)
        #        img = prep.preprocess(input_pair[0],use_oversample = True)
        t3 = time.time()
        pred = net.net_pred(img)
        t4 = time.time()
        print("class %d, download %.6fms, preprocess %.6fms,net_pred %.6fms" % (pred,(t2-t1)*1000,(t3-t2)*1000,(t4-t3)*1000))
 
    del net
    del dl
    del prep

    
def case2():
    event = multiprocessing.Event()
    workers = [Process(target = inference_process, args=(event,i)) for i in range(WORKER_NUM)]
    for worker in workers:
        worker.start()
    time.sleep(10)
    event.set()
    for worker in workers:
        worker.join()
    time.sleep(10)
        
        
if __name__ == "__main__":
    case1()
    print("ok.")

