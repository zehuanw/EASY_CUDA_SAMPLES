#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import urllib
import urllib2
import StringIO
import time
import numpy as np
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


CAFFE_ROOT = "/root/work/caffe-master-new/"
DEPLOY_FILE = "/root/lu/112_Youtu/VGG_ILSVRC_16_layers_deploy.prototxt"#"/root/work/caffe-master-new/models/bvlc_alexnet/deploy.prototxt"#
MODEL_FILE = "/root/lu/112_Youtu/VGG_ILSVRC_16_layers.caffemodel"#"/root/work/caffe-master-new/models/bvlc_alexnet/caffe_alexnet_train_iter_100.caffemodel"#
MEAN_FILE = "/root/lu/112_Youtu/meanfile.npy"#"/root/work/caffe-master-new/python/caffe/imagenet/ilsvrc_2012_mean.npy"#
TEST_SET = "/root/lu/112_Youtu/data/VOCdevkit/VOC2007/JPEGImages/"
URL_FILE = "/root/lu/112_Youtu/data/VOCdevkit/VOC2007/ImageSets/Layout/trainval.txt"
DETAIL_TIME = False
HAVE_PYTHON_OPENCV = False

PREPROCESS_WORKER_NUM = 4
PREDICT_WORKER_NUM = 4
DOWNLOAD_WORKER_NUM = 4

if HAVE_PYTHON_OPENCV == True:
    import cv2
else:
    import skimage
    from skimage.io import imread
    from skimage.transform import resize


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
            if HAVE_PYTHON_OPENCV == False:
                f = open(self._test_set + url.strip() + '.jpg')
                s = f.read()
            t2 = time.time()
            if HAVE_PYTHON_OPENCV == False:
                g = skimage.io.imread(StringIO.StringIO(s))
            else:
                g = cv2.imread(self._test_set + url.strip() + '.jpg')
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
            if DETAIL_TIME == True:
                print "download",1000*(t2-t1),1000*(t3-t2),1000*(t4-t3),1000*(t5-t4),1000*(t6-t5)
            return (img, url)
        except Exception, e:
            return e
    def __del__(self):
        self._url_file.close()

class Downloader_v2:
    _test_set = []
    _url_file = []
    _url_iter = []
    def get_url(self):
        for url in self._url_file:
            yield url
    def __init__(self, url_file = URL_FILE):
        self._url_file = open(url_file)
        self._url_iter = self.get_url()
    def __del__(self):
        self._url_file.close()

def download_one(url,test_set = TEST_SET):
    try:
        t1 = time.time()
        if HAVE_PYTHON_OPENCV == False:
            f = open(test_set + url.strip() + '.jpg')
            s = f.read()
        t2 = time.time()
        if HAVE_PYTHON_OPENCV == False:
            g = skimage.io.imread(StringIO.StringIO(s))
        else:
            g = cv2.imread(test_set + url.strip() + '.jpg')
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
        if DETAIL_TIME == True:
            print "download",1000*(t2-t1),1000*(t3-t2),1000*(t4-t3),1000*(t5-t4),1000*(t6-t5)
        return (img, url)
    except Exception, e:
        return (None,url)
        

class Preprocessor:
    _image_dims = []
    _crop_dims = []
    _transformer = []
    def __init__(self,net_interface,image_dims = (256, 256),mean_file = MEAN_FILE):
        self._mean_file = mean_file
        self._image_dims = image_dims
        transformer = caffe.io.Transformer({'data': net_interface})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
        transformer.set_raw_scale('data', 255)
        if HAVE_PYTHON_OPENCV == False:
            transformer.set_channel_swap('data', (2, 1, 0))
        else:
            transformer.set_channel_swap('data', (0, 1, 2))
        self._transformer = transformer
        self._crop_dims = net_interface[-2:]

    def preprocess(self, img_in, use_oversample = True):
        try:
            img_batch = []
            img_batch.append(img_in)
            if use_oversample == True:
                img = np.zeros((len(img_batch),self._image_dims[0], self._image_dims[1], img_in.shape[2]), dtype=np.float32)
                for ix, in_ in enumerate(img_batch):
                    if HAVE_PYTHON_OPENCV == False:
                        img[ix] = caffe.io.resize_image(in_, self._image_dims)
                    else:
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
                    if HAVE_PYTHON_OPENCV == False:
                        img[ix] = caffe.io.resize_image(in_, self._crop_dims)
                    else:
                        img[ix] = cv2.resize(in_,self._crop_dims)
                t3 = time.time()
                img_out = np.zeros(np.array(img.shape)[[0, 3, 1, 2]], dtype=np.float32) #some opt?
                t4 = time.time()
                for ix, in_ in enumerate(img):
                    img_out[ix] = self._transformer.preprocess('data', in_)
                t5 = time.time()
                if DETAIL_TIME == True:
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
    prep = Preprocessor(net._net.blobs['data'].data.shape)

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
        #img = prep.preprocess(input_pair[0],use_oversample = True)
        t3 = time.time()
        pred = net.net_pred(img)
        t4 = time.time()
        if DETAIL_TIME == True:
            print("gpu# %d, class %d,  download %.6fms, preprocess %.6fms,net_pred %.6fms" % (gid,pred,(t2-t1)*1000,(t3-t2)*1000,(t4-t3)*1000))
        print("gpu# %d, class %d,  download %.6fms, preprocess %.6fms,net_pred %.6fms" % (gid,pred,(t2-t1)*1000,(t3-t2)*1000,(t4-t3)*1000))
    t5 = time.time()
    print("gpu# %d, average time %.6fms" % (gid, 1000*(t5-t0)/IMAGE_NUM))
    del net
    del dl
    del prep

def preprocess_process(event,fileQueue,imgQueue,net_interface):
    pid = os.getpid()
    mask = affinity.get_process_affinity_mask(pid)
    affinity.set_process_affinity_mask(pid,(1 << pid%20))
    prep = Preprocessor(net_interface)
    event.wait()
    while True:
        try:
            img_in, url = fileQueue.get(timeout=6)
            img = prep.preprocess(img_in,use_oversample = False)
            imgQueue.put((img,url),block=True)
        except (TimeoutError, Empty):
            print("finish preproc: %.6f" % (time.time()))
            del prep
            return None
        
def predict_process(event,imgQueue,gid=0):
    pid = os.getpid()
    mask = affinity.get_process_affinity_mask(pid)
    affinity.set_process_affinity_mask(pid,(1 << pid%20))
    net = CaffePredictor(gid)
    event.wait()
    while True:
        try:
            img, url = imgQueue.get(timeout=6)
            pred = net.net_pred(img)#batchsize
            print("gpu# %d, url %s, class %d" % (gid,url.strip(),pred)) 
        except (TimeoutError, Empty):
            print("finish predict: %.6f" % (time.time()))
            del net
            return None
    
    
def case1():
    net = CaffePredictor(0)
    dl = Downloader()

    #warming up
    prep = Preprocessor(net._net.blobs['data'].data.shape)
    input_pair = dl.download_one()
    img = prep.preprocess(input_pair[0],use_oversample = False)
    pred = net.net_pred(img)
    for i in range(IMAGE_NUM):
        t1 = time.time()
        input_pair = dl.download_one()
        t2 = time.time()
        img = prep.preprocess(input_pair[0],use_oversample = False)
        #img = prep.preprocess(input_pair[0],use_oversample = True)
        t3 = time.time()
        pred = net.net_pred(img)
        t4 = time.time()
        print("class %d, download %.6fms, preprocess %.6fms,net_pred %.6fms" % (pred,(t2-t1)*1000,(t3-t2)*1000,(t4-t3)*1000))
    del net
    del dl
    del pred
    
def case2():
    event = multiprocessing.Event()
    workers = [Process(target = inference_process, args=(event,i)) for i in range(WORKER_NUM)]
    for worker in workers:
        worker.start()
    time.sleep(30)
    event.set()
    for worker in workers:
        worker.join()
    time.sleep(10)
        
def case3():
    event = multiprocessing.Event()
    net_interface = (1,3,224,224)#net._net.blobs['data'].data.shape
    fileQueue, imgQueue = Queue(maxsize=40), Queue(maxsize=40)
    preprocess_workers = [Process(target = preprocess_process, args=(event,fileQueue,imgQueue,net_interface)) for i in range(PREPROCESS_WORKER_NUM)]
    predict_workers = [Process(target = predict_process, args=(event,imgQueue,i)) for i in range(PREDICT_WORKER_NUM)]
    dl = Downloader_v2()
    for preprocess_worker in preprocess_workers:
        preprocess_worker.start()
    for predict_worker in predict_workers:
        predict_worker.start()
    time.sleep(30)
    event.set()
    p = Pool(DOWNLOAD_WORKER_NUM)
    for f in p.imap_unordered(download_one, dl._url_iter):
        fileQueue.put(f, block = True)
    for preprocess_worker in preprocess_workers:
        preprocess_worker.join()
    for predict_worker in predict_workers:
        predict_worker.join()


if __name__ == "__main__":
    case3()
    print("ok.")

