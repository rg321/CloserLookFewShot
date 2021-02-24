# /content/drive/MyDrive/frinks/Cam_1/Lunch/headCrops_prediction/masked

flag = 'torch'
flag_fewShots = True

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5' 
 
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model as LoadModel
import argparse
import numpy as np
import time
import cv2
from scipy.spatial import distance as dist
import json
 
import torchvision
import torch
from torchvision import transforms

from fastai.imports import *
from fastai.vision import *

import warnings
warnings.filterwarnings('ignore')
from google.colab.patches import cv2_imshow

from timer import Timer

from methods.baselinetrain import BaselineTrain
import backbone
 
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
 
 
parser = argparse.ArgumentParser(description='HawkAI')
# parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
# parser.add_argument('--frame', default='test.png', help='image file input path')
# parser.add_argument('--outpath', default='result.avi', help='image file output path')
#***************  WARNING : ONLY USE save_video = True when running on a video, but not on cctv footages ******************************
# parser.add_argument('--save_image', default=True, type = str2bool) 
parser.add_argument('--maskDet', default=True, type = str2bool)
# parser.add_argument('--gloveDet', default=False, type = str2bool)
# parser.add_argument('--socDis', default=False, type = str2bool)
# parser.add_argument('--DistCalibrated', default=False, type = str2bool)
# parser.add_argument('--highRisk', default=False, type = str2bool)
# parser.add_argument('--num_frames', default=1, help='number of frames processed till now')
args = parser.parse_args()

 
#Global timer
'''
_t = {'faceFP': Timer(), 
'faceMisc': Timer(), 
'maskDet': Timer(),
'handFP': Timer(),
'handMisc': Timer(),
'gloveDet': Timer(),
'peopleFP': Timer(),
'distanceCalc': Timer(),
'socDisMisc': Timer(),
'wristFP': Timer(),
'depthFP': Timer(),
'detectron' : Timer(),
'highRiskMisc':Timer()}
'''
 

# ////////////////////////////
def load_mask_detector():
    ############### Change path to path of model ###################
    # path_model='/content/drive/MyDrive/frinks/models/fastai_resnet101'
    path_model='/content/drive/MyDrive/frinks/fewShots/CloserLookFewShot/checkpoints_masks_Conv4_baseline_aug/20.tar'
    path_data='/content/drive/MyDrive/frinks/Faces/data'
    if flag is 'torch':
      if not flag_fewShots:
        model = torch.load(path_model)
      if flag_fewShots:
        # import pdb; pdb.set_trace()
        model = BaselineTrain(backbone.Conv4, 4)
        model_dict = torch.load(path_model)
        model.load_state_dict(model_dict['state'])
        model=model.cuda()
    elif flag is 'fastai':
      data = ImageDataBunch.from_folder(path_data, valid_pct=0.2, size = 120)
      model = cnn_learner(data, models.resnet101, metrics=error_rate)
      model.load(path_model)
    else:
      model = LoadModel(path_model)
    return model
 
 
def getHeadCrops(points):
    #Points are in the order: 0-Nose, 1-Neck, 17-REar, 18-LEar , 15-LEye, 16-REye, 8-Hip
    #And every keypoint has: x, y, acc
 faceCrops = []
#  helCrops = []
#  maskCrops = []
 for i in range(len(points)):
  left=int(10000)
  left_y=int(10000)
  if((points[i][0][2] < 0.5) or (points[i][15][2] < 0.5 and points[i][16][2] < 0.5)):
    continue
  # if(points[i][17][2] == 0 or points[i][18][2] == 0):
  #   continue
  for j in (0, 15, 16, 17, 18):
   if(points[i][j][0] != 0):
    left = int(min(left, points[i][j][0]))
   if(points[i][j][1] != 0):
    left_y = int(min(left_y, points[i][j][1]))
  if(left == 0):
    continue
  if(left_y == 0):
    continue
  right = int(max(points[i][18][0], points[i][0][0], points[i][17][0], points[i][15][0], points[i][16][0]))
  right_y = int(max(points[i][18][1], points[i][0][1], points[i][17][1], points[i][15][1], points[i][16][1]))
  
  temp = (right - left) / 4
  left = left - temp
  right = right + temp
  
  # bottom = int((points[i][0][1] + 3*points[i][1][1])/4)
  bottom = int(max(points[i][1][1], points[i][2][1], points[i][5][1]))
  # top = int(max(0, 2*points[i][0][1]-points[i][1][1]))
  top = int(bottom - 2*(max(points[i][8][1], points[i][9][1], points[i][12][1]) - bottom)/3)
  if(int(max(points[i][1][1], points[i][2][1], points[i][5][1])) == 0 or max(points[i][8][1], points[i][9][1], points[i][12][1]) == 0):
    faceCrops.append((((right+left)/2 - (right-left)), ((left_y+right_y)/2 - (right-left)), (right-left)*2, ((right-left)*2)))
    # helCrops.append((((right+left)/2 - (right-left)), ((left_y+right_y)/2 - (right-left)), (right-left)*2, ((right-left))))
    # maskCrops.append((((right+left)/2 - (right-left)), (left_y+right_y)/2, (right-left)*2, ((right-left))))
    continue
  temp = (left + right) / 2
 
  left = int((temp + points[i][1][0]) / 2 + 7*(top - bottom)/12)
  right = int((temp + points[i][1][0]) / 2 - 7*(top - bottom)/12)
 
  temp = (bottom - top) / 8
  top = top - temp
  bottom = bottom - temp
  
  temp = (bottom - top)
  top = (left_y + right_y) / 2 - 7*temp/12
  bottom = (left_y + right_y) / 2 + 7*temp/12
 
  # left = int(points[i][1][0] + (top - bottom)/2)
  # right = int(points[i][1][0] - (top - bottom)/2)
 
  if(top == 0):
    top = bottom + 1.5*(right-left)
  # if(points[i][17][0] != 0 and points[i][18][0] != 0 and (points[i][17][0] < points[i][18][0])):
  #   continue
  # if(points[i][15][0] != 0 and points[i][16][0] != 0 and (points[i][16][0] < points[i][15][0])):
  #   continue
 
  ####################################### FOR DEMO ONLY ###############################################
  ####################################### DECREASE BOX SIZE ###########################################
  if False:
    box_size = 3 * (right - left) / 4
    faceCrops.append(((right+left)/2 - box_size/2, (bottom+top)/2 - box_size/2 - box_size/5, box_size, box_size+box_size/5))
    # helCrops.append(((right+left)/2 - box_size/2, (bottom+top)/2 - box_size/2 - box_size/5, box_size, (box_size+box_size/5)/2))
    # maskCrops.append(((right+left)/2 - box_size/2, (bottom+top)/2 - box_size/2 - box_size/5 + (box_size+box_size/5)/2, box_size, (box_size+box_size/5)/2))
    # continue
  else:
    faceCrops.append((left, top, right-left, bottom-top))
    # helCrops.append((left, top, right-left, (bottom-top)/2))
    # maskCrops.append((left, top + (bottom-top)/2, right-left, (bottom-top)/2))
  #####################################################################################################
 
  # crops.append((left, top, right-left, bottom-top))
 return faceCrops
#  return helCrops, maskCrops
 
 
 
# # @ray.remote
# def getBodyCrops(points):
#   crops = []
#   for i in range(len(points)):
#       left, right, top, bottom = 10000,0,10000,0
#       for j in range(25):
#           if points[i][j][2]!=0 and left>points[i][j][0]:
#               left = int(points[i][j][0])
#           if points[i][j][2]!=0 and right<points[i][j][0]:
#               right = int(points[i][j][0])
#           if points[i][j][2]!=0 and top>points[i][j][1]:
#               top = int(points[i][j][1])
#           if points[i][j][2]!=0 and bottom<points[i][j][1]:
#               bottom = int(points[i][j][1])
#       temp = int((right - left) / 4)
#       left = left - temp
#       right = right + temp
#       top = top - temp
#       bottom = bottom + temp
#       crops.append((left, top, right, bottom,(int((left+right)/2), int((top+bottom)/2))))
#   return crops
 
# @ray.remote
def detectFaceMask(frame, dets, _t):
        
        maskPreds = []
        detFaces = np.array([])
        im_height, im_width, _ = np.asarray(frame).shape
        
        _t['faceMisc'].tic()
        for i, b in enumerate(dets):
                startX, startY, endX, endY = int(b[0]), int(b[1]), int(b[2]+b[0]), int(b[3]+b[1])
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(im_width - 1, endX), min(im_height - 1, endY))
                cropped = frame[startY:endY, startX:endX,:]
                if cropped.any():
                  cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                  cropped = cv2.resize(cropped, (128, 128))
                  cropped = img_to_array(cropped)
                  # cropped = preprocess_input(cropped)
                  cropped = cropped / 255.0
                  cropped = np.expand_dims(cropped, axis=0)
                  if detFaces.size ==0:
                    detFaces = cropped
                  else:
                    detFaces = np.vstack((detFaces, cropped))
# ////////////////////////////     
        if flag is 'torch':
            import pdb; pdb.set_trace()
            print('shape before transforms .. ')
            print(detFaces.shape)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            detFaces=torch.Tensor(detFaces)
            if not flag_fewShots:
              detFaces=torch.reshape(detFaces, (-1,3,128,128))
            if flag_fewShots:
              detFaces=detFaces.permute(0,3,1,2)
              tr=transforms.Compose([transforms.Resize((84,84))])
              detFaces=tr(detFaces)
              # detFaces=torch.reshape(detFaces, (-1,3,84,84)) # doesnot work

            detFaces = detFaces.to(device)
                   
 
            # print(detFaces.shape)
        if flag is 'fastai':
          if detFaces.size != 0:
            detFaces=torch.Tensor(detFaces)
            detFaces=detFaces.permute(0,3,1,2)

            # print(detFaces.shape)
        _t['faceMisc'].toc()
 
        # ************** starting prediction -----------------------------------------------
        _t['maskDet'].tic()
        ### Mask detection ###
        
        if len(detFaces) > 0:
                # with tf.device(settings.maskCpu):
                if flag is 'torch':
                      print("detfaces")
                      print(detFaces.size())
                      logps = maskNet.forward(detFaces)
                      maskPreds = torch.exp(logps)
                      # probability and class of prediction. 0=masked, 1=unmasked. uncomment following line
                      # top_p, top_class = maskPreds.topk(1, dim=1)
                      print("maskPreds")
                      print(maskPreds.size())
                      maskPreds = maskPreds.cpu().detach().numpy()
                if flag is 'fastai':
                  maskPreds = []
                  # print('-------- entering fastai ----')
                  # print('len is ', len(detFaces))
                  for i in range(len(detFaces)):
                      t=Image(detFaces[i])
                      pred = maskNet.predict(t)
                      # top_p, top_class = pred.topk(1, dim=1)
                      # print('prediction is ', pred[2])
                      maskPreds.append(pred[2])
                else:
                    maskPreds = maskNet.predict(detFaces, batch_size=32)
        _t['maskDet'].toc()
        return (maskPreds, _t)
 
 
if args.maskDet:
        print("Loading Face and Mask detector cascades...")
        maskNet = load_mask_detector()
 
 
 
# @ray.remote
def run_on_frame(frame, keypoints, img_path, path_results):
        
        starttt = time.time()
        _t = {'faceFP': Timer(), 
        'faceMisc': Timer(), 
        'maskDet': Timer(),
        'handFP': Timer(),
        'handMisc': Timer(),
        'gloveDet': Timer(),
        'peopleFP': Timer(),
        'distanceCalc': Timer(),
        'socDisMisc': Timer(),
        'wristFP': Timer(),
        'depthFP': Timer(),
        'detectron' : Timer(),
        'highRiskMisc':Timer()}
        
        if frame is None:
                return None
        else:
                # import pdb; pdb.set_trace()
                frame_raw = frame.copy()
 
                frame_raw = np.array(frame_raw)
                # get the iuv array if either mask or glove detection is enabled
                _t['faceFP'].tic()        
                head_crops = getHeadCrops(keypoints)
                _t['faceFP'].toc()
 
                # body_crops = getBodyCrops(keypoints)
 
                
                if args.maskDet:
                        maskPreds, _t = detectFaceMask(frame, head_crops, _t)
                        print('--- got maskPreds ---')
                        
                      
                        # img = Image.open(image_path)
                        for b,lab in zip(head_crops, maskPreds):
                                if lab[1] > 0.5:
                                        b = list(map(int, b)) 
                                        y_start=b[1]
                                        y_end=b[3]+b[1]
                                        x_start=b[0]
                                        x_end=b[2]+b[0]
                                        print('no mask ----')
                                        text = "No mask : {}".format(lab)
                                        final_path=os.path.join(path_results , 'unmasked', image_name)
                                        print('final_path ',final_path)
                                        if not os.path.exists(final_path):
                                          cropped_img=frame_raw[y_start:y_end,x_start:x_end].copy()
                                          cv2.imwrite(final_path, (cropped_img))
                                        # print('path exists ', os.path.exists(final_path))
                                        # print(b)
                                        # im=frame_raw[b[0]:b[2]+b[0], b[1]:b[3]+b[1]].copy()
                                        # area = (b[0],  b[1], b[2]+b[0], b[3]+b[1])
                                        # cropped_img = img.crop(area).save(final_path)
                                        # cv2_imshow(im)
                                        # cv2_imshow(frame)
                                        # print('writing image ---------')
                                        # cv2.imwrite(final_path, (im))
                                        # cv2.imwrite(final_path, (cropped_img))
                                        b = list(map(int, b))    
                                        cv2.rectangle(frame_raw, (b[0], b[1]), (b[2]+b[0], b[3]+b[1]), (0, 0, 255), 2)
                                        cx = b[0]
                                        cy = b[1] + 12
                                        cv2.putText(frame_raw, text, (cx, cy),cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
 
                                elif lab[1] <= 0.5:
                                        print('mask ----')
                                        b = list(map(int, b)) 
                                        y_start=b[1]
                                        y_end=b[3]+b[1]
                                        x_start=b[0]
                                        x_end=b[2]+b[0]
                                        text = "mask : {}".format(lab)
                                        final_path=os.path.join(path_results , 'masked', image_name)
                                        print('final_path ',final_path)
                                        if not os.path.exists(final_path):
                                          cropped_img=frame_raw[y_start:y_end,x_start:x_end].copy()
                                          cv2.imwrite(final_path, (cropped_img))
                                        # print('path exists ', os.path.exists(final_path))
                                        # print(b)
                                        # im=frame[b[1]:b[3]+b[1],b[0]:b[2]+b[0]].copy()
                                        # cv2_imshow(im)
                                        # cv2_imshow(frame)
                                        # print('writing image -------------------')
                                        # cv2.imwrite(final_path, (im))
                                        # b = list(map(int, b))    
                                        cv2.rectangle(frame_raw, (b[0], b[1]), (b[2]+b[0], b[3]+b[1]), (0, 255, 0), 2)
                                        cx = b[0]
                                        cy = b[1] + 12
                                        cv2.putText(frame_raw, text, (cx, cy),cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                                
                # if args.socDis:
 
                #         violate, centroids, results, dists, _t = detectSocDis(frame, body_crops, _t)
                #         # wScale = float(w/w1)
                #         # hScale = float(h/h1)
 
                #         for (i, (startX, startY, endX, endY, centroid)) in enumerate(results):
                #                 #(startX, startY, endX, endY) = (int(startX * wScale), int(startY * hScale), int(endX * wScale), int(endY * hScale))
                #                 (cX, cY) = centroid
                #                 #(cX, cY) = (int(cX * wScale), int(cY * hScale))
                #                 if i in violate:
                #                         color = (0, 0, 255)
                #                         #cv2.circle(frame_raw, (cX, cY), 5, color, 1)
                #                         cv2.rectangle(frame_raw, (startX, startY), (endX, endY), color, 2)
                #                         #            cv2.putText(frame_raw, "Distance: {:.2f}".format(dists[i][0]), (startX, startY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                #                         #cv2.circle(frame_socdis, (cX, cY), 5, color, 1)
                #                 else:
                #                         color = (0, 255, 0)
                #                         #cv2.circle(frame_raw, (cX, cY), 5, color, 1)
                #                         cv2.rectangle(frame_raw, (startX, startY), (endX, endY), color, 2)
                #                         #                cv2.putText(frame_raw, "Distance: {:.2f}".format(dists[i][0]), (startX, startY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
 
                print('-----------------')
                # print('Frame: {:d}/{:d}'.format(num_frames, tot_frames))
                print('faceFP: {:.4f}s faceMisc: {:.4f}s maskDet: {:.4f}s'.format(_t['faceFP'].average_time, _t['faceMisc'].average_time, _t['maskDet'].average_time))
                print('peopleFP: {:.4f}s socDisMisc: {:.4f}s distanceCalc: {:.4f}s'.format(_t['peopleFP'].average_time, _t['socDisMisc'].average_time, _t['distanceCalc'].average_time))
                print('-----------------')
        enddd = time.time()
        print("Time Taken for one instance:                            {}".format(enddd-starttt))
        return frame_raw
         
 
 
if __name__ == '__main__':
        ########################################################################################
        ################### Give path to jsons and images folder################################
        ########################################################################################
        # /content/drive/MyDrive/frinks/Cam_1/Breakfast/frame with json breakfast/Images
        # /content/drive/MyDrive/frinks/Cam_1/Lunch/Frame with json lunch
        # /content/drive/MyDrive/frinks/Video_2
        base_path = "/content/drive/MyDrive/frinks/Video_2/"
        # image_path = base_path+"frame with json breakfast/Images"
        image_path = base_path+"Images"
        # json_path = base_path+"frame with json breakfast/jsons"
        json_path = base_path+"jsons"
        # path_results = base_path + '../headCrops_prediction'
        path_results = '/content/drive/MyDrive/frinks/headCrops_prediction'
        image_files = os.listdir(image_path)
        print('--------- total number of images are ',len(image_files))
        # size = (1280, 720)
        # out = cv2.VideoWriter('/content/drive/MyDrive/frinks/Video_2/demo_mask.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20.0, size)
        ii = 0
        for image_name in image_files:
            img_path = os.path.join(image_path , image_name)
            current_frame = cv2.imread(img_path)
            # if(image_name == '50.jpg'):
            #   break
            ii = ii+1
            # if(ii < 150):
            #   continue
            if(ii % 15 != 0):
              continue
            if(ii >= 1500):
              break
 
            try:
              with open(json_path+'/'+os.path.splitext(image_name)[0]+'_keypoints.json') as f:
                  json_file = json.load(f)
                  json_file = json_file['people']
                  keyy = []
                  for person in range(len(json_file)):
                      keyy.append(np.array(json_file[person]['pose_keypoints_2d']).reshape((25, 3)))
                  current_frame = run_on_frame(current_frame, keyy, img_path, path_results)
            except:
              pass 
            ########### PATH WHERE TO SAVE THE FINAL FRAMES ###################
            # cv2.imwrite(os.path.join('/content' , image_name), (current_frame))
            # cv2.imwrite(os.path.join('/content/drive/MyDrive/frinks/Cam_1/Lunch/fastai_demo' , image_name), (current_frame))
            # out.write(current_frame)
        # out.release()