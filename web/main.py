import copy
import os
import time

import torch
from torchvision import datasets, models, transforms
import io
import tensorflow as tf
import cv2
import datetime, time
import os, sys
import numpy as np
from PIL import Image 
from base64 import b64decode, b64encode
from flask import Flask, json, render_template,jsonify,Response,request
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import cv2 #for resizing image
# from tensorflow import keras
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# model = keras.models.load_model('./my_h5_model.h5')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_ft = torch.load('.\model_18_ver_5.pth')
model_ft.eval()

app = Flask(__name__)


def predict(data_path):
    transform = transforms.Compose([
            # transforms.RandomResizedCrop(219),
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize(28),

            transforms.ToTensor(),
            # transforms.Normalize([0.485], [0.229])
            # transforms.Normalize([0.5], [0.5])
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','Nothing']

    img = Image.fromarray(data_path).convert('RGB').resize((100,100))

    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    batch_t = batch_t.to(device)

    out = model_ft(batch_t)
    values, preds = torch.max(out, 1)

    sm = torch.nn.Softmax(dim=1)
    top_out = sm(out)
    top_prob, top_label = torch.topk(top_out,3)
    prob = top_prob[0][0].cpu().detach().numpy()
    label = "top_label :"+ str(top_label[0][0].cpu().detach().numpy())
    return (class_names[preds[0]], prob)

def nms_pytorch(P : torch.tensor ,thresh_iou : float):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the image 
            along with the class predscores, Shape: [num_boxes,5].
        thresh_iou: (float) The overlap thresh for suppressing unnecessary boxes.
    Returns:
        A list of filtered boxes, Shape: [ , 5]
    """
 
    # we extract coordinates for every 
    # prediction box present in P
    x1 = P[:, 0]
    y1 = P[:, 1]
    x2 = P[:, 2]
    y2 = P[:, 3]
 
    # we extract the confidence scores as well
    scores = P[:, 4]
 
    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)
     
    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()
 
    # initialise an empty list for 
    # filtered prediction boxes
    keep = []
     
 
    while len(order) > 0:
         
        # extract the index of the 
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]
 
        # push S in filtered predictions list
        keep.append(P[idx])
 
        # remove S from P
        order = order[:-1]
 
        # sanity check
        if len(order) == 0:
            break
         
        # select coordinates of BBoxes according to 
        # the indices in order
        xx1 = torch.index_select(x1,dim = 0, index = order)
        xx2 = torch.index_select(x2,dim = 0, index = order)
        yy1 = torch.index_select(y1,dim = 0, index = order)
        yy2 = torch.index_select(y2,dim = 0, index = order)
 
        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])
 
        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1
         
        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
 
        # find the intersection area
        inter = w*h
 
        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim = 0, index = order) 
 
        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]
         
        # find the IoU of every prediction in P with S
        IoU = inter / union
 
        # keep the boxes with IoU less than thresh_iou
        mask = IoU < thresh_iou
        order = order[mask]
     
    return keep

def get_dominant_color(image, k=4, image_processing_size = None):
    """
    takes an image as input
    returns the dominant color of the image as a list
    
    dominant color is found by running k means on the 
    pixels & returning the centroid of the largest cluster

    processing time is sped up by working with a smaller image; 
    this resizing can be done with the image_processing_size param 
    which takes a tuple of image dims as input

    >>> get_dominant_color(my_image, k=4, image_processing_size = (25, 25))
    [56.2423442, 34.0834233, 70.1234123]
    """
    #resize image if new dims provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size, 
                            interpolation = cv2.INTER_AREA)
    
    #reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    #cluster and assign labels to the pixels 
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(image)

    #count labels to find most popular
    label_counts = Counter(labels)

    #subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    return list(dominant_color)

def boxing(img, dilateIteration, erodeIteration, alpha, beta):
    image = img
    # image = cv2.imread(".\\data\\test\\angka\\all.png")
    try:
        # alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
        # beta = int(input('* Enter the beta value [0-100]: '))
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    except ValueError:
        print('Error, not a number')

    # image = cv2.resize(image, (400,400))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (2, 2), 0)
    canny = cv2.Canny(gray,30,250)

    # threshold the image
    # ret,th3 = cv2.threshold(gray ,200,255,cv2.THRESH_BINARY)
    # th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
    # th3 = cv2.adaptiveThreshold(canny,200,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,11)
    ret3,th3 = cv2.threshold(canny,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    # dilate the white portions
    dilate = cv2.dilate(th3, None, iterations=dilateIteration)
    erode = cv2.erode(dilate,None,iterations=erodeIteration)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3, 3))
    erode = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel, iterations=4)
    
    # find contours in the image
    # connectivity = 8  # You need to choose 4 or 8 for connectivity type
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(erode , connectivity , cv2.CV_32S)

    # cnts = cv2.findContours(erode.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    
    mser = cv2.MSER_create()
    mser.setMaxArea(int(400*400/2))
    regions, rects = mser.detectRegions(erode)


    orig = image.copy()
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.3
    fontColor              = (0,0,0)
    thickness              = 1
    lineType               = 1

    i = 0
    # P = []
    # for cnt in cnts:
    #     # Check the area of contour, if it is very small ignore it
    #     if(cv2.contourArea(cnt) < 100):
    #         # print(cv2.contourArea(cnt))
    #         continue

    #     # Filtered countours are detected
    #     x,y,w,h = cv2.boundingRect(cnt)

    #     # Taking ROI of the cotour
    #     roi = image[y:y+h, x:x+w]
    #     prediction = predict(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    #     # print(prediction)
    #     if(prediction[0] == "Nothing"):
    #         continue
    #     P.append([x,y,w,h,prediction[1].tolist()])


    #     # cv2.imwrite("roi" + str(i) + ".png", roi)
    #     i = i + 1
    # print(P)
    # P = torch.tensor(P)
    # # print(P) 
    # filtered_box=nms_pytorch(P,0.5)
    # # print("filtered_box : "+ str(filtered_box))

    i = 0
    for roi in rects:
        # print(roi.numpy())
        # x = stats[j, cv2.CC_STAT_LEFT]
        # y = stats[j, cv2.CC_STAT_TOP]
        # w = stats[j, cv2.CC_STAT_WIDTH]
        # h = stats[j, cv2.CC_STAT_HEIGHT]
        x,y,w,h = roi
        # x,y,w,h = cv2.boundingRect(roi)
        # x,y,w,h,prob = roi.numpy().astype(int)
        # print(x,y,w,h,prob)
        
        roi = image[y:y+h, x:x+w]
        
        value = get_dominant_color(roi, k=4, image_processing_size = (100,100))
        
        # roi = cv2.copyMakeBorder(roi, 20, 20, 20, 20, cv2.BORDER_CONSTANT, None, [255,255,255])
        roi = cv2.copyMakeBorder(roi, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value)
        roi_resize = cv2.resize(roi, (100,100))

        prediction = predict(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        cv2.imwrite("./gathered_data/"+str(prediction[0])+"/"+str(int(time.time()))+str(i)+".png", roi_resize)
        
        if(prediction[0] == "Nothing"):
            continue
        cv2.rectangle(orig,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(orig,str(prediction[0]), 
            # (x+int(w/2),y+int(h)),
            (x,y),  
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
        cv2.putText(orig,str(round(prediction[1]*100,2)) + " %", 
            (x,y+int(h)),
            # (x,y),  
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
        i = i + 1

    return (image, erode, orig)

def js_to_image(js_reply):
  """
  Params:
          js_reply: JavaScript object containing image from webcam
  Returns:
          img: PIL Image
  """
  # decode  
  image_bytes = b64decode(js_reply.split(',')[1])
  
  buf = io.BytesIO(image_bytes)
  img = np.asarray(Image.open(buf))

#   img = Image.open(buf).convert('RGB').resize((100,100))
  return img

def im_2_b64(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buff = io.BytesIO()
    image.save(buff, format="JPEG")
    img_str = b64encode(buff.getvalue()).decode("utf-8")
    return img_str


@app.route('/api', methods =['GET','POST'])
def api():
    if request.method == "GET":
        return jsonify({"class": "no picture"})
    elif request.method == "POST":
        image = request.form.get("content")
        dilateIteration = request.form.get("dilateIteration")
        erodeIteration = request.form.get("erodeIteration")
        alpha = request.form.get("alpha")
        beta = request.form.get("beta")

        print(dilateIteration, erodeIteration)
        img = js_to_image(image)
        result = boxing(img,int(dilateIteration), int(erodeIteration),float(alpha),int(beta))
        dilate_js_img = im_2_b64(result[0])
        bnd_js_img = im_2_b64(result[1])
        boxed_js_img = im_2_b64(result[2])
        # predictions = testing(img)
        dataReply = {'dlt': dilate_js_img, 'bnd': bnd_js_img, 'img': boxed_js_img}
        return jsonify(dataReply)

# @application.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    if request.method=='GET':
        return render_template('test.html')
    return render_template('test.html')

# @application.route('/')
# def homepage():
#     return render_template('test.html',x = name)

app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT",8080)))
