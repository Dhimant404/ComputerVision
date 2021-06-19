import cv2
import numpy as np


classes = []
with open('coco.names','r') as f:
    classes = f.read().splitlines()


modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'
net = cv2.dnn.readNet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



cap = cv2.VideoCapture("video.mp4")
width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))
size = (width,height)

out = cv2.VideoWriter('sample_output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 25, size)

_, frame1 = cap.read()
_, frame2 = cap.read()

def get_all_boxes(layer_outputs):

    all_boxes = []
    confidences = [] 

    for output in layer_outputs:
        for detection in output:

            scores=detection[5:]                
            class_id=np.argmax(scores)          
            confidence =scores[class_id]

            if confidence > 0.5 and class_id==0:

                center_x=int(detection[0]*width)
                center_y=int(detection[1]*height)
                w=int(detection[2]*width)
                h=int(detection[3]*height)

                x=int(center_x-w/2) 
                y=int(center_y-h/2)

                all_boxes.append([x,y,w,h])
                confidences.append(float(confidence))

    return all_boxes, confidences


def get_bounding_boxes(frame):

    blob = cv2.dnn.blobFromImage(frame,1/255,(320,320),(0,0,0),1,crop=False)
    net.setInput(blob)

    output_layer_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layer_names)

    all_boxes, confidences = get_all_boxes(layer_outputs)

    indexes=cv2.dnn.NMSBoxes(all_boxes,confidences,0.5,0.3)

    return indexes, confidences, all_boxes

def mid_point(img, box):
    x1, y1, w, h = box[0], box[1], box[2], box[3]
    x2, y2 = x1+w, y1+h
  
    x_mid = int((x1+x2)/2)
    y_mid = int(y2)
    mid = (x_mid,y_mid)
  
    _ = cv2.circle(img, mid, 5, (255, 0, 0), -1)
  
    return mid

def draw_boxes(indexes, frame, all_boxes):

    bbox = []
    mid_points = []

    for i in indexes:
        x = i[0]
        box = all_boxes[x]
        bbox.append(box)
        mid_points.append(mid_point(frame, box))
        x1, y1, w, h = box[0], box[1], box[2], box[3]
        x2, y2 = x1+w, y1+h

        cv2.rectangle(frame, (x1,y1),(x2,y2),(255,0,0),2) 

    return mid_points, bbox

def compute_distance(point_1, point_2):

    x1, y1, x2, y2 = point_1[0], point_1[1], point_2[0], point_2[1]
    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)

    return distance

def get_distances_list(mid_points):

    n = len(mid_points)
    dist_list = np.zeros((n,n))

    for i in range(n):
        for j in range(i+1, n):
            dist_list[i][j] = compute_distance(mid_points[i], mid_points[j])
    
    return dist_list


def find_closest(distances, threshold):

    n = len(distances)
    person_1 = []
    person_2 = []
    d = []

    for i in range(n):
        for j in range(i+1, n):
            if distances[i][j] <= threshold:
                person_1.append(i)
                person_2.append(j)
                d.append(distances[i][j])

    return person_1, person_2, d


def change_bbox_color(img, boxes, p1, p2):

    points = np.unique(p1 + p2)

    for i in points:
        x1, y1, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        x2, y2 = x1+w, y1+h
        _ = cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)  

    return img

while cap.isOpened():
    indexes, confidences, all_boxes = get_bounding_boxes(frame1)
    mid_points, bounding_boxes = draw_boxes(indexes, frame1, all_boxes)
    distances_list = get_distances_list(mid_points)
    p1,p2,d = find_closest(distances_list,100)
    img = change_bbox_color(frame1, bounding_boxes, p1, p2)
    out.write(img)
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break


cv2.destroyAllWindows()
cap.release()
out.release()

            


