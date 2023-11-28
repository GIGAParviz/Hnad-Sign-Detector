from ultralytics import YOLO
import cv2
import math
cap=cv2.VideoCapture(0)

frame_width=int(cap.get(3))
frame_height = int(cap.get(4))

out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))


text = []


model=YOLO(r"C:\Users\ASUS\OneDrive\Desktop\YOLO-8\Sign Language\modeol\best.pt")
classNames =  ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
while True:
    success, img = cap.read()
    # Doing detections using YOLOv8 frame by frame
    #stream = True will use the generator and it is more efficient than normal
    results=model(img,stream=True)
    #Once we have the results we can check for individual bounding boxes and see how well it performs
    # Once we have have the results we will loop through them and we will have the bouning boxes for each of the result
    # we will loop through each of the bouning box
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            #print(x1, y1, x2, y2)
            x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
            # print(x1,y1,x2,y2)
            #print(box.conf[0])
            conf=math.ceil((box.conf[0]*100))/100
            cls=int(box.cls[0])
            class_name=classNames[cls]
            label=f'{class_name}'#{conf}
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            #print(t_size)
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            # if conf > 0.45:
            cv2.rectangle(img, (x1,y1), c2, [100,100,100], -1, cv2.LINE_AA) 
            cv2.rectangle(img, (x1,y1), (x2,y2), (100,100,100),3)# filled
            cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
            #     if label in text:
            #         pass
            #     else:
            #         text.append(label)
            # s = (f'pred is:' + label)
            # cv2.putText(img, str(text) , (50,100),0, 1,[0,255,0], thickness=1,lineType=cv2.LINE_AA)

    out.write(img)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
out.release()



