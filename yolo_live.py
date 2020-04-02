import cv2

from imageai.Detection.Custom import CustomObjectDetection

cap = cv2.VideoCapture(0)
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("det.h5")
detector.setJsonPath("det.json")
print("done")
detector.loadModel()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    dim = (352, 288)
    # resize image
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    #print(frame)
    cv2.imwrite('mypic.jpg',frame)

    # Our operations on the frame come here
    detections = detector.detectObjectsFromImage(input_image="mypic.jpg", output_image_path="holo3-detected.jpg")
    a = cv2.imread('holo3-detected.jpg')
    cv2.imshow('',a)
    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

    # Display the resulting frame
    #cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()