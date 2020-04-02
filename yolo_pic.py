from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("det1.h5")
detector.setJsonPath("det1.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="sign.png", output_image_path="holo3-detected.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])