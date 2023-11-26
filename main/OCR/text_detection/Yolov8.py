from ultralytics import YOLO
import numpy as np
import os
import cv2

class TextDetector:
    def __init__(self, model_path, save_path):
        self.save_path = save_path
        self.model_path = model_path

    def detect_text_in_image(self, image_path):
        """Detects text in the given images using YOLOv8.

        Args:
            image_path: A path to the image.

        Returns:
            A list of lists of text detections, where each text detection is a tuple
            of (bounding_box).
        """

        # Load the YOLOv8 model.
        # Load a model
        model = YOLO('yolov8n.yaml')  # build a new model from YAML
        model = YOLO(self.model_path)

        image = cv2.imread(image_path)

        # Perform object detection on the image.
        yolo_detections = model.predict(image)

        # List of bounding boxes
        bounding_boxes_list = []

        for detection in yolo_detections:
            # Get the bounding box of the text.
            for bounding_boxes in detection.boxes.xyxy:
                xmin = int(bounding_boxes.tolist()[0])
                ymin = int(bounding_boxes.tolist()[1])
                xmax = int(bounding_boxes.tolist()[2])
                ymax = int(bounding_boxes.tolist()[3])

                # Append the bounding box to the list of bounding boxes
                bounding_boxes_list.append((xmin, ymin, xmax, ymax))

        return bounding_boxes_list
    
    def non_max_suppression_fast(boxes, overlapThresh):
        # Convert the list into 2d numpy array
        boxes = np.array(boxes)

        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []
        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        # initialize the list of picked indexes	
        pick = []
        # grab the coordinates of the bounding boxes
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))
        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")
    
    def delete_cropped_images(self):
        """Deletes the cropped images in the temp folder.
        """
        for filename in os.listdir(self.save_path):
            os.remove(self.save_path + '\\' + filename)