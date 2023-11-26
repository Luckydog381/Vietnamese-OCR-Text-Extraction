import os
import cv2

class Utilities:
    def __init__(self, save_path):
        self.save_path = save_path

    def get_cropped_images_path(self):
        """Returns the paths to the cropped images.

        Returns:
            A list of paths to the cropped images.
        """
        # Get all the images in the temp folder
        TEMP_IMAGES_PATH = []
        for filename in os.listdir(self.save_path):
            TEMP_IMAGES_PATH.append(self.save_path + '\\' + filename)
        return TEMP_IMAGES_PATH
    
    def delete_cropped_images(self):
        """Deletes the cropped images in the temp folder.
        """
        for filename in os.listdir(self.save_path):
            os.remove(self.save_path + '\\' + filename)

    def visualize_results(bounding_boxes_list, text_list, image):
        for bounding_box, text in zip(bounding_boxes_list, text_list):
            xmin, ymin, xmax, ymax = bounding_box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        pass