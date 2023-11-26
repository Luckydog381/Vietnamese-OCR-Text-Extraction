from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

class TextRecognition:
    def __init__(self):
        self.config = Cfg.load_config_from_name('vgg_transformer')
        self.config['cnn']['pretrained']=False
        self.config['device'] = 'cuda:0'
        self.detector = Predictor(self.config)
    
    def predict(self, image, bounding_boxes_list):
        # images_path: list of image path
        # input : list of images path
        # output: list of predicted text

        predicted_text = []
        original_image = Image.open(image)
        
        for bounding_box in bounding_boxes_list:
            xmin, ymin, xmax, ymax = bounding_box
            cropped_image = original_image.crop((xmin, ymin, xmax, ymax))
            s = self.detector.predict(cropped_image)
            predicted_text.append(s)

        return predicted_text