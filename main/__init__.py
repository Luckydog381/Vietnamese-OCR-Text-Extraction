from flask import Flask
from main.OCR.text_detection.Yolov8 import TextDetector
from main.OCR.text_recognition.VietOCR import TextRecognition

app = Flask(__name__)

# Create the text detector
text_detector = TextDetector(model_path='D:\\Capstone\\Code\\Vietnamese-OCR-Text-Extraction\\main\\OCR\\text_detection\\weights\\YOLOv8\\train13\\weights\\best.pt', save_path='None')
# Create the text recognizer
text_recognizer = TextRecognition()

from main import routes