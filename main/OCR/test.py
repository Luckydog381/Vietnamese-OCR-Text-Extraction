from text_detection.Yolov8 import TextDetector
from text_recognition.VietOCR import TextRecognition

IMAGES_PATH = ['D:\\Capstone\\Dataset\\Dataset\\4 - PICK (Final)\\preprocessed_images\\mcocr_public_145013aaprl.jpg',
               'D:\\Capstone\\Dataset\\Dataset\\4 - PICK (Final)\\preprocessed_images\\mcocr_public_145013acjke.jpg',
               'D:\\Capstone\\Dataset\\Dataset\\4 - PICK (Final)\\preprocessed_images\\mcocr_public_145013alpyf.jpg',]

IMAGE_PATH = 'D:\\Capstone\\Dataset\\Dataset\\4 - PICK (Final)\\preprocessed_images\\mcocr_public_145013aaprl.jpg'

print('Loading text detection module...')
# Detect text in the images.
yolo_text_detector = TextDetector(model_path='D:\\Capstone\\Code\\Web\\OCR\\text_detection\\weights\\YOLOv8\\train13\\weights\\best.pt', save_path='D:\\Capstone\\Code\\Web\\OCR\\text_detection\\temp')
yolo_text_detector.detect_text_in_image(IMAGE_PATH)

# Retrieve the cropeed images path
TEMP_PATH = yolo_text_detector.get_cropped_images_path()

print('Loading text recognition module...')
# Recognize text in the images.
vietocr_text_recognition = TextRecognition()
predicted_text = vietocr_text_recognition.predict(TEMP_PATH)

print('Predicted text:')
print(predicted_text)

