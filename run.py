from flask import Flask, render_template, request
from OCR.text_detection.Yolov8 import TextDetector
from OCR.text_recognition.VietOCR import TextRecognition
from PIL import ImageFont, ImageDraw, Image
import os

app = Flask(__name__)

# Create the text detector
text_detector = TextDetector(model_path='D:\\Capstone\\Code\\Web\\OCR\\text_detection\\weights\\YOLOv8\\train13\\weights\\best.pt', save_path='D:\\Capstone\\Code\\Web\\OCR\\text_detection\\temp')
# Create the text recognizer
text_recognizer = TextRecognition()

@app.route('/', methods=['GET', 'POST'])
def home():

    # Delete the result.jpeg file in the static/images folder
    if os.path.exists('D:\\Capstone\\Code\\Web\\static\\images\\result.jpg'):
        os.remove('D:\\Capstone\\Code\\Web\\static\\images\\result.jpg')
        print('Deleted result.jpeg')

    if request.method == 'POST':

        # If the user does not choose an image to predict
        if request.files['image_file'].filename == '':
            return render_template('upload_img.html', predicted_text='Please choose an image to predict!')
        
        image = request.files['image_file']
        
        # Detect text in the image
        bounding_boxes_list = text_detector.detect_text_in_image(image_path=image)

        # Non-maximum suppression
        bounding_boxes_list = text_detector.non_max_suppression_fast(boxes=bounding_boxes_list, overlapThresh=0.5)

        # Predict the text in the image
        predicted_text = text_recognizer.predict(image=image, bounding_boxes_list=bounding_boxes_list)

        # Visualize the results
        # Open the image file with PIL
        image = Image.open(image.stream)

        # Convert the image to RGB (OpenCV uses BGR)
        image = image.convert('RGB')

        # Initialize ImageDraw
        draw = ImageDraw.Draw(image)

        # Specify font .ttf file (you should have a .ttf font file that supports Vietnamese characters)
        font = ImageFont.truetype('D:\\Capstone\\Code\\Web\\static\\fonts\\BeVietnamPro-Black.ttf', 30)

        for bounding_box, text in zip(bounding_boxes_list, predicted_text):
            xmin, ymin, xmax, ymax = bounding_box
            draw.rectangle([xmin, ymin, xmax, ymax], outline ="green", width=2)
            draw.text((xmin, ymin), text, fill ="red", font=font)

        # Save the visualized image
        image.save('D:\\Capstone\\Code\\Web\\static\\images\\result.jpg')

        # Return the predicted text
        return render_template('upload_img.html', predicted_text=predicted_text, predicted_image='images/result.jpg')
    return render_template('upload_img.html', predicted_image='images/default.jpg')



if __name__ == '__main__':
    app.run(debug=True)
