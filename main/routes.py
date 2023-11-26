from main import app, text_detector, text_recognizer
from flask import render_template
from flask import request
from PIL import Image, ImageDraw, ImageFont
import os

@app.route('/', methods=['GET', 'POST'])
def home():

    # Delete the result.jpeg file in the static/images folder
    if os.path.exists('D:\\Capstone\\Code\\Vietnamese-OCR-Text-Extraction\\main\\static\\images\\result.jpg'):
        os.remove('D:\\Capstone\\Code\\Vietnamese-OCR-Text-Extraction\\main\\static\\images\\result.jpg')

    if request.method == 'POST':

        # If the user does not choose an image to predict
        if request.files['image_file'].filename == '':
            return render_template('upload_img.html', predicted_text='Please choose an image to predict!')
        
        image = request.files['image_file']
        
        # Detect text in the image
        bounding_boxes_list = text_detector.detect_text_in_image(image_path=image)

        # Non-maximum suppression
        # bounding_boxes_list = text_detector.non_max_suppression_fast(boxes=bounding_boxes_list, overlapThresh=0.5)

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
        font = ImageFont.truetype('D:\\Capstone\\Code\\Vietnamese-OCR-Text-Extraction\\main\\static\\fonts\\BeVietnamPro-Black.ttf', 20)

        for bounding_box, text in zip(bounding_boxes_list, predicted_text):
            xmin, ymin, xmax, ymax = bounding_box
            draw.rectangle([xmin, ymin, xmax, ymax], outline ="green", width=2)
            draw.text((xmin, ymin), text, fill ="red", font=font)

        # Save the visualized image
        image.save('D:\\Capstone\\Code\\Vietnamese-OCR-Text-Extraction\\main\\static\\images\\result.jpg')

        # Return the predicted text
        return render_template('upload_img.html', predicted_text=predicted_text, predicted_image='images/result.jpg')
    return render_template('upload_img.html', predicted_image='images/default.jpg')

