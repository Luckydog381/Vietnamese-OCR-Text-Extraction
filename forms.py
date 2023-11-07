from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField, TextAreaField
from flask_wtf.file import FileAllowed


class UploadImage(FlaskForm):
    image_file = FileField(label='Upload Image', validators=[FileAllowed(['jpg', 'jpeg', 'png'])], render_kw={'accept': 'image/*', 'onchange': 'document.getElementById("profile-picture-preview").src = window.URL.createObjectURL(this.files[0])'})
    submit = SubmitField(label='Upload Image')