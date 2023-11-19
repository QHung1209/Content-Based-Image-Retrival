from flask import Flask, render_template, request, session
import os
import base64
import cv2

app = Flask(__name__)
app.secret_key = "your_secret_key" 

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files['file']
    image_path = ".\\image_upload\\" + file.filename
    if (image_path == ".\\image_upload\\"):
        return render_template("index.html")
    file.save(image_path)
    session['image_path'] = image_path
    return render_template("index.html")
"""
    data = request.get_json()

    # Access the variables
    x = data.get('x')
    y = data.get('y')
    width = data.get('width')
    height = data.get('height')
    image_path = session.get('image_path', None)
    fileName = data.get('fileName')
    image_path = ".\\image_upload\\" + fileName
    print(x, height, fileName)

    img = cv2.imread(image_path)

    # Crop image to specified area using slicing
    crop_img = img[int(y):int(y+height), int(x):int(x+width)]
# Show image
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)
    return render_template("index.html")
"""

@app.route('/get_Data', methods=['POST'])
def get_Data():
    data = request.get_json()

    # Access the variables
    x = data.get('x')
    y = data.get('y')
    width = data.get('width')
    height = data.get('height')
    image_path = session.get('image_path', None)
    fileName = data.get('fileName')
    image_path = ".\\image_upload\\" + fileName
    print(x, height, fileName)

    img = cv2.imread(image_path)

    # Crop image 
    crop_img = img[int(y):int(y+height), int(x):int(x+width)]
# Show image
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)
    return render_template("index.html")

    
if __name__ == '__main__':
    app.run(debug=True)
