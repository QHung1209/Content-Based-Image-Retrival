from flask import Flask, render_template, request, session,url_for
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import pickle
import numpy as np
from cv2 import resize,cvtColor,COLOR_BGR2RGB,imread
 
app = Flask(__name__)
app.secret_key = "your_secret_key" 
app.config['OUTPUT_FOLDER'] = 'dataset2/' 
app.config['UPLOAD_DIRECTORY'] = 'input_dir/'
app.config['ALLOWED_EXTENSIONS'] = ['.png']

vectors = pickle.load(open(".\\vectors.pkl", "rb"))

paths = pickle.load(open(".\\paths.pkl", "rb"))

def get_extract_feature():
    vgg16_model = VGG16(weights='imagenet')
    extract_model = Model(inputs=vgg16_model.inputs,
                          outputs=vgg16_model.get_layer("fc1").output)
    return extract_model


def image_preprocess(img):
    img = resize(img, (224, 224))
    img = cvtColor(img, COLOR_BGR2RGB)  
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def extract_vector(model, img):
    print("Xy ly: ")
    img_tensor = image_preprocess(img)

    vector = model.predict(img_tensor)[0]
    vector = vector / np.linalg.norm(vector)
    return vector


@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files['file']
    image_path = ".\\image_upload\\" + file.filename
    if (image_path == ".\\image_upload\\"):
        return render_template("index.html")
    file.save(image_path)
    session['image_path'] = image_path
    return render_template("index.html")

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
    img = imread(image_path)
    # Crop image 
    crop_img = img[int(y):int(y+height), int(x):int(x+width)]

    model = get_extract_feature()

    search_vector = extract_vector(model, crop_img)

    distance = [np.linalg.norm(vector - search_vector) for vector in vectors]

    ids = np.argsort(distance)[:5]

    url = [paths[id] for id in ids]


    return render_template("index.html", image_urls=url)

@app.route("/")
def home():
    return render_template("index.html")
    
if __name__ == '__main__':
    app.run(debug=True)
