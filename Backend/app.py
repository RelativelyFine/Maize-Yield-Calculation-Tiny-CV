from flask import Flask, request, send_file
from flask_cors import CORS
import os
from threading import Lock
import datetime
from Model import estimate, corn_doc
import io
from base64 import encodebytes
from PIL import Image
from flask import jsonify

lock = Lock()

app = Flask(__name__)
CORS(app)

def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

@app.route('/upload', methods=['POST'])
def upload_file():

    lock.acquire()

    now = datetime.datetime.now()

    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    os.mkdir(os.path.join('uploads', current_time))

    for file in request.files.getlist('file'):

        # save files to new folder
        file.save(os.path.join('uploads', current_time, file.filename))
    
    print(os.listdir(os.path.join('uploads', current_time)))

    img_arr = []
    for i in [(os.path.join('uploads', current_time, i)) for i in os.listdir(os.path.join('uploads', current_time))]:
        img_arr.append(estimate.read_image(i))

    results, time, heatmap_time = estimate.RunOnImages(img_arr)
    # disease = corn_doc.RunOnImages(img_arr)

    # if disease:
    #     print(results, disease)
    
    # total = sum(results) - sum(disease) if disease else sum(results)
    
    total = sum(results)

    # get all images paths from Heatmaps folder
    images = [os.path.join('Heatmaps', i) for i in os.listdir('Heatmaps')]
    encoded_imges = []
    for image_path in images:
        encoded_imges.append(get_response_image(image_path))

    lock.release()

    return jsonify({'heatmaps': encoded_imges, 'total': str(int(total)), 'time': str(time), 'heatmap_time': str(heatmap_time)})

if __name__ == "__main__":
    app.run(debug=True)
