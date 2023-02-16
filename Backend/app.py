from flask import Flask, request, send_file
from flask_cors import CORS
from ultralytics import YOLO
import os
from threading import Lock
import datetime

model = YOLO('top-down-park.pt')  # load a pretrained YOLOv8n classification model
lock = Lock()

app = Flask(__name__)
CORS(app)


@app.route('/upload', methods=['POST'])
def upload_file():

    lock.acquire()

    # get current time
    now = datetime.datetime.now()

    # format time
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    # create folder
    os.mkdir(os.path.join('uploads', current_time))

    for file in request.files.getlist('file'):

        # save files to new folder
        file.save(os.path.join('uploads', current_time, file.filename))
        
    # Run YoloV8n on images
    results = model.predict(source = os.path.join('uploads', current_time), save = True)


    total = sum([len(result.boxes) for result in results])

    lock.release()

    return str(total)


if __name__ == "__main__":
    app.run(debug=True)
