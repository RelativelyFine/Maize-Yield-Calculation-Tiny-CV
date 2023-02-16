## Maize Tassel Detector Backend

This is a flask backend with a custom model. Results are returned as a count of the maize tassel.

## Getting Started

- Ensure you cd into /Backend:
- Create a venv with python version 3.9.13
- Ensure you have a cuda enabled GPU
- Install CUDA 11.7
- Run the following

```bash
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt

python ./api.py
```

Endpoints are hosted at [http://localhost:5000](http://localhost:5000)

There is a single post endpoint /upload, that takes a series of files and returns a string of the total count.

Prediction images are saved in `/runs`

### Note:

Currently Using YoloV8n on a custom trained tensor for parking detection. This will be changed to the proper model in the future.
