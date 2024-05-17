# api.py
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import cv2
import numpy as np
from flask_cors import CORS
from flask import Flask, send_file, make_response
import io
import insightface
import base64
from insightface.app import FaceAnalysis

swapper = insightface.model_zoo.get_model('./models/inswapper_128.onnx',
                                download=False,
                                download_zip=False)

app = Flask(__name__)
CORS(app)

def swap_n_show(image1, image2, app, swapper,
                plot_before=True, plot_after=True):
    
    face1 = app.get(image1)[0]
    face2 = app.get(image2)[0]
    image1_ = swapper.get(image1, face1, face2, paste_back=True)
    image2_ = swapper.get(image2, face2, face1, paste_back=True)
        
    return image1_, image2_

@app.route('/process_image', methods=['POST'])
def process_image():
    # Check if the request contains files
    if 'src_image' not in request.files or 'tar_image' not in request.files:
        return jsonify({'error': 'No files found'})

    src_image_file = request.files['src_image']
    tar_image_file = request.files['tar_image']

    # Read image files using OpenCV
    src_img = cv2.imdecode(np.frombuffer(src_image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    tar_img = cv2.imdecode(np.frombuffer(tar_image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    img1, img2 = swap_n_show(src_img, tar_img, app, swapper)

    image_data1 = cv2.imencode('.jpg', img1)[1].tostring()
    image_data2 = cv2.imencode('.jpg', img2)[1].tostring()

    image_data1 = base64.b64encode(cv2.imencode('.jpg', img1)[1]).decode('utf-8')
    image_data2 = base64.b64encode(cv2.imencode('.jpg', img2)[1]).decode('utf-8')

    return jsonify({
        'image1': image_data1,
        'image2': image_data2
    })

if __name__ == '__main__':
    app.run(debug=True)
