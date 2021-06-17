import os
import io
import uuid
import sys
from werkzeug.utils import secure_filename
import yaml
import traceback
import jsonpickle

# read config file
with open('./config.yaml', 'r') as fd:
    opts = yaml.safe_load(fd)

# other packages
sys.path.insert(0, './white_box_cartoonizer/')
sys.path.insert(1, './bgremoval/')

import cv2
from flask import Flask, request, Response, send_file, after_this_request
import flask
from PIL import Image
import numpy as np

from cartoonize import WB_Cartoonize
from seg import run_visualization

# flask app instance
app = Flask(__name__)
# destination location
app.config['CARTOONIZED_FOLDER'] = 'static/cartoonized_images'
# config options
app.config['OPTS'] = opts

IMAGE_UPLOAD_FOLDER = 'static/uploadedImages'
app.config['UPLOAD_FOLDER'] = IMAGE_UPLOAD_FOLDER

## Init Cartoonizer and load its weights with modal
wb_cartoonizer = WB_Cartoonize(os.path.abspath("white_box_cartoonizer/saved_models/"), opts['gpu'])

# home api
@app.route('/', methods=["POST", "GET"])
def cartoonize():
    if flask.request.method == 'POST':
        try:
            if flask.request.files.get('image'):
                # read image and secure filename
                img = request.files["image"]

                # generate unique image name
                img_name = str(uuid.uuid4())
                
                # temp save image
                image_location = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
                img.save(image_location)

                # bg_remove config = from API request, default false
                bg_remove = request.form['bg_remove'] == 'True' or False
                
                # convert image - cartoonize
                cartoon_image = wb_cartoonizer.infer(image_location)

                # image extenstion
                img_ext = '.png' if bg_remove else '.jpg'
                
                cartoonized_img_name = os.path.join(app.config['CARTOONIZED_FOLDER'], img_name + img_ext)
                cv2.imwrite(cartoonized_img_name, cartoon_image)

                # remove background image1
                if bg_remove:
                    run_visualization(cartoonized_img_name, cartoonized_img_name)

                 # remove file
                @after_this_request
                def remove_file(response):
                    try:
                        os.remove(cartoonized_img_name)
                        os.remove(image_location)
                    except Exception as error:
                        app.logger.error("Error removing or closing downloaded file handle", error)
                    return response

                # send response
                return send_file(cartoonized_img_name, mimetype='image/PNG')

        except Exception:
            # print error log
            print(traceback.print_exc())
            # build a response dict to send back to client
        
        response = {'message': 'Something wrong, Please try later'}
        # encode response using jsonpickle
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=500, mimetype="application/json")

    else:
        # build a response dict to send back to client
        response = {'message': 'Invalid Request'}
        # encode response using jsonpickle
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=404, mimetype="application/json")

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))