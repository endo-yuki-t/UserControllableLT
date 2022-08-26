from flask import Flask, render_template, request, redirect, url_for, abort
import json

app = Flask(__name__)

import sys
sys.path.append(".")
sys.path.append("..")

import argparse
from PIL import Image, ImageOps
import numpy as np
import base64
import cv2
from inference import demo

def Base64ToNdarry(img_base64):
    img_data = base64.b64decode(img_base64)
    img_np = np.fromstring(img_data, np.uint8)
    src = cv2.imdecode(img_np, cv2.IMREAD_ANYCOLOR)

    return src

def NdarrayToBase64(dst):
    result, dst_data = cv2.imencode('.png', dst)
    dst_base64 = base64.b64encode(dst_data)

    return dst_base64

parser = argparse.ArgumentParser(description='User controllable latent transformer')
parser.add_argument('--checkpoint_path', default='pretrained_models/latent_transformer/cat.pt')
args = parser.parse_args()

demo = demo(args.checkpoint_path)

@app.route("/", methods=["GET", "POST"])
#@auth.login_required
def init():
    if request.method == "GET":
        input_img = demo.run()
        input_base64 = "data:image/png;base64,"+NdarrayToBase64(input_img).decode()
        return render_template("index.html", filepath1=input_base64, canvas_img=input_base64, result=True)
    if request.method == "POST":
        if 'zi' in request.form.keys():
            input_img = demo.move(z=-0.05)
        elif 'zo' in request.form.keys():
            input_img = demo.move(z=0.05)
        elif 'u' in request.form.keys():
            input_img = demo.move(y=-0.5, z=-0.0)
        elif 'd' in request.form.keys():
            input_img = demo.move(y=0.5, z=-0.0)
        elif 'l' in request.form.keys():
            input_img = demo.move(x=-0.5, z=-0.0)
        elif 'r' in request.form.keys():
            input_img = demo.move(x=0.5, z=-0.0)
        else:
            input_img = demo.run()
        
        input_base64 = "data:image/png;base64,"+NdarrayToBase64(input_img).decode()
        return render_template("index.html", filepath1=input_base64, canvas_img=input_base64, result=True)

@app.route('/zoom', methods=["POST"])
def zoom_func():
    
    dz = json.loads(request.form['dz'])
    sx = json.loads(request.form['sx'])
    sy = json.loads(request.form['sy'])
    stop_points = json.loads(request.form['stop_points'])
    
    input_img = demo.zoom(dz,sxsy=[sx,sy],stop_points=stop_points)
    input_base64 = "data:image/png;base64,"+NdarrayToBase64(input_img).decode()
    res = {'img':input_base64}
    return json.dumps(res)

@app.route('/translate', methods=["POST"])
def translate_func():
    
    dx = json.loads(request.form['dx'])
    dy = json.loads(request.form['dy'])
    dz = json.loads(request.form['dz'])
    sx = json.loads(request.form['sx'])
    sy = json.loads(request.form['sy'])
    stop_points = json.loads(request.form['stop_points'])
    zi = json.loads(request.form['zi'])
    zo = json.loads(request.form['zo'])

    input_img = demo.translate([dx,dy],sxsy=[sx,sy],stop_points=stop_points,zoom_in=zi,zoom_out=zo)
    input_base64 = "data:image/png;base64,"+NdarrayToBase64(input_img).decode()
    res = {'img':input_base64}
    return json.dumps(res)

@app.route('/changestyle', methods=["POST"])
def changestyle_func():
    input_img = demo.change_style()
    input_base64 = "data:image/png;base64,"+NdarrayToBase64(input_img).decode()
    res = {'img':input_base64}
    return json.dumps(res)

@app.route('/reset', methods=["POST"])
def reset_func():
    input_img = demo.reset()
    input_base64 = "data:image/png;base64,"+NdarrayToBase64(input_img).decode()
    res = {'img':input_base64}
    return json.dumps(res)
    
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8000)