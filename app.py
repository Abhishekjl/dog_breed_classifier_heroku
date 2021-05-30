from flask import Flask, render_template, request
import os
app = Flask(__name__)
import pandas as pd
from prediction import predictor

app.config['IMAGE_FOLDER'] = r"static\images"

@app.route('/')
def hello_world():
    return render_template('index.html')



@app.route('/upload',methods = ["POST","GET"])

def upload():
    image = request.files['myfile']  
    image.save(os.path.join(app.config['IMAGE_FOLDER'],image.filename)) 
    prediction = predictor(image.filename)
    os.remove(os.path.join(app.config['IMAGE_FOLDER'],image.filename))
    return render_template('index.html', image = image.filename, data =  prediction )


if __name__ == "__main__":
    app.run(debug = False)    