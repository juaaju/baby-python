from flask import Flask, render_template
from flaskwebgui import FlaskUI
from sam_satu_kamera import all_params

app = Flask(__name__)
ui = FlaskUI(app=app, server="flask", width=800, height=600)

@app.route("/")
def hello():  
    result = all_params("baby5-up.jpeg")
    return render_template('index.html', result=result)

if __name__ == "__main__":
    # app.run() for debug
    ui.run()