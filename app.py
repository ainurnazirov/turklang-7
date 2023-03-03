from flask import Flask, render_template, request
from translate import Translator
from paths import models_dict

app = Flask(
    __name__,
    template_folder='templates',
    static_folder='static',
)

models = {}
for i in models_dict.keys():
    models.update({i: Translator(i[0], i[1])})

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        in_text = request.form['in_text']
        in_lang = request.form['in_lang']
        out_lang = request.form['out_lang']
        out_text = models[(in_lang, out_lang)].translate(in_text)
        return render_template("main.html", result=out_text)
    elif request.method == "GET":
        return render_template("main.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81)
