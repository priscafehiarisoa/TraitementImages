import os
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, redirect, url_for, request, flash
from werkzeug.utils import secure_filename

from method.Classification import Classification

UPLOAD_FOLDER = '/Users/priscafehiarisoadama/me/S4/optimisation/TraitementImages/static/upload/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)


@app.route('/')
def hello_world():  # put application's code here
    return upload()

@app.route('/result')
def displayResult():
    return render_template("displayResults.html")

@app.route('/upload')
def upload():
    return  render_template("pictureForm.html")

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/treatimg',methods=['GET','POST'])
def treatimage():
    files=""
    print("srep0")

    if request.method == 'POST':
        # check if the post request has the file part
        print("srep1")

        if 'file' not in request.files:
            flash('No file part')
            print("srep2")
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            print("srep3")

            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

   # ---------------------------
            files=filename
            print("srep4")
            imgres=checkImg(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            imgres=imgres.replace("%","=")
            res=Classification.resoudre(imgres)
            # res=""
            print(imgres)


                # Enregistrer la courbe dans un fichier temporaire
            graph_file = '/Users/priscafehiarisoadama/me/S4/optimisation/TraitementImages/static/courbe.png'  # Choisissez le chemin approprié dans votre projet Flask
                # plt.savefig(graph_file)
            return render_template("displayResults.html", result=imgres, image=filename , valiny=res,graph_file=graph_file)

@app.route('/courbe')
def afficher_courbe():
        # Définition des limites de l'axe x
        x = np.linspace(-10, 10, 100)

        # Calcul des valeurs correspondantes de y pour chaque x
        y = 5 * x + 8
        y2 = 51 * x - 6

        # Tracer les courbes
        plt.plot(x, y, label='5 * x + 8')
        plt.plot(x, y2, label='51 * x - 6')
        plt.fill_between(x, y, y2, where=(y < y2), color='gray', alpha=0.3)

        # Ajouter des labels aux axes
        plt.xlabel('x')
        plt.ylabel('y')

        # Ajouter une légende
        plt.legend()

        # Enregistrer la courbe dans un fichier temporaire
        graph_file = '/Users/priscafehiarisoadama/me/S4/optimisation/TraitementImages/static/courbe.png'  # Choisissez le chemin approprié dans votre projet Flask
        plt.savefig(graph_file)

        # Rendre le template HTML et passer le nom du fichier de la courbe
        return render_template('courbe.html', graph_file=graph_file)

    # return "uploaded"\

def checkImg(img_path):
    model_path="/Users/priscafehiarisoadama/me/S4/mr_Tsinjo/AnalyseDeDonnes/analyse-Sary/"
    result=Classification.treatImage(img_path,model_path)
    return result

if __name__ == '__main__':
    app.run()

