from PIL import Image,ImageOps
import numpy as np
import pandas as pd
import cv2
import re
import h5py

import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer
from sklearn import tree
from joblib import dump, load
from sklearn.linear_model import LogisticRegression

import method.expression

class Classification:

    @staticmethod
    def getLetter(letterIndex,letterChar):

        abd=pd.read_csv("/Users/priscafehiarisoadama/me/S4/mr_Tsinjo/11_tp_equation/data/A_Z Handwritten Data.csv")

        lettre=abd[abd['0']==letterIndex]
        lettre.drop(['0'], axis=1)
        lettre.to_csv(f"./{letterChar}.csv", index=False)

    @staticmethod
    def createDAtaCLassifierModel(img_dir):
        categories = [os.listdir(img_dir)[1], os.listdir(img_dir)[2]]
        print(categories)

        data = []
        labels = []

        for categories_index, category in enumerate(categories):
            for file in os.listdir(os.path.join(img_dir, category)):
                # maka an'le image any anaty fichiers
                img_path = os.path.join(img_dir, category, file)
                img = imread(img_path)
                img = resize(img, (28, 28))
                # pour obtenir une image en une seule ligne
                data.append(img.flatten())
                labels.append(category)
        data = np.asarray(data)
        labels = np.asarray(labels)

        # train the data and then set the test

        #     shuffle : manafangaro an;le data
        #     stratify :
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

        # train classifier
        classifier = SVC()

        # liste ana dictionnaire
        parametters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 18, 100, 1000]}]
        grid_search = GridSearchCV(classifier, parametters)
        grid_search.fit(x_train, y_train)
        best_estim = grid_search.best_estimator_
        pickle.dump(best_estim, open("./model.pkl", "wb"))
        y_pred = best_estim.predict(x_test)
        score = accuracy_score(y_pred, y_test)
        print(score)

    @staticmethod
    def classification(img_dir):

        # categories = [os.listdir(img_dir)[1], os.listdir(img_dir)[2],os.listdir(img_dir)[3],os.listdir(img_dir)[4],os.listdir(img_dir)[5],os.listdir(img_dir)[6],os.listdir(img_dir)[7],os.listdir(img_dir)[8],os.listdir(img_dir)[9],os.listdir(img_dir)[10]]
        categories = Classification.getCategory(img_dir)
        print(categories)

        data = []
        labels = []

        for categories_index, category in enumerate(categories):
            i=0
            for file in os.listdir(os.path.join(img_dir, category)):
                i=i+1
                print(category)
                # maka an'le image any anaty fichiers

                if(file!=".DS_Store" ):
                    img_path = os.path.join(img_dir, category, file)
                    img = imread(img_path)
                    img = resize(img, (28, 28))
                    print(img.shape)
                    if img.shape[-1] == 4:  # check if image has alpha channel
                        img = img[:, :, :1]
                        img = np.reshape(img, (28, 28))

                    # pour obtenir une image en une seule ligne
                    print(len(img.flatten()), category,i)
                    # if(len(img.flatten())==784):
                    data.append(img.flatten())
                    labels.append(category)
        data = np.asarray(data)
        labels = np.asarray(labels)
        print(f'labels: {labels}')

        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(x_train, y_train)

        # save the model
        dump(clf, "model_saved.joblib")

        y_pred = clf.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        score = accuracy_score(y_test, y_pred)
        print(score)

    @staticmethod
    def classification2(img_dir):
        image_list = []
        image_value = []
        categories = Classification.getCategory(img_dir)

        # mamaky an'le image
        for nombre in tqdm(os.listdir(img_dir)):
            if(nombre!=".DS_Store" and nombre!="data.csv"):
                doss = img_dir + nombre + '/'
                for nom_fichier in os.listdir(doss):
                    if nom_fichier.endswith(".png"):
                        # Charger l'image avec cv2 et l'ajouter à la liste
                        image = imread(os.path.join(doss, nom_fichier))
                        if image.shape[-1] == 4:  # check if image has alpha channel
                            image = image[:, :, :1]
                            image = np.reshape(image, (28, 28))
                        image_list.append(image)
                        image_value.append(nombre)

         # manoratra an'le images ao amina fichier h5py
        with h5py.File('images.h5', 'w') as f:
            f.create_dataset('images', data=image_list)
        image_height = 28

        # mamaky an'le fichier
        with h5py.File('images.h5', 'r') as f:
            images = f['images'][:]

        # Générer les étiquettes
        labels = np.array(image_value)

        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

        # Appliquer un aplatissement aux images pour qu'elles soient compatibles avec l'arbre de décision
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        # Créer un arbre de décision et l'entraîner sur les données d'entraînement
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        dump(clf, "model2_saved.joblib")

        # Évaluer les performances du modèle sur les données de test
        accuracy = clf.score(X_test, y_test)
        print('Accuracy:', accuracy)


    @staticmethod
    def randomforest(img_dir):
        image_list = []
        image_value = []
        categories = Classification.getCategory(img_dir)

        # mamaky an'le image
        for nombre in tqdm(os.listdir(img_dir)):
            if (nombre != ".DS_Store" and nombre != "data.csv"):
                doss = img_dir + nombre + '/'
                for nom_fichier in os.listdir(doss):
                    if nom_fichier.endswith(".png"):
                        image = imread(os.path.join(doss, nom_fichier))
                        if image.shape[-1] == 4:  # check if image has alpha channel
                            image = image[:, :, :1]
                            image = np.reshape(image, (28, 28))
                        image_list.append(image)
                        image_value.append(nombre)

        # manoratra an'le images ao amina fichier h5py
        with h5py.File('images.h5', 'w') as f:
            f.create_dataset('images', data=image_list)
        image_height = 28

        # mamaky an'le fichier
        with h5py.File('images.h5', 'r') as f:
            images = f['images'][:]

        # Générer les étiquettes
        labels = np.array(image_value)

        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

        # Appliquer un aplatissement aux images pour qu'elles soient compatibles avec l'arbre de décision
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        random_forest = RandomForestClassifier()

        parameters = {'n_estimators': [3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45], 'criterion': ['entropy', 'gini'],
                      'max_depth': [2, 3, 5, 10]}
        acc_scorer = make_scorer(accuracy_score)
        grid_obj = GridSearchCV(random_forest, parameters, scoring=acc_scorer, cv=3)
        grid_obj = grid_obj.fit(X_train, y_train)
        clf = grid_obj.best_estimator_
        print(grid_obj.best_estimator_)
        # Fit the best algorithm to the data.
        clf.fit(X_train, y_train)
        dump(clf, "model3_saved.joblib")
        y_pred = clf.predict(X_test)
        print(clf.score(X_train, y_train))
        print(clf.score(X_test, y_test))



    @staticmethod
    def randomforest2(img_dir, y_pred=None):
        image_list = []
        image_value = []
        categories = Classification.getCategory(img_dir)

        # mamaky an'le image
        for nombre in tqdm(os.listdir(img_dir)):
            if (nombre != ".DS_Store" and nombre != "data.csv"):
                doss = img_dir + nombre + '/'
                for nom_fichier in os.listdir(doss):
                    if nom_fichier.endswith(".png"):
                        image = imread(os.path.join(doss, nom_fichier))
                        if image.shape[-1] == 4:  # check if image has alpha channel
                            image = image[:, :, :1]
                            image = np.reshape(image, (28, 28))
                        image_list.append(image)
                        image_value.append(nombre)

        # manoratra an'le images ao amina fichier h5py
        with h5py.File('images.h5', 'w') as f:
            f.create_dataset('images', data=image_list)
        image_height = 28

        # mamaky an'le fichier
        with h5py.File('images.h5', 'r') as f:
            images = f['images'][:]

        # Générer les étiquettes
        labels = np.array(image_value)
        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=0)

        # Appliquer un aplatissement aux images pour qu'elles soient compatibles avec l'arbre de décision
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        clf = RandomForestClassifier(random_state=0)
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(accuracy_score(y_test, y_pred))

    @staticmethod
    def getCategory(img_dir):
        category = []
        for i in os.listdir(img_dir):
            if i != ".DS_Store" and i != 'data.csv':
                category.append(i)

        return category

    @staticmethod
    def check_image(image_path, model):
    
        # try:
        img = Image.open(image_path)

            # Redimensionner l'image en 28x28 pixels
        img = img.resize((28, 28))

        plt.imshow(img)
        plt.show()
            # Convertir l'image en une seule ligne de pixels
        img_pixels = np.asarray(img).flatten()
        # except:
        #     img = imread(image_path)
        #     img = resize(img, (28, 28))
        #     print(img.shape)
        #     if img.shape[-1] == 4:  # check if image has alpha channel
        #         img = img[:, :, :1]

            # Prédire la classe de l'image
        class_prediction = model.predict([img_pixels])[0]
        if class_prediction == 0:
            print("L'image représente un chiffre 9.")
        elif class_prediction == 1:
            print("L'image représente un chiffre 0.")
        else:
            print("La classe prédite n'est pas reconnue.")
        print(class_prediction)

    # methode de check qui marche
    @staticmethod
    def checkImage2(image_path,model):
        imgage=os.path.join(image_path)
        img=imread(imgage)
        if img.shape[-1] >= 2:  # check if image has alpha channel
            img = img[:, :, :1]
            img=np.reshape(img,(28,28))

        print(img.shape)
        plt.imshow(img)
        plt.show()
        img_pixels = np.asarray(img.flatten())
        class_prediction = model.predict([img_pixels])[0]
        print(class_prediction)
        return class_prediction

    # methode de check sans la path de l'image
    @staticmethod
    def checkImageWithoutPath(image,model):
        img=image
        # if img.shape[-1] >= 2:  # check if image has alpha channel
        #     img = img[:, :, :1]
        #     img=np.reshape(img,(28,28))

        print(img.shape)
        plt.imshow(img)
        plt.show()
        img_pixels = np.asarray(img.flatten())
        class_prediction = model.predict([img_pixels])[0]
        print(class_prediction)
        return class_prediction

    @staticmethod
    def showImageFromCsv(imgindex):
        df=pd.read_csv("A.csv")

        # Get the pixel values as a NumPy array
        normalized_pixels = df.iloc[imgindex].values

        normalized_pixels= Classification.inverseImageColor(normalized_pixels[1:])
        print(len(normalized_pixels))
        print(normalized_pixels)
        # Calculate the number of pixels in the image
        num_pixels = len(normalized_pixels)

        # Calculate the width and height of the image
        width = int(np.sqrt(num_pixels))
        height = int(num_pixels / width)
        denormalized_image = normalized_pixels.reshape((height, width))

        # Display the image using matplotlib
        import matplotlib.pyplot as plt
        plt.imshow(denormalized_image, cmap='gray')
        plt.show()
        # plt.imsave("./test.png", denormalized_image, cmap='gray')

    @staticmethod
    def transformAllImagesToDirectory(img_dir,char):
        df = pd.read_csv(f"{char}.csv")
        print(df.head())
        df=df.drop(['0'],axis=1)
        for i in range(len(df)):
            # Get the pixel values as a NumPy array
            normalized_pixels = df.iloc[i].values

            # normalized_pixels = Classification.inverseImageColor(normalized_pixels[1:])
            print(len(normalized_pixels))
            print(normalized_pixels)
            # Calculate the number of pixels in the image
            num_pixels = len(normalized_pixels)

            # Calculate the width and height of the image
            width = int(np.sqrt(num_pixels))
            height = int(num_pixels / width)
            denormalized_image = normalized_pixels.reshape((28, 28))
            denormalized_image = Classification.inverseImageColor(denormalized_image)

            # Display the image using matplotlib
            import matplotlib.pyplot as plt
            plt.imshow(denormalized_image, cmap='gray')
            plt.show()
            directory=img_dir+char
            # Create the directory if it doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory)
            print(directory)
            if denormalized_image.shape[-1] ==4 :  # check if image has alpha channel
                denormalized_image = denormalized_image[:, :, :1]

            plt.imsave(f"{directory}/{i}{char}.png", denormalized_image, cmap='gray')

    @staticmethod
    def inverseImageColor(letterLine):
        for i in range(len(letterLine)):
            letterLine[i]=255-letterLine[i]
        return letterLine

    @staticmethod
    def printimg(img):
        for i in img:
            for j in i:
                print(j[0], end="\t")
            print()
    # traintement misy randomforest model numero 3
    @staticmethod
    def treatImage(img_path,model_path):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        inverse = cv2.bitwise_not(image) #inverse la couleur de l'image mba ho blanc sur noir
    #maka an'le modele entregistré
        model=load(os.path.join(model_path,'model5_saved.joblib'))

    # string misy ny soratra teo @ le sary
        img_string=""

    #parcours tous les contours presents sur l'image
        contours, _ = cv2.findContours(inverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#   creer une copie de l'image originale
        output = image.copy()
        espace = 5 # espace autours de l'image aapres traitements

        for contour in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
    # Obtenir les coordonnées du rectangle englobant le chiffre
            x, y, w, h = cv2.boundingRect(contour)
            x -= espace
            y -= espace
            w += 2 * espace
            h += 2 * espace

    # Dessiner un rectangle autour du chiffre sur l'image de sortie
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extraire les contours  en tant qu'image individuelle
            digit = image[max(0, y):y + h, max(0, x):x + w]

    #image finale
            resized_digit = cv2.resize(digit, (28, 28))
            print(resized_digit)
    #traitement avec le modele
            mydigit=Classification.checkImageWithoutPath(resized_digit,model)
            if(mydigit=='K'):
                mydigit='>'
            elif(mydigit=='M'):
                mydigit='<'
            img_string+=mydigit
        return img_string

    @staticmethod
    def resoudre(equation):
        return method.expression.resolve(equation)

    @staticmethod
    def createCourbe(equation):
        operateur = method.expression.getSigneOp(equation)
        t = equation.split(operateur)
        x = np.linspace(-10, 10, 100)
        y = 5 * x + 8
        y2 = 51 * x - 6
        plt.plot(x, y, label=y)
        plt.plot(x, y2, label=y2)
        graph_file=""
        if (operateur == "<"):
            plt.fill_between(x, y, y2, where=(y < y2), color='gray', alpha=0.3)
        elif (operateur == ">"):
            plt.fill_between(x, y, y2, where=(y > y2), color='gray', alpha=0.3)
            # Ajouter des labels aux axes
            plt.xlabel('x')
            plt.ylabel('y')

            # Ajouter une légende
            plt.legend()

            # Enregistrer la courbe dans un fichier temporaire
            graph_file = '/Users/priscafehiarisoadama/me/S4/optimisation/TraitementImages/static/courbe.png'  # Choisissez le chemin approprié dans votre projet Flask
            plt.savefig(graph_file)
        return operateur