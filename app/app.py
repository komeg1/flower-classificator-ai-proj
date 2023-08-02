import os
import tkinter as tk
from tkinter import ttk

import customtkinter
import cv2
from keras.optimizers import Adam

from app_parameters import *
from PIL import Image, ImageTk
import customtkinter as ctk
import numpy as np
customtkinter.set_appearance_mode('light')
fileName = ''
fullPredictions = []
CHOSEN_MODEL = Model.REFERENCE
CLASSES_5 = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
CLASSES_16 = ['astlibe', 'bellflower', 'black_eyed_susan', 'calendula','california_poppy','carnation','commmon_daisy','coreopsis','daffodil','dandelion','iris','magnolia','rose','sunflower','tulip','water_lily']
CLASSES_7 = ['bellflower','daisy','dandelion','lotus','rose','sunflower','tulip']
modelFile = load_model('../models/ref_model_16k-0.90.h5')

CHOSEN_DATASET = Dataset.FLOWERS_16
def setModelFile():
    global modelFile
    global CHOSEN_MODEL
    global CHOSEN_DATASET
    if CHOSEN_MODEL == Model.REFERENCE and CHOSEN_DATASET == Dataset.FLOWERS_5:
        modelFile = load_model('../models/ref_model_5k-0.89.h5')
    elif CHOSEN_MODEL == Model.REFERENCE and CHOSEN_DATASET == Dataset.FLOWERS_16:
        modelFile = load_model('../models/ref_model_16k-0.90.h5')
    elif CHOSEN_MODEL == Model.REFERENCE and CHOSEN_DATASET == Dataset.FLOWERS_7:
        modelFile = load_model('../models/ref_model_7k-0.92.h5')

    elif CHOSEN_MODEL == Model.CNN and CHOSEN_DATASET == Dataset.FLOWERS_5:
        modelFile = load_model('../models/cnn_model_5k.h5')
    elif CHOSEN_MODEL == Model.CNN and CHOSEN_DATASET == Dataset.FLOWERS_16:
        modelFile = load_model('../models/cnn_model_16k.h5')
    elif CHOSEN_MODEL == Model.CNN and CHOSEN_DATASET == Dataset.FLOWERS_7:
        modelFile = load_model('../models/cnn_model_7k.h5')

    elif CHOSEN_MODEL == Model.DISTILLED and CHOSEN_DATASET == Dataset.FLOWERS_5:
        modelFile = load_model('../models/student_model_5k')
        modelFile.compile(optimizer=Adam(learning_rate=0.001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    elif CHOSEN_MODEL == Model.DISTILLED and CHOSEN_DATASET == Dataset.FLOWERS_16:
        modelFile = load_model('../models/student_model_16k')
        modelFile.compile(optimizer=Adam(learning_rate=0.001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    elif CHOSEN_MODEL == Model.DISTILLED and CHOSEN_DATASET == Dataset.FLOWERS_7:
        modelFile = load_model('../models/student_model_7k')
        modelFile.compile(optimizer=Adam(learning_rate=0.001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

    elif CHOSEN_MODEL == Model.PRUNED and CHOSEN_DATASET == Dataset.FLOWERS_5:
        modelFile = load_model('../models/sparse_model_5k.h5')
    elif CHOSEN_MODEL == Model.PRUNED and CHOSEN_DATASET == Dataset.FLOWERS_16:
        modelFile = load_model('../models/sparse_model_16k.h5')
    elif CHOSEN_MODEL == Model.PRUNED and CHOSEN_DATASET == Dataset.FLOWERS_7:
        modelFile = load_model('../models/sparse_model_7k.h5')

    elif CHOSEN_MODEL == Model.NO_DISTILLATION and CHOSEN_DATASET == Dataset.FLOWERS_5:
        modelFile = load_model('../models/student_without_distiller_5k.h5')
    elif CHOSEN_MODEL == Model.NO_DISTILLATION and CHOSEN_DATASET == Dataset.FLOWERS_16:
        modelFile = load_model('../models/student_without_distiller_16k.h5')
    elif CHOSEN_MODEL == Model.NO_DISTILLATION and CHOSEN_DATASET == Dataset.FLOWERS_7:
        modelFile = load_model('../models/student_without_distiller_7k.h5')
def load_image():

    setModelFile()
    filename = tk.filedialog.askopenfilename(initialdir="/",
                                          title="Select a File")
    #load image from filename
    img = Image.open(filename)
    img = img.resize((300, 300))
    prepareImageToCheck(filename)
    img = ImageTk.PhotoImage(img)
    # delete previous image
    for widget in imageFrame.winfo_children():
        widget.destroy()

    panel = tk.Label(imageFrame, image=img)

    panel.image = img
    panel.pack()


def prepareImageToCheck(filename):

    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    if CHOSEN_MODEL == Model.REFERENCE:
     img = cv2.resize(img, (300, 300))
    else:
        img = cv2.resize(img, (150, 150))

    print(CHOSEN_DATASET)
    prediction = modelFile.predict(np.array([img])/255)

    #add prediction text to the window
    dataset=[]
    if CHOSEN_DATASET == Dataset.FLOWERS_16:
        dataset = CLASSES_16
    elif CHOSEN_DATASET== Dataset.FLOWERS_5:
        dataset = CLASSES_5
    elif CHOSEN_DATASET== Dataset.FLOWERS_7:
        dataset = CLASSES_7
    print(dataset)
    print(CHOSEN_DATASET)
    print(CHOSEN_MODEL)

    predictedText = ctk.CTkLabel(window, text=dataset[np.argmax(prediction)], font=('Lato', 15))
    predictedText.place(x=130, y=350)

    #add text that lists the probabilities of each class
    probabilitiesText = ctk.CTkLabel(window, text='Probabilities:', font=('Lato', 15))
    probabilitiesText.place(x=400, y=150)

    winner,text = setPredictions(CHOSEN_DATASET,prediction)

    for windowLabel in window.winfo_children():
        if isinstance(windowLabel, ctk.CTkLabel) and windowLabel.winfo_y() > 150 and windowLabel.cget('text') != 'Predicted flower:':
            windowLabel.destroy()


    winnerProbability = ctk.CTkLabel(window, text=winner, font=('Lato', 15,'bold'))
    winnerProbability.place(x=400, y=170)

    probabilities = ctk.CTkLabel(window, text=text, font=('Lato', 15))
    probabilities.place(x=400, y=190)



def setPredictions(dataset, prediction):

    chosenClasses = []
    if dataset == Dataset.FLOWERS_16:
        chosenClasses = CLASSES_16
    elif dataset == Dataset.FLOWERS_5:
        chosenClasses = CLASSES_5
    elif dataset == Dataset.FLOWERS_7:
        chosenClasses = CLASSES_7

    print(chosenClasses)
    predictions = {}
    for i in range(len(chosenClasses)):
        predictions[chosenClasses[i]] = prediction[0][i]


    sortedPredictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    global fullPredictions
    fullPredictions = sortedPredictions
    #create string with predictions
    winner = str(sortedPredictions[0][0]) + ': ' + str(round(sortedPredictions[0][1]*100, 2)) + '%'
    text = ''
    for i in range(1, 3):
        text = text + str(sortedPredictions[i][0]) + ': ' + str(round(sortedPredictions[i][1]*100, 2)) + '%\n'

    return winner,text


def setDataset(event):
    global CHOSEN_DATASET
    if datasetChoice.get() == 'Flowers 16 classes':
        CHOSEN_DATASET = Dataset.FLOWERS_16
    elif datasetChoice.get() == 'Flowers 5 classes':
        CHOSEN_DATASET = Dataset.FLOWERS_5
    elif datasetChoice.get() == 'Flowers 7 classes':
        CHOSEN_DATASET = Dataset.FLOWERS_7
def chosenModel(event):
    global CHOSEN_MODEL
    if modelChoice.get() == 'Reference model':
        CHOSEN_MODEL = Model.REFERENCE
    elif modelChoice.get() == 'Student without distillation model':
        CHOSEN_MODEL = Model.NO_DISTILLATION
    elif modelChoice.get() == 'CNN model':
        CHOSEN_MODEL = Model.CNN
    elif modelChoice.get() == 'Teacher-Student model':
        CHOSEN_MODEL = Model.DISTILLED
    elif modelChoice.get() == 'Pruned model':
        CHOSEN_MODEL = Model.PRUNED




def showClassesInfo():
    newWindows = tk.Toplevel(window)
    newWindows.title('Flower classifier information')
    newWindows.geometry('500x300')
    newWindows.resizable(False, False)
    text = "Welcome to the Flower classifier program!\nChoose a flower database and model, then upload a photo.\nGet the top 3 predictions in percentages!"
    tk.Label(newWindows, text=text, font=('Lato', 12)).pack()

    classesAvailableText = '\n3 datasets are available'
    tk.Label(newWindows, text=classesAvailableText, font=('Lato', 11)).pack()
    datasetText = '- 5 classes - daisy, dandelion, rose, sunflower, tulip \n- 7 classes - bellflower, daisy, dandelion, lotus, rose, sunflower, tulip\n- 16 classes - astlibe, bellflower, black_eyed_susan, calendula,\ncalifornia_poppy,carnation,commmon_daisy,coreopsis,daffodil,dandelion,iris,\nmagnolia,rose,sunflower,tulip,water_lily'
    tk.Label(newWindows, text=datasetText, font=('Lato', 10),anchor='e', justify=tk.LEFT).pack()

    modelsAvailableText = '5 models are available'
    tk.Label(newWindows, text=modelsAvailableText, font=('Lato', 11)).pack()

    modelText = '- Reference model\n- Student without distillation model\n- CNN model\n- Teacher-Student model\n- Pruned model'
    tk.Label(newWindows, text=modelText, font=('Lato', 10),anchor='e', justify=tk.LEFT).pack()


def showFullPredictions():
    newWindow2 = tk.Toplevel(window)
    newWindow2.title('Full predictions')
    newWindow2.geometry('500x400')
    newWindow2.resizable(False, False)

    for label in newWindow2.winfo_children():
            label.destroy()

    if not fullPredictions:
        text = 'Upload a photo first!'
    else:
        text = 'Predictions:\n'
        for i in range(len(fullPredictions)):
            text = text + str(fullPredictions[i][0]) + ': ' + str(round(fullPredictions[i][1]*100, 2)) + '%\n'

    tk.Label(newWindow2, text=text, font=('Lato', 12)).pack()



window = ctk.CTk()
window.title("Flower classifier")
window.geometry('650x400')
window.resizable(False, False)

modelChoice = ctk.CTkComboBox(window, values=['Reference model', 'Student without distillation model', 'CNN model', 'Teacher-Student model','Pruned model'], command=chosenModel)
modelChoice.place(x=500, y=10)

datasetChoice = ctk.CTkComboBox(window, values=['Flowers 16 classes', 'Flowers 5 classes', 'Flowers 7 classes'], command=setDataset)
datasetChoice.place(x=500, y=60)


modelChooseLabel = ctk.CTkLabel(window, text='Choose model:')
modelChooseLabel.place(x=410, y=10)
datasetChooseLabel = ctk.CTkLabel(window, text='Choose database:')
datasetChooseLabel.place(x=395, y=60)

imageFrame = ctk.CTkFrame(window, width = 300, height = 300)
imageFrame.place(x=20,y =40)

#add button to upload a photo
uploadButton = ctk.CTkButton(window, text = 'Upload a photo', width = 20, height = 2, command=load_image)
uploadButton.place(x=250, y = 369)


predictedTextLabel = ctk.CTkLabel(window, text='Predicted flower:', font=('Lato', 15))
predictedTextLabel.place(x=20, y=350)

classesInfoButton = ctk.CTkButton(window, text = 'Info', width = 10, height = 2, command=showClassesInfo)
classesInfoButton.place(x=10, y = 10)

fullPredictionsButton = ctk.CTkButton(window, text = 'Check every prediction', width = 40, height = 5, command=showFullPredictions, font=('Lato', 15))
fullPredictionsButton.place(x=400, y = 250)

window.mainloop()