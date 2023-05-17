import os

import numpy as np
from PIL import Image
import pandas as pd
from matplotlib import pyplot as plt


def convert_to_csv():
    labels = []
    imgs = []
    flag = 0
    # iterate through all files in the 'flowers' folder and then through all files in the subfolders
    for root, dirs, files in os.walk("../flowers"):
        for file in files:
            if file.endswith("jpg"):
                # open the image
                img = Image.open(os.path.join(root, file))


                #print filename to check if it works
                print(os.path.join(root, file))
                # resize the image
                img = img.resize((128, 128)).convert('HSV')
                if flag == 0:
                    #plot the image with only hue value in matplotlib
                    plt.imshow(img, cmap='hsv')
                    plt.show()
                    flag = 1
                #take only hue value
                img = img.getdata(band=0)
                # convert image to numpy array
                img = np.array(img)

                imgs.append(img)
                #add label as first element of the array (label is the name of the subfolder)


                if root[11:] == 'daisy':
                    labels.append(0)
                    print(root[11:])
                elif root[11:] == 'dandelion':
                    labels.append(1)
                    print(root[11:])
                elif root[11:] == 'rose':
                    labels.append(2)
                    print(root[11:])
                elif root[11:] == 'sunflower':
                    labels.append(3)
                    print(root[11:])
                else:
                    labels.append(4)


                # add the name of subfolder as first element of the array
    print(imgs)
    # create a dataframe that will use imgs array as data and labels array as indexes
    df = pd.DataFrame(imgs, index=labels)
    # save dataframe to csv file
    df.to_csv("128dataH.csv")


def load_data():
    # load data from csv file
    data = pd.read_csv("128dataH.csv")
    return data