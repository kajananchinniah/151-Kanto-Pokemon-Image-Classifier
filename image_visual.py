'''
-------------------------------------------------------------
image_visual.py
-------------------------------------------------------------
Contains the following functions:
    - showImage(image, mean, std):
        - Changes image to CPU
        - Unnormalizes image (note: assumes image is 3x48x48; do not put the batch through this function)
        - Shows the unnormalized image
        
    - visualizeData(images, mean, std, classes, top_class, labels):
        
-------------------------------------------------------------

'''

#TODO: generalize visualizeData

import matplotlib.pyplot as plt

def showImage(image, mean, std):
    #Converting image to CPU
    image = image.cpu()
    #Unnormalize image, assuming it's 3x48x48 tensor
    image[0] = image[0] * mean[0] + std[0]
    image[1] = image[1] * mean[1] + std[1]
    image[2] = image[2] * mean[2] + std[2]
    plt.imshow(image.permute(1,2,0).numpy()) #Found permute online, don't fully understand it. TODO: Research on this


def visualizeData(images, mean, std, classes, top_class, labels):
    figure = plt.figure(figsize = (20, 20)) #arbiturary numbers
    for i in range(0, 16, 1):
        ax = figure.add_subplot(4, 4, i+1)
        ax.set_yticks([])
        ax.set_xticks([])
        showImage(images[i], mean, std)
        predicted_class = classes[top_class[i]]
        actual_class = classes[top_class[i]]
        if top_class[i] == labels[i]:
            str_out = predicted_class + ' (' + actual_class + ')'
            ax.set_title(str_out, color = "green")
        else:
            str_out = predicted_class + ' (' + actual_class + ')'
            ax.set_title(str_out, color = "red")
