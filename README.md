# 151-Kanto-Pokemon-Image-Classifier

Note: transfer_net.py does both architecture intialization, and training. test.py and image_visual.py are basically the same as my Kanto Pokemon image classifier. 

Image classifier that predicts what Pokemon the image is (from the original 151). Classifier was implemented using PyTorch. Network architecture derived using a pretrained densenet network, and then retraining it's classifier. 

Accuracy achieved on validation set ~73% after 25 epochs. 

Dataset I used: https://www.kaggle.com/thedagger/pokemon-generation-one/

