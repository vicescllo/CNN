# CNN - Densenet201 - Tranfer Learning

In this repo, we will train a model to detect the following dataset.
The dataset can be downloaded [here.](https://drive.google.com/file/d/1JqXu1tSejWrwESAU_V5pYBw3PMdK2-yA/view?usp=sharing "Click here")

## Densenet201 final trained model (A Single Click Solution)

This team has trained a model which we have used to do this analysis. This trained model can be downloaded [here. ](https://drive.google.com/drive/folders/1Km1B9xPj4ra8YBGugduULftw7rYzmVIZ?usp=sharing "Click here")

Our team has trained a model with 100 epochs and 2 layers has been unfreezed and this is the results. We added a early-stopping criteria to avoid overfitting and this model only was trained 29 epochs.

![alt text](Image/output.png "Results")

Also de matrix confusion has been represented.

![alt text](Image/cm.png "Results")

It can be observed that the obtained validation set accuracy is 0.917

There is a called in this fuction to predict whether an image is correct or not. This will be the output. In addition, you can see top 3 predictions.

![alt text](Image/tabla_im.png "Results")

In addition, you can see where the model focus to classify an image. Here an example.

![alt text](Image/image1.png "Results")

