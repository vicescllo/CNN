import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras
import cv2

# Display
from IPython.display import Image, display
import matplotlib.cm as cm
from PIL import Image
import pandas as pd
import cv2


def get_img_array(img_path, size):
    """
    This function takes in an image path and size and returns a numpy array of the image.
    `img_path` is a string path to an image
    `size` is a tuple of (img_height, img_width)

    Returns a numpy array of shape (img_height, img_width, 3)
    """
    # `img` is a PIL image of size 299x299
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)
    return img


def make_gradcam_heatmap(img_array, model, last_conv_layer_name,img_path,values_top, pred_index=None,cosa=1):
    """
    This function takes in a model, image array, and the name of the last convolutional layer
    and generates a class activation heatmap for the top predicted class, or the provided index.

    `img_array` is the image array that you want to make the heatmap for
    `model` is a model object
    `last_conv_layer_name` is the name of the last convolutional layer in the classification model
    `pred_index` is an optional index specifying the class to take the heatmap for

    Returns a heatmap as an array

    """
    # First, we create a model that maps the input image to the activations

    # Create a graph that outputs target convolution and output
    img=img_array
    grad_model = tf.keras.models.Model(inputs=[model.input],
                                       outputs=[model.output, model.get_layer(last_conv_layer_name).output])

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        predictions, conv_outputs = grad_model(img)
        loss = predictions[:, 1]
    print('Prediction shape:', predictions.get_shape())
    # Extract filters and gradients
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    # Apply guided backpropagation
    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = gate_f * gate_r * grads

    # Average gradients spatially
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    # Build a ponderated map of filters according to gradients importance
    cam = np.ones(output.shape[0:2], dtype=np.float32)

    for index, w in enumerate(weights):
        cam += w * output[:, :, index]

    # Heatmap visualization
    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    
    
    
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    npimg = np.array(img[0])

    
    output_image = cv2.addWeighted(cv2.cvtColor(npimg.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)


    list_img=img_path.split("\\")

    img_ori = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(output_image)
    ax[1].imshow(img_ori)
    ax[0].axis('off')
    ax[0].set_title('Heatmap')
    ax[1].set_title('Original Image')
    ax[1].axis('off')
    #Create a directory called Unclassified_images
    fig.suptitle('Original image: {0} and Top 3 predictions: {1},{2},{3}'.format(values_top[0],values_top[1],values_top[2],values_top[3]), fontsize=16, fontweight='bold')
    import os
    if cosa==1:
        if not os.path.exists('Unclassified_images'):
            os.makedirs('Unclassified_images')
        if not os.path.exists('Unclassified_images/{0}'.format(values_top[0])):
            os.makedirs('Unclassified_images/{0}'.format(values_top[0]))
        fig.savefig('Unclassified_images/{0}/{1}.png'.format(values_top[0],list_img[1].split(".")[0]))
    else:
        plt.show()
        


def analysis_misclassified(model, df2):
    """
    This function takes in a model and a dataframe with the results of the predictions and generates a heatmap for the misclassified images.
    `model` is a model object
    `df2` is a dataframe with the results of the predictions

    Returns a heatmap for the misclassified images. 
    """
    ss=pd.DataFrame()
    class_labels=df2.loc[df2["Correct"]==False, "Predicted Label (Top 1)"].unique()
    for j in class_labels:
        a=df2[(df2["Predicted Label (Top 1)"]==j)&(df2["Correct"]==False)]
        ss=pd.concat([ss,a])
    b_path=["../dataset/validation/{0}".format(i) for i in ss.Filename.values]
    b_true=ss["True Label"].values
    b_top1=ss["Predicted Label (Top 1)"].values
    b_top2=ss["Top 2"].values
    b_top3=ss["Top 3"].values
    for n in range(len(b_path)):
        img_size = (224, 224)
        img_path = b_path[n]
        values_top=[b_true[n],b_top1[n],b_top2[n],b_top3[n]]
        preprocess_input = keras.applications.xception.preprocess_input
        img_array = preprocess_input(get_img_array(img_path, size=img_size))
        make_gradcam_heatmap(img_array, model, "conv5_block32_concat", img_path,values_top)
