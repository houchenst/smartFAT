from __future__ import absolute_import, division, print_function
import cv2
import numpy as np
import tensorflow as tf
import hipCNN

def predict_file(file, classifier):
    predict_x = cv2.imread(file)
    predict_x = cv2.cvtColor(predict_x, cv2.COLOR_BGR2GRAY)
    predict_x = cv2.resize(predict_x, (28, 28))
    predict_array(predict_x, classifier)

def predict_array(predict_x, classifier):
    predict_x = predict_x/np.float32(255)
    predictions = classifier.predict(
    input_fn=tf.estimator.inputs.numpy_input_fn(
    x={"x": predict_x},
    num_epochs=1,
    shuffle=False))
    return list(predictions)[0]['probabilities'][1]

def locate_hip(file):

    hip_classifier = tf.estimator.Estimator(
        model_fn=hipCNN.cnn_model_fn, model_dir="model")

    i = cv2.imread(file)
    i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

    dim = int(i.shape[1]/25)

    width = i.shape[1]
    height = i.shape[0]

    y = 0
    x = 0
    step = 20

    vals = np.zeros((int((height-dim)/step)+1, int((width-dim)/step)+1))

    while y < height-dim:
        print(y)
        x = 0
        while x < width-dim:
            sub_image = i[y:y+dim, x:x+dim]
            sub_image = cv2.resize(sub_image, (28, 28))
            score = predict_array(sub_image, hip_classifier)
            # score = 1.0
            vals[int(y/step), int(x/step)] = score
            x+=step
        y+=step

    print(vals)
    # cv2.imshow("results", vals)
    best_y = np.argmax(vals, axis=0)
    best_x = np.argmax(vals, axis=1)
    best_image = vals[best_y:best_y+dim, best_x:best_x+dim]
    best_image = cv2.resize(best_image, (300, 300))
    cv2.imshow("hip", best_image)

    cv2.waitKey(0)






# predict_file("../data/finish-line/bmps/train/eval/20190413_144457_1327_neg1.bmp")
locate_hip("../data/finish-line/bmps/marked/20190413_140509_011.bmp")
image = cv2.imread("../data/finish-line/bmps/marked/20190413_140509_011.bmp")
cv2.imshow("image", image)
