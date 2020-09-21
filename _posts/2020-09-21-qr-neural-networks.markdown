---
layout: post
title:  "Decoding QR codes using neural networks"
date:   2020-09-21
tags: [machine-learning]
---

In this example, a neural network (NN) will decode QR codes with the text being up to 11 characters long.

This jupyter notebook is available <a href="https://github.com/andrewromanenco/ml/blob/master/qr-ff.ipynb" target="_blank">here</a>.

It is worth mentioning that using neural networks for QR decoding is not an optimal approach. There are well defined and simple algorithms to achieve these tasks. At the same time, due to ease of training data collection, QR codes provide a good practical experience for building NNs.

Decoding a QR code may be defined as a classification task. The result is a combination of outputs of eleven classifiers, one for each position. Each classifier tries to predict a symbol for a given spot. If an encoded text is shorter than eleven characters, a special symbol end-of-input (EOI) should be produced for empty positions. In this particular example, all texts are lowercase characters a..z and space. The total number of classes to be predicted is 28, which is 26 letter classes, one for space and one for EOI. For example, a string 'hello' may be represented by classes: 'h', 'e', 'l', 'l', 'o', '#', '#', '#', '#', '#', '#'. As a minor optimization, a working example could abandon symbols classification as soon as EOI class is detected since it is by definition that EOI may show up only at the end of the output.

p.s.
Why is the supported size up to 11 characters (and not 10 for example)? This is to be able to decode the 'hello world'.

## 1. Initialization
To run the example, the latest Anaconda distribution is used. It comes with all the main libraries. QR code library has to be installed via Anaconda package manager - "r-qrcode" is the package to add.

{% highlight python %}
# import all required libraries

import random
import string
import qrcode

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

# the next two lines fix a runtime bug on mac os
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
{% endhighlight %}

As it was mentioned above, input texts are up to eleven characters long. An input may contain lower case letters and spaces. End-of-input is a special class to assign to empty positions. Below are definitions for constants used to build and train NNs.


{% highlight python %}
LETTERS = string.ascii_lowercase + ' ' # letters and space
EOI = '#' # end of input
MAX_SIZE = 11 # max input/output size
ALL_CHAR_CLASSES = LETTERS + EOI # all available classes
{% endhighlight %}


{% highlight python %}
# define a function for a QR code generation
# box_size is an option to manage size of an output image
def make_qr(text, box_size = 1):
    qr = qrcode.QRCode(
    version=1,
    box_size=box_size,
    border=0)
    qr.add_data(text)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")
{% endhighlight %}


{% highlight python %}
# generate a sample image
make_qr("hello world", box_size=10)
{% endhighlight %}




![png](/assets/qr-nn/output_7_0.png)



In the above example, the box size is set to 10 to make details of the image look larger. It makes sense to minimize the size of an image being fed into an NN. To do so, all QR codes below are using a box size of 1. The size of the generated images is 21x21 pixels.


{% highlight python %}
print('FFNN input size is', make_qr("hello world").size)
{% endhighlight %}

    FFNN input size is (21, 21)


The training data set needs to be generated. The function below creates a given number of examples with length constraints.

The output is uniformly distributed. This applies to both sizes and characters in texts.

The output contains a NumPy array of all images and corresponding texts. Image data is already formatted to be fed to a NN. The output texts will be converted to the required format on the go.


{% highlight python %}
# a function to generate a train data set
# output: (numpy array of images, list of corresponding texts)
def generate_dataset(n_of_samples, min_size = 1, max_size = 11):
    data = []
    labels = []
    report_step = int(n_of_samples * .1)
    report = report_step
    print("Generating")
    for i in range(n_of_samples):
        if i == report:
            print("Done:", report / report_step * 10, "%")
            report = report + report_step
        size = random.randint(min_size, max_size)
        s = ''.join(random.choice(LETTERS) for i in range(size))
        img = make_qr(s)
        assert img.size == (21, 21)
        qr = np.asarray(img, dtype='float')
        data.append(qr)
        labels.append(s)
    print("Done:", "100", "%")
    return (np.asarray(data), labels)
{% endhighlight %}

The training set is 50k. This is an arbitrary number. It was chosen for code to be relatively fast on an average laptop. But, of course, a larger data set will lead to better accuracy.


{% highlight python %}
(training_data, training_labels) = generate_dataset(50000)
{% endhighlight %}

    Generating
    Done: 10.0 %
    Done: 20.0 %
    Done: 30.0 %
    Done: 40.0 %
    Done: 50.0 %
    Done: 60.0 %
    Done: 70.0 %
    Done: 80.0 %
    Done: 90.0 %
    Done: 100 %


## 2. Building a feed-forward neural network (FFNN) to estimate encoded text size

To get the ball rolling, the first FFNN will extract a text length from a QR image. This problem may be defined as a classification one. The input is a QR image. The output is one of 11 classes, one for each possible size.

The training set already has all the QR images. Size labels should be created based on original texts:


{% highlight python %}
# create a list of sizes based on original text labels
training_label_sizes = list(map(lambda x: [len(x) - 1], training_labels))
{% endhighlight %}


{% highlight python %}
# confirm sizes are uniformly distributed
sizes_frame = pd.DataFrame(training_label_sizes)
sizes_frame.hist(bins=11)
{% endhighlight %}




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7fbe5f453a90>]],
          dtype=object)




![png](/assets/qr-nn/output_16_1.png)


As expected, sizes are uniformly distributed.

Define first NN classifier with a single hidden layer:


{% highlight python %}
size_classifier = keras.Sequential([
    keras.layers.Flatten(input_shape=(21, 21)), # all input images are 21x21 pixels
    keras.layers.Dense(21*21, activation='relu'), # set hidden layer to the same size as the input
    keras.layers.Dense(MAX_SIZE) # output size is equal to number of classes, one for each size
])

size_classifier.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

size_classifier.summary()
{% endhighlight %}

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten (Flatten)            (None, 441)               0         
    _________________________________________________________________
    dense (Dense)                (None, 441)               194922    
    _________________________________________________________________
    dense_1 (Dense)              (None, 11)                4862      
    =================================================================
    Total params: 199,784
    Trainable params: 199,784
    Non-trainable params: 0
    _________________________________________________________________


Time to train the network. The validation split is set to 5%, which gives 2.5k of validation samples. Since the original data is randomly generated, there is no need to re-shuffle.

The early stop is based on validation loss. There is no saving for the best model, as this is an exploration run.


{% highlight python %}
# train the size classifier
size_history = size_classifier.fit(
    training_data, np.asarray(training_label_sizes),
    epochs=30, batch_size=128,
    validation_split=0.05,
    callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3)]
)

# plot the validation loss
plt.plot(size_history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="best")
plt.show()
{% endhighlight %}

    Train on 47500 samples, validate on 2500 samples
    Epoch 1/30
    47500/47500 [==============================] - 3s 56us/sample - loss: 0.4086 - accuracy: 0.9015 - val_loss: 0.0739 - val_accuracy: 0.9920
    Epoch 2/30
    47500/47500 [==============================] - 2s 45us/sample - loss: 0.0338 - accuracy: 0.9976 - val_loss: 0.0175 - val_accuracy: 1.0000
    Epoch 3/30
    47500/47500 [==============================] - 2s 47us/sample - loss: 0.0100 - accuracy: 0.9997 - val_loss: 0.0063 - val_accuracy: 1.0000
    Epoch 4/30
    47500/47500 [==============================] - 3s 57us/sample - loss: 0.0042 - accuracy: 1.0000 - val_loss: 0.0032 - val_accuracy: 1.0000
    Epoch 5/30
    47500/47500 [==============================] - 2s 52us/sample - loss: 0.0021 - accuracy: 1.0000 - val_loss: 0.0018 - val_accuracy: 1.0000
    Epoch 6/30
    47500/47500 [==============================] - 2s 50us/sample - loss: 0.0013 - accuracy: 1.0000 - val_loss: 0.0011 - val_accuracy: 1.0000
    Epoch 7/30
    47500/47500 [==============================] - 3s 54us/sample - loss: 8.1631e-04 - accuracy: 1.0000 - val_loss: 7.8178e-04 - val_accuracy: 1.0000
    Epoch 8/30
    47500/47500 [==============================] - 3s 54us/sample - loss: 5.6064e-04 - accuracy: 1.0000 - val_loss: 5.6709e-04 - val_accuracy: 1.0000
    Epoch 9/30
    47500/47500 [==============================] - 2s 48us/sample - loss: 3.9577e-04 - accuracy: 1.0000 - val_loss: 4.0009e-04 - val_accuracy: 1.0000
    Epoch 10/30
    47500/47500 [==============================] - 2s 47us/sample - loss: 2.8806e-04 - accuracy: 1.0000 - val_loss: 3.1972e-04 - val_accuracy: 1.0000
    Epoch 11/30
    47500/47500 [==============================] - 2s 49us/sample - loss: 2.1483e-04 - accuracy: 1.0000 - val_loss: 2.3780e-04 - val_accuracy: 1.0000
    Epoch 12/30
    47500/47500 [==============================] - 2s 50us/sample - loss: 1.6206e-04 - accuracy: 1.0000 - val_loss: 1.8862e-04 - val_accuracy: 1.0000
    Epoch 13/30
    47500/47500 [==============================] - 2s 49us/sample - loss: 1.2450e-04 - accuracy: 1.0000 - val_loss: 1.4431e-04 - val_accuracy: 1.0000
    Epoch 14/30
    47500/47500 [==============================] - 2s 48us/sample - loss: 9.6217e-05 - accuracy: 1.0000 - val_loss: 1.1433e-04 - val_accuracy: 1.0000
    Epoch 15/30
    47500/47500 [==============================] - 2s 48us/sample - loss: 7.5075e-05 - accuracy: 1.0000 - val_loss: 8.4301e-05 - val_accuracy: 1.0000
    Epoch 16/30
    47500/47500 [==============================] - 2s 49us/sample - loss: 5.8773e-05 - accuracy: 1.0000 - val_loss: 6.9176e-05 - val_accuracy: 1.0000
    Epoch 17/30
    47500/47500 [==============================] - 3s 53us/sample - loss: 4.6483e-05 - accuracy: 1.0000 - val_loss: 5.5826e-05 - val_accuracy: 1.0000



![png](/assets/qr-nn/output_20_1.png)


The training finished pretty fast and the validation accuracy is 1.

It is worth running the classifier on a few examples:


{% highlight python %}
# function to get a text size from a QR image using the pretrained NN
# the original NN is extended with a softmax layer
def get_size(qr_img):
        assert qr_img.size == (21, 21)
        qr = np.asarray(qr_img, dtype='float') # convert image to numpy array
        input = np.asarray([qr]) # NN takes a list of images as an input, make one item numpy array
        soft_max_model = keras.Sequential([size_classifier, keras.layers.Softmax()]) # make interpretation of classifier confidence easier
        output = soft_max_model.predict(input) # softmax returns a distribution of probabilities for each size
        largest_index = np.argmax(output[0], axis=0) # get an index of largest probability
        print("Size", largest_index + 1, ", confidence", output[0][largest_index])

test_set = [
    'x',
    'yo',
    'ham',
    'four',
    'f ive',
    'sixsix',
    'seven z',
    'ei ght x',
    'nine hops',
    'ten strike',
    'hello world'
]

for t in test_set:
    get_size(make_qr(t))
{% endhighlight %}

    Size 1 , confidence 0.9999956
    Size 2 , confidence 0.9999944
    Size 3 , confidence 0.9999914
    Size 4 , confidence 0.99998057
    Size 5 , confidence 0.9999546
    Size 6 , confidence 0.99999034
    Size 7 , confidence 0.9999995
    Size 8 , confidence 0.9999913
    Size 9 , confidence 0.9998647
    Size 10 , confidence 0.99928135
    Size 11 , confidence 0.99999785


All sizes are correct and confidence is at least three nines. This is a good result.

## 3. Exploring options by predicting the first and the last characters

NN for size prediction works very well. As the next step of exploration, we can move to character prediction. We start with the first and last characters to see how the model behaves.

Since the same structured networks are trained for different positions, it makes sense to define common functions:


{% highlight python %}
# create training labels for existing texts and required position
def make_labels_for_position(labels, pos):
    chars = list(map(lambda x: x[pos] if pos < len(x) else EOI, labels)) # either a letter or EOI
    return list(map(lambda x: [ALL_CHAR_CLASSES.index(x)], chars)) # all classes are indexed [0..len(ALL_CHAR_CLASSES))

# define a simple model. Input is a QR image. Output is a character class.
# start with singel hidden layer.
def define_char_classifier():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(21, 21)),
        keras.layers.Dense(21*21, activation='relu'),
        keras.layers.Dense(len(ALL_CHAR_CLASSES))
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# train a model using common options
def train_model(model, data, labels, epochs=75, patience=3):
    return model.fit(
        data, np.asarray(labels),
        epochs=epochs, batch_size=128,
        validation_split=0.05,
        callbacks=[EarlyStopping(monitor='val_loss' , min_delta=0.0001, patience=patience)]
    )

# get an actual letter predicted by a model
# since there is one network per position, no need in position argument
# input is a QR image, output is a letter, for the position the model is trained for
def get_letter(qr_img, model):
        assert qr_img.size == (21, 21)
        qr = np.asarray(qr_img, dtype='float')
        input = np.asarray([qr]) # convert image to NN input
        output = model.predict(input) # get a vector of logits for each class
        largest_index = np.argmax(output[0], axis=0) # pick an index of most probable class
        c = ALL_CHAR_CLASSES[largest_index] # get the actual character
        return (c, output[0][largest_index], np.around(output,2)) # return character

{% endhighlight %}

Start with the first letter:


{% highlight python %}
# create train labels using the first character of every original text
training_labels_char0 = make_labels_for_position(training_labels, 0)
# create and train model
model = define_char_classifier()
model.summary()
hist0 = train_model(model, training_data, training_labels_char0)
{% endhighlight %}

    Model: "sequential_12"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_1 (Flatten)          (None, 441)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 441)               194922    
    _________________________________________________________________
    dense_3 (Dense)              (None, 28)                12376     
    =================================================================
    Total params: 207,298
    Trainable params: 207,298
    Non-trainable params: 0
    _________________________________________________________________
    Train on 47500 samples, validate on 2500 samples
    Epoch 1/75
    47500/47500 [==============================] - 3s 59us/sample - loss: 2.8364 - accuracy: 0.1838 - val_loss: 2.2639 - val_accuracy: 0.2784
    Epoch 2/75
    47500/47500 [==============================] - 2s 46us/sample - loss: 1.6036 - accuracy: 0.4995 - val_loss: 0.9522 - val_accuracy: 0.7720
    Epoch 3/75
    47500/47500 [==============================] - 2s 51us/sample - loss: 0.5686 - accuracy: 0.8949 - val_loss: 0.3345 - val_accuracy: 0.9316
    Epoch 4/75
    47500/47500 [==============================] - 3s 55us/sample - loss: 0.2226 - accuracy: 0.9574 - val_loss: 0.1639 - val_accuracy: 0.9608
    Epoch 5/75
    47500/47500 [==============================] - 3s 58us/sample - loss: 0.1177 - accuracy: 0.9832 - val_loss: 0.0919 - val_accuracy: 0.9900
    Epoch 6/75
    47500/47500 [==============================] - 2s 52us/sample - loss: 0.0576 - accuracy: 0.9992 - val_loss: 0.0425 - val_accuracy: 1.0000
    Epoch 7/75
    47500/47500 [==============================] - 3s 53us/sample - loss: 0.0287 - accuracy: 1.0000 - val_loss: 0.0237 - val_accuracy: 1.0000
    Epoch 8/75
    47500/47500 [==============================] - 3s 56us/sample - loss: 0.0170 - accuracy: 1.0000 - val_loss: 0.0147 - val_accuracy: 1.0000
    Epoch 9/75
    47500/47500 [==============================] - 3s 55us/sample - loss: 0.0110 - accuracy: 1.0000 - val_loss: 0.0098 - val_accuracy: 1.0000
    Epoch 10/75
    47500/47500 [==============================] - 3s 59us/sample - loss: 0.0075 - accuracy: 1.0000 - val_loss: 0.0069 - val_accuracy: 1.0000
    Epoch 11/75
    47500/47500 [==============================] - 3s 57us/sample - loss: 0.0054 - accuracy: 1.0000 - val_loss: 0.0050 - val_accuracy: 1.0000
    Epoch 12/75
    47500/47500 [==============================] - 3s 53us/sample - loss: 0.0039 - accuracy: 1.0000 - val_loss: 0.0037 - val_accuracy: 1.0000
    Epoch 13/75
    47500/47500 [==============================] - 2s 52us/sample - loss: 0.0029 - accuracy: 1.0000 - val_loss: 0.0028 - val_accuracy: 1.0000
    Epoch 14/75
    47500/47500 [==============================] - 3s 55us/sample - loss: 0.0022 - accuracy: 1.0000 - val_loss: 0.0021 - val_accuracy: 1.0000
    Epoch 15/75
    47500/47500 [==============================] - 3s 53us/sample - loss: 0.0017 - accuracy: 1.0000 - val_loss: 0.0017 - val_accuracy: 1.0000
    Epoch 16/75
    47500/47500 [==============================] - 3s 58us/sample - loss: 0.0013 - accuracy: 1.0000 - val_loss: 0.0013 - val_accuracy: 1.0000
    Epoch 17/75
    47500/47500 [==============================] - 3s 59us/sample - loss: 0.0010 - accuracy: 1.0000 - val_loss: 9.7294e-04 - val_accuracy: 1.0000
    Epoch 18/75
    47500/47500 [==============================] - 3s 56us/sample - loss: 7.7575e-04 - accuracy: 1.0000 - val_loss: 7.4393e-04 - val_accuracy: 1.0000
    Epoch 19/75
    47500/47500 [==============================] - 3s 60us/sample - loss: 5.9778e-04 - accuracy: 1.0000 - val_loss: 5.7824e-04 - val_accuracy: 1.0000
    Epoch 20/75
    47500/47500 [==============================] - 3s 60us/sample - loss: 4.6365e-04 - accuracy: 1.0000 - val_loss: 4.4954e-04 - val_accuracy: 1.0000
    Epoch 21/75
    47500/47500 [==============================] - 3s 55us/sample - loss: 3.6141e-04 - accuracy: 1.0000 - val_loss: 3.5091e-04 - val_accuracy: 1.0000
    Epoch 22/75
    47500/47500 [==============================] - 3s 55us/sample - loss: 2.8391e-04 - accuracy: 1.0000 - val_loss: 2.7763e-04 - val_accuracy: 1.0000
    Epoch 23/75
    47500/47500 [==============================] - 3s 53us/sample - loss: 2.2476e-04 - accuracy: 1.0000 - val_loss: 2.1912e-04 - val_accuracy: 1.0000
    Epoch 24/75
    47500/47500 [==============================] - 3s 55us/sample - loss: 1.7838e-04 - accuracy: 1.0000 - val_loss: 1.7540e-04 - val_accuracy: 1.0000
    Epoch 25/75
    47500/47500 [==============================] - 3s 55us/sample - loss: 1.4194e-04 - accuracy: 1.0000 - val_loss: 1.4003e-04 - val_accuracy: 1.0000
    Epoch 26/75
    47500/47500 [==============================] - 3s 55us/sample - loss: 1.1334e-04 - accuracy: 1.0000 - val_loss: 1.1256e-04 - val_accuracy: 1.0000
    Epoch 27/75
    47500/47500 [==============================] - 3s 56us/sample - loss: 9.0879e-05 - accuracy: 1.0000 - val_loss: 8.9801e-05 - val_accuracy: 1.0000


After a short training, validation accuracy is 1. Loss and validation loss are also close:


{% highlight python %}
def plot_loss_vs_validation_loss_diff(train_history, abs_output = False):
    loss = np.asarray(train_history.history['loss'])
    val_loss = np.asarray(train_history.history['val_loss'])
    loss_diff = (loss - val_loss)/loss*100 # how much validation loss is different to loss
    if abs_output:
        loss_diff = np.abs(loss_diff)

    plt.plot(loss_diff, label='Loss diff %')
    plt.title('Loss vs Validation loss (diff)')
    plt.ylabel('Epochs')
    plt.ylabel('Loss diff %')
    plt.legend(loc="best")
    plt.show()

def plot_loss_vs_validation_loss(train_history, abs_output = False):
    plt.plot(np.asarray(train_history.history['loss']), label='Loss')
    plt.plot(np.asarray(train_history.history['val_loss']), label='Validation loss')
    plt.title('Loss vs Validation loss')
    plt.ylabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc="best")
    plt.show()

plot_loss_vs_validation_loss_diff(hist0)
plot_loss_vs_validation_loss(hist0)
{% endhighlight %}


![png](/assets/qr-nn/output_29_0.png)



![png](/assets/qr-nn/output_29_1.png)


Let's apply the trained network to a few examples:


{% highlight python %}
# function to try a text sample
# text is converted to a QR image, and then the network is predicting a character
def predict_letters_in_position(list_of_samples, model, postion):
    for t in list_of_samples:
        qr = make_qr(t)
        char_to_predict = t[postion] if len(t) > postion else '#'
        softmax_model = keras.Sequential([model, keras.layers.Softmax()])
        (c, confidence, _) = get_letter(qr, softmax_model)
        print(c == char_to_predict,
              'Expected:', char_to_predict,
              'Predicted:', c,
              'Input:', t,
              'Confidence:', confidence)

# use a text set used earlier
predict_letters_in_position(test_set, model, 0)
{% endhighlight %}

    True Expected: x Predicted: x Input: x Confidence: 0.99999416
    True Expected: y Predicted: y Input: yo Confidence: 0.99993885
    True Expected: h Predicted: h Input: ham Confidence: 0.99992704
    True Expected: f Predicted: f Input: four Confidence: 0.99985504
    True Expected: f Predicted: f Input: f ive Confidence: 0.999928
    True Expected: s Predicted: s Input: sixsix Confidence: 0.9997311
    True Expected: s Predicted: s Input: seven z Confidence: 0.99991214
    True Expected: e Predicted: e Input: ei ght x Confidence: 0.99977547
    True Expected: n Predicted: n Input: nine hops Confidence: 0.9998703
    True Expected: t Predicted: t Input: ten strike Confidence: 0.9998048
    True Expected: h Predicted: h Input: hello world Confidence: 0.999866


Again, great accuracy and confidence.

Time to try the last character:


{% highlight python %}
# create training labels from existing texts
training_labels_char10 = make_labels_for_position(training_labels, 10)
# build and train a model, and apply to text samples
model = define_char_classifier()
hist10 = train_model(model, training_data, training_labels_char10)
predict_letters_in_position(test_set, model, 10)
{% endhighlight %}

    Train on 47500 samples, validate on 2500 samples
    Epoch 1/75
    47500/47500 [==============================] - 3s 60us/sample - loss: 0.4419 - accuracy: 0.9071 - val_loss: 0.2932 - val_accuracy: 0.9192
    Epoch 2/75
    47500/47500 [==============================] - 2s 46us/sample - loss: 0.3016 - accuracy: 0.9155 - val_loss: 0.2711 - val_accuracy: 0.9232
    Epoch 3/75
    47500/47500 [==============================] - 2s 53us/sample - loss: 0.2629 - accuracy: 0.9220 - val_loss: 0.2296 - val_accuracy: 0.9260
    Epoch 4/75
    47500/47500 [==============================] - 3s 57us/sample - loss: 0.2224 - accuracy: 0.9305 - val_loss: 0.1973 - val_accuracy: 0.9348
    Epoch 5/75
    47500/47500 [==============================] - 3s 57us/sample - loss: 0.1786 - accuracy: 0.9419 - val_loss: 0.1556 - val_accuracy: 0.9456
    Epoch 6/75
    47500/47500 [==============================] - 2s 51us/sample - loss: 0.1386 - accuracy: 0.9553 - val_loss: 0.1253 - val_accuracy: 0.9564
    Epoch 7/75
    47500/47500 [==============================] - 3s 53us/sample - loss: 0.1036 - accuracy: 0.9691 - val_loss: 0.0963 - val_accuracy: 0.9692
    Epoch 8/75
    47500/47500 [==============================] - 2s 51us/sample - loss: 0.0764 - accuracy: 0.9790 - val_loss: 0.0735 - val_accuracy: 0.9768
    Epoch 9/75
    47500/47500 [==============================] - 2s 52us/sample - loss: 0.0545 - accuracy: 0.9874 - val_loss: 0.0557 - val_accuracy: 0.9828
    Epoch 10/75
    47500/47500 [==============================] - 3s 61us/sample - loss: 0.0391 - accuracy: 0.9918 - val_loss: 0.0469 - val_accuracy: 0.9868
    Epoch 11/75
    47500/47500 [==============================] - 3s 60us/sample - loss: 0.0278 - accuracy: 0.9949 - val_loss: 0.0325 - val_accuracy: 0.9904
    Epoch 12/75
    47500/47500 [==============================] - 3s 55us/sample - loss: 0.0192 - accuracy: 0.9972 - val_loss: 0.0254 - val_accuracy: 0.9948
    Epoch 13/75
    47500/47500 [==============================] - 3s 58us/sample - loss: 0.0133 - accuracy: 0.9987 - val_loss: 0.0193 - val_accuracy: 0.9968
    Epoch 14/75
    47500/47500 [==============================] - 3s 59us/sample - loss: 0.0094 - accuracy: 0.9993 - val_loss: 0.0156 - val_accuracy: 0.9972
    Epoch 15/75
    47500/47500 [==============================] - 3s 57us/sample - loss: 0.0065 - accuracy: 0.9997 - val_loss: 0.0120 - val_accuracy: 0.9988
    Epoch 16/75
    47500/47500 [==============================] - 3s 53us/sample - loss: 0.0047 - accuracy: 0.9999 - val_loss: 0.0109 - val_accuracy: 0.9976
    Epoch 17/75
    47500/47500 [==============================] - 3s 53us/sample - loss: 0.0035 - accuracy: 0.9999 - val_loss: 0.0083 - val_accuracy: 0.9984
    Epoch 18/75
    47500/47500 [==============================] - 3s 55us/sample - loss: 0.0027 - accuracy: 1.0000 - val_loss: 0.0065 - val_accuracy: 0.9992
    Epoch 19/75
    47500/47500 [==============================] - 3s 57us/sample - loss: 0.0019 - accuracy: 1.0000 - val_loss: 0.0061 - val_accuracy: 0.9992
    Epoch 20/75
    47500/47500 [==============================] - 3s 55us/sample - loss: 0.0015 - accuracy: 1.0000 - val_loss: 0.0049 - val_accuracy: 0.9996
    Epoch 21/75
    47500/47500 [==============================] - 3s 54us/sample - loss: 0.0012 - accuracy: 1.0000 - val_loss: 0.0040 - val_accuracy: 0.9992
    Epoch 22/75
    47500/47500 [==============================] - 3s 67us/sample - loss: 8.9789e-04 - accuracy: 1.0000 - val_loss: 0.0041 - val_accuracy: 0.9992
    Epoch 23/75
    47500/47500 [==============================] - 3s 56us/sample - loss: 6.9052e-04 - accuracy: 1.0000 - val_loss: 0.0028 - val_accuracy: 1.0000
    Epoch 24/75
    47500/47500 [==============================] - 3s 55us/sample - loss: 5.7058e-04 - accuracy: 1.0000 - val_loss: 0.0029 - val_accuracy: 0.9988
    Epoch 25/75
    47500/47500 [==============================] - 3s 66us/sample - loss: 4.2955e-04 - accuracy: 1.0000 - val_loss: 0.0022 - val_accuracy: 1.0000
    Epoch 26/75
    47500/47500 [==============================] - 3s 59us/sample - loss: 3.4260e-04 - accuracy: 1.0000 - val_loss: 0.0026 - val_accuracy: 0.9996
    Epoch 27/75
    47500/47500 [==============================] - 3s 56us/sample - loss: 2.7358e-04 - accuracy: 1.0000 - val_loss: 0.0017 - val_accuracy: 0.9996
    Epoch 28/75
    47500/47500 [==============================] - 3s 58us/sample - loss: 2.2452e-04 - accuracy: 1.0000 - val_loss: 0.0014 - val_accuracy: 1.0000
    Epoch 29/75
    47500/47500 [==============================] - 3s 59us/sample - loss: 1.8053e-04 - accuracy: 1.0000 - val_loss: 0.0012 - val_accuracy: 1.0000
    Epoch 30/75
    47500/47500 [==============================] - 3s 57us/sample - loss: 0.0066 - accuracy: 0.9982 - val_loss: 0.0015 - val_accuracy: 1.0000
    Epoch 31/75
    47500/47500 [==============================] - 3s 57us/sample - loss: 4.7402e-04 - accuracy: 1.0000 - val_loss: 9.5541e-04 - val_accuracy: 1.0000
    Epoch 32/75
    47500/47500 [==============================] - 3s 56us/sample - loss: 3.2779e-04 - accuracy: 1.0000 - val_loss: 8.1996e-04 - val_accuracy: 1.0000
    Epoch 33/75
    47500/47500 [==============================] - 3s 64us/sample - loss: 2.6733e-04 - accuracy: 1.0000 - val_loss: 7.9233e-04 - val_accuracy: 1.0000
    Epoch 34/75
    47500/47500 [==============================] - 3s 58us/sample - loss: 2.1291e-04 - accuracy: 1.0000 - val_loss: 6.6085e-04 - val_accuracy: 1.0000
    Epoch 35/75
    47500/47500 [==============================] - 3s 62us/sample - loss: 1.7700e-04 - accuracy: 1.0000 - val_loss: 5.9818e-04 - val_accuracy: 1.0000
    Epoch 36/75
    47500/47500 [==============================] - 4s 78us/sample - loss: 1.4876e-04 - accuracy: 1.0000 - val_loss: 5.4285e-04 - val_accuracy: 1.0000
    Epoch 37/75
    47500/47500 [==============================] - 4s 81us/sample - loss: 1.2318e-04 - accuracy: 1.0000 - val_loss: 5.3188e-04 - val_accuracy: 1.0000
    Epoch 38/75
    47500/47500 [==============================] - 3s 65us/sample - loss: 1.0261e-04 - accuracy: 1.0000 - val_loss: 4.9138e-04 - val_accuracy: 1.0000
    Epoch 39/75
    47500/47500 [==============================] - 3s 60us/sample - loss: 8.7857e-05 - accuracy: 1.0000 - val_loss: 4.4589e-04 - val_accuracy: 1.0000
    True Expected: # Predicted: # Input: x Confidence: 1.0
    True Expected: # Predicted: # Input: yo Confidence: 1.0
    True Expected: # Predicted: # Input: ham Confidence: 1.0
    True Expected: # Predicted: # Input: four Confidence: 1.0
    True Expected: # Predicted: # Input: f ive Confidence: 1.0
    True Expected: # Predicted: # Input: sixsix Confidence: 1.0
    True Expected: # Predicted: # Input: seven z Confidence: 0.99999964
    True Expected: # Predicted: # Input: ei ght x Confidence: 1.0
    True Expected: # Predicted: # Input: nine hops Confidence: 0.99999595
    True Expected: # Predicted: # Input: ten strike Confidence: 0.99999976
    True Expected: d Predicted: d Input: hello world Confidence: 0.99700135


Validation accuracy seems good. Let's compare metrics for the first and the last characters:


{% highlight python %}
plot_loss_vs_validation_loss_diff(hist10)
plot_loss_vs_validation_loss(hist10)

plt.plot(hist0.history['val_accuracy'], label='First char')
plt.plot(hist10.history['val_accuracy'], label='Last char')
plt.title('Accuracy first and last chars')
plt.ylabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="lower right")
axes = plt.gca()
plt.show()

{% endhighlight %}


![png](/assets/qr-nn/output_35_0.png)



![png](/assets/qr-nn/output_35_1.png)



![png](/assets/qr-nn/output_35_2.png)


Accuracy looks suspicious: from the very beginning, the accuracy for the last character is 90%! How so? Let's see how labels for the last character look like:


{% highlight python %}
frame10 = pd.DataFrame(training_labels_char10)
frame10[0].value_counts().plot(kind='bar')
{% endhighlight %}




    <matplotlib.axes._subplots.AxesSubplot at 0x7fbe4b062810>




![png](/assets/qr-nn/output_37_1.png)


Hm. The graph shows the number of samples per character. Character class 27 completely dominates sample space.


{% highlight python %}
# print 27th class
print("Character 27 is", ALL_CHAR_CLASSES[27])
{% endhighlight %}

    Character 27 is #


So the most frequent last character is # - which is the end of input. This should make sense. Since our data is generated in a uniform random fashion, the data set has the same number of text for every length - 1 to 11. Every example with a length of less than 11 has EOI symbols at 11th position. This is a problem with the data, which leads to poor NN performance. From the NN point of view, just classifying every last character as EOD, already gives a high percentage of accuracy.

The second conclusion comes to the number of samples. For the first character, there are 50k uniformly distributed samples. But from the second character and up, the uniform distribution is no longer true. Since the size is uniformly distributed, the number of EOI symbols is accumulated for every next position. This leaves fewer and fewer samples for character classes. For example, let's see how often 'a' is appearing in first and last positions:


{% highlight python %}
frame0 = pd.DataFrame(training_labels_char0)
print("Count 'a' in the first position:", len(frame0[frame0[0] == 0]))
print("Count 'a' in the last position:", len(frame10[frame10[0] == 0]))
{% endhighlight %}

    Count 'a' in the first position: 1860
    Count 'a' in the last position: 165


Quite a difference in the order of magnitude. This is expected. Since every string of sizes 1 to 11 has the first character - it is expected to have every class to appear 50k divided by the number of classes: 50k/28 ~ 1785.

This is not the case for the last character. The expected number of texts with size 11 is 50k/11 -> 4545. And there are 27 classes (28 minus one for EOI, since it can't be there) - 4545/27 = 168.

## 4. Improving accuracy for the last character

NN performs well for the first character and poorly for the last one. We should improve that. In general, the issue could be in data or NN design or both. We will start with the data.

### 4.1 Better data

We can try to train a separate network, with a data set of 11 size texts only. For the experiment, the dataset size is 50k/11 - the number of samples per size is the same, but the distribution of character is different, no more EOI. This will allow collecting some info if the dominance of EOI is actually a problem.


{% highlight python %}
(training_data11, training_labels11) = generate_dataset(int(50000/11), min_size=11, max_size=11)
training_labels11_char10 = make_labels_for_position(training_labels11, 10)
model = define_char_classifier()
hist10_with_size_11 = train_model(model, training_data11, training_labels11_char10, epochs=150)
predict_letters_in_position(test_set, model, 10)
{% endhighlight %}

    Generating
    Done: 10.0 %
    Done: 20.0 %
    Done: 30.0 %
    Done: 40.0 %
    Done: 50.0 %
    Done: 60.0 %
    Done: 70.0 %
    Done: 80.0 %
    Done: 90.0 %
    Done: 100.0 %
    Done: 100 %
    Train on 4317 samples, validate on 228 samples
    Epoch 1/150
    4317/4317 [==============================] - 1s 177us/sample - loss: 3.3678 - accuracy: 0.0477 - val_loss: 3.3240 - val_accuracy: 0.0307
    Epoch 2/150
    4317/4317 [==============================] - 0s 66us/sample - loss: 3.2260 - accuracy: 0.0737 - val_loss: 3.3186 - val_accuracy: 0.0658
    Epoch 3/150
    4317/4317 [==============================] - 0s 58us/sample - loss: 3.1283 - accuracy: 0.1161 - val_loss: 3.2285 - val_accuracy: 0.0789
    Epoch 4/150
    4317/4317 [==============================] - 0s 58us/sample - loss: 3.0084 - accuracy: 0.1663 - val_loss: 3.2488 - val_accuracy: 0.0570
    Epoch 5/150
    4317/4317 [==============================] - 0s 59us/sample - loss: 2.8719 - accuracy: 0.2080 - val_loss: 3.0817 - val_accuracy: 0.0965
    Epoch 6/150
    4317/4317 [==============================] - 0s 59us/sample - loss: 2.7087 - accuracy: 0.2391 - val_loss: 2.9870 - val_accuracy: 0.1360
    Epoch 7/150
    4317/4317 [==============================] - 0s 58us/sample - loss: 2.5340 - accuracy: 0.2912 - val_loss: 2.8377 - val_accuracy: 0.1535
    Epoch 8/150
    4317/4317 [==============================] - 0s 60us/sample - loss: 2.3747 - accuracy: 0.3336 - val_loss: 2.7823 - val_accuracy: 0.1491
    Epoch 9/150
    4317/4317 [==============================] - 0s 62us/sample - loss: 2.2087 - accuracy: 0.3702 - val_loss: 2.6128 - val_accuracy: 0.2018
    Epoch 10/150
    4317/4317 [==============================] - 0s 60us/sample - loss: 2.0330 - accuracy: 0.4341 - val_loss: 2.4352 - val_accuracy: 0.2061
    Epoch 11/150
    4317/4317 [==============================] - 0s 62us/sample - loss: 1.8556 - accuracy: 0.4925 - val_loss: 2.2842 - val_accuracy: 0.2632
    Epoch 12/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 1.6990 - accuracy: 0.5469 - val_loss: 2.1971 - val_accuracy: 0.2500
    Epoch 13/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 1.5564 - accuracy: 0.5830 - val_loss: 2.0539 - val_accuracy: 0.3070
    Epoch 14/150
    4317/4317 [==============================] - 0s 72us/sample - loss: 1.3966 - accuracy: 0.6477 - val_loss: 1.8881 - val_accuracy: 0.3465
    Epoch 15/150
    4317/4317 [==============================] - 0s 70us/sample - loss: 1.2522 - accuracy: 0.6933 - val_loss: 1.7991 - val_accuracy: 0.3904
    Epoch 16/150
    4317/4317 [==============================] - 0s 72us/sample - loss: 1.1229 - accuracy: 0.7385 - val_loss: 1.6400 - val_accuracy: 0.4342
    Epoch 17/150
    4317/4317 [==============================] - 0s 71us/sample - loss: 1.0108 - accuracy: 0.7788 - val_loss: 1.5657 - val_accuracy: 0.4386
    Epoch 18/150
    4317/4317 [==============================] - 0s 75us/sample - loss: 0.9020 - accuracy: 0.8135 - val_loss: 1.4222 - val_accuracy: 0.5000
    Epoch 19/150
    4317/4317 [==============================] - 0s 76us/sample - loss: 0.7954 - accuracy: 0.8552 - val_loss: 1.3541 - val_accuracy: 0.5570
    Epoch 20/150
    4317/4317 [==============================] - 0s 77us/sample - loss: 0.7122 - accuracy: 0.8731 - val_loss: 1.2750 - val_accuracy: 0.5482
    Epoch 21/150
    4317/4317 [==============================] - 0s 84us/sample - loss: 0.6331 - accuracy: 0.8972 - val_loss: 1.1936 - val_accuracy: 0.5789
    Epoch 22/150
    4317/4317 [==============================] - 0s 73us/sample - loss: 0.5631 - accuracy: 0.9189 - val_loss: 1.1300 - val_accuracy: 0.6184
    Epoch 23/150
    4317/4317 [==============================] - 0s 72us/sample - loss: 0.5075 - accuracy: 0.9338 - val_loss: 1.0307 - val_accuracy: 0.6184
    Epoch 24/150
    4317/4317 [==============================] - 0s 70us/sample - loss: 0.4497 - accuracy: 0.9537 - val_loss: 0.9534 - val_accuracy: 0.7018
    Epoch 25/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.3994 - accuracy: 0.9618 - val_loss: 0.8998 - val_accuracy: 0.6930
    Epoch 26/150
    4317/4317 [==============================] - 0s 72us/sample - loss: 0.3595 - accuracy: 0.9736 - val_loss: 0.8685 - val_accuracy: 0.7325
    Epoch 27/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.3234 - accuracy: 0.9782 - val_loss: 0.7939 - val_accuracy: 0.7500
    Epoch 28/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 0.2904 - accuracy: 0.9824 - val_loss: 0.7635 - val_accuracy: 0.7588
    Epoch 29/150
    4317/4317 [==============================] - 0s 74us/sample - loss: 0.2588 - accuracy: 0.9907 - val_loss: 0.7200 - val_accuracy: 0.8026
    Epoch 30/150
    4317/4317 [==============================] - 0s 74us/sample - loss: 0.2360 - accuracy: 0.9926 - val_loss: 0.6916 - val_accuracy: 0.7675
    Epoch 31/150
    4317/4317 [==============================] - 0s 78us/sample - loss: 0.2093 - accuracy: 0.9956 - val_loss: 0.6719 - val_accuracy: 0.8026
    Epoch 32/150
    4317/4317 [==============================] - 0s 69us/sample - loss: 0.1904 - accuracy: 0.9965 - val_loss: 0.6259 - val_accuracy: 0.8289
    Epoch 33/150
    4317/4317 [==============================] - 0s 70us/sample - loss: 0.1736 - accuracy: 0.9977 - val_loss: 0.5968 - val_accuracy: 0.8202
    Epoch 34/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.1598 - accuracy: 0.9981 - val_loss: 0.5789 - val_accuracy: 0.8553
    Epoch 35/150
    4317/4317 [==============================] - 0s 65us/sample - loss: 0.1458 - accuracy: 0.9984 - val_loss: 0.5585 - val_accuracy: 0.8465
    Epoch 36/150
    4317/4317 [==============================] - 0s 71us/sample - loss: 0.1312 - accuracy: 0.9993 - val_loss: 0.5115 - val_accuracy: 0.8553
    Epoch 37/150
    4317/4317 [==============================] - 0s 65us/sample - loss: 0.1192 - accuracy: 0.9998 - val_loss: 0.4978 - val_accuracy: 0.8728
    Epoch 38/150
    4317/4317 [==============================] - 0s 65us/sample - loss: 0.1105 - accuracy: 0.9995 - val_loss: 0.4853 - val_accuracy: 0.8728
    Epoch 39/150
    4317/4317 [==============================] - 0s 72us/sample - loss: 0.1033 - accuracy: 0.9995 - val_loss: 0.4869 - val_accuracy: 0.8596
    Epoch 40/150
    4317/4317 [==============================] - 0s 69us/sample - loss: 0.0948 - accuracy: 0.9995 - val_loss: 0.4531 - val_accuracy: 0.8772
    Epoch 41/150
    4317/4317 [==============================] - 0s 70us/sample - loss: 0.0871 - accuracy: 1.0000 - val_loss: 0.4480 - val_accuracy: 0.8947
    Epoch 42/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0807 - accuracy: 1.0000 - val_loss: 0.4217 - val_accuracy: 0.9123
    Epoch 43/150
    4317/4317 [==============================] - 0s 88us/sample - loss: 0.0748 - accuracy: 1.0000 - val_loss: 0.4156 - val_accuracy: 0.9035
    Epoch 44/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0699 - accuracy: 1.0000 - val_loss: 0.3985 - val_accuracy: 0.9123
    Epoch 45/150
    4317/4317 [==============================] - 0s 69us/sample - loss: 0.0654 - accuracy: 1.0000 - val_loss: 0.3836 - val_accuracy: 0.9079
    Epoch 46/150
    4317/4317 [==============================] - 0s 69us/sample - loss: 0.0610 - accuracy: 1.0000 - val_loss: 0.3815 - val_accuracy: 0.9123
    Epoch 47/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 0.0578 - accuracy: 1.0000 - val_loss: 0.3640 - val_accuracy: 0.9211
    Epoch 48/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 0.0539 - accuracy: 1.0000 - val_loss: 0.3510 - val_accuracy: 0.9298
    Epoch 49/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 0.0503 - accuracy: 1.0000 - val_loss: 0.3391 - val_accuracy: 0.9298
    Epoch 50/150
    4317/4317 [==============================] - 0s 72us/sample - loss: 0.0472 - accuracy: 1.0000 - val_loss: 0.3295 - val_accuracy: 0.9342
    Epoch 51/150
    4317/4317 [==============================] - 0s 75us/sample - loss: 0.0444 - accuracy: 1.0000 - val_loss: 0.3253 - val_accuracy: 0.9342
    Epoch 52/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0421 - accuracy: 1.0000 - val_loss: 0.3217 - val_accuracy: 0.9254
    Epoch 53/150
    4317/4317 [==============================] - 0s 72us/sample - loss: 0.0398 - accuracy: 1.0000 - val_loss: 0.3170 - val_accuracy: 0.9386
    Epoch 54/150
    4317/4317 [==============================] - 0s 81us/sample - loss: 0.0381 - accuracy: 1.0000 - val_loss: 0.3051 - val_accuracy: 0.9386
    Epoch 55/150
    4317/4317 [==============================] - 0s 72us/sample - loss: 0.0357 - accuracy: 1.0000 - val_loss: 0.2931 - val_accuracy: 0.9298
    Epoch 56/150
    4317/4317 [==============================] - 0s 69us/sample - loss: 0.0338 - accuracy: 1.0000 - val_loss: 0.2929 - val_accuracy: 0.9342
    Epoch 57/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0321 - accuracy: 1.0000 - val_loss: 0.2870 - val_accuracy: 0.9342
    Epoch 58/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0307 - accuracy: 1.0000 - val_loss: 0.2890 - val_accuracy: 0.9386
    Epoch 59/150
    4317/4317 [==============================] - 0s 73us/sample - loss: 0.0291 - accuracy: 1.0000 - val_loss: 0.2695 - val_accuracy: 0.9430
    Epoch 60/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0278 - accuracy: 1.0000 - val_loss: 0.2661 - val_accuracy: 0.9342
    Epoch 61/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 0.0264 - accuracy: 1.0000 - val_loss: 0.2644 - val_accuracy: 0.9386
    Epoch 62/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0252 - accuracy: 1.0000 - val_loss: 0.2619 - val_accuracy: 0.9386
    Epoch 63/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0241 - accuracy: 1.0000 - val_loss: 0.2538 - val_accuracy: 0.9386
    Epoch 64/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 0.0230 - accuracy: 1.0000 - val_loss: 0.2477 - val_accuracy: 0.9386
    Epoch 65/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0219 - accuracy: 1.0000 - val_loss: 0.2456 - val_accuracy: 0.9386
    Epoch 66/150
    4317/4317 [==============================] - 0s 71us/sample - loss: 0.0210 - accuracy: 1.0000 - val_loss: 0.2373 - val_accuracy: 0.9386
    Epoch 67/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 0.0202 - accuracy: 1.0000 - val_loss: 0.2345 - val_accuracy: 0.9430
    Epoch 68/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0193 - accuracy: 1.0000 - val_loss: 0.2317 - val_accuracy: 0.9430
    Epoch 69/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0185 - accuracy: 1.0000 - val_loss: 0.2297 - val_accuracy: 0.9430
    Epoch 70/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0178 - accuracy: 1.0000 - val_loss: 0.2233 - val_accuracy: 0.9430
    Epoch 71/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 0.0170 - accuracy: 1.0000 - val_loss: 0.2252 - val_accuracy: 0.9474
    Epoch 72/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0164 - accuracy: 1.0000 - val_loss: 0.2202 - val_accuracy: 0.9430
    Epoch 73/150
    4317/4317 [==============================] - 0s 65us/sample - loss: 0.0157 - accuracy: 1.0000 - val_loss: 0.2169 - val_accuracy: 0.9430
    Epoch 74/150
    4317/4317 [==============================] - 0s 65us/sample - loss: 0.0152 - accuracy: 1.0000 - val_loss: 0.2107 - val_accuracy: 0.9430
    Epoch 75/150
    4317/4317 [==============================] - 0s 66us/sample - loss: 0.0146 - accuracy: 1.0000 - val_loss: 0.2088 - val_accuracy: 0.9430
    Epoch 76/150
    4317/4317 [==============================] - 0s 66us/sample - loss: 0.0140 - accuracy: 1.0000 - val_loss: 0.2087 - val_accuracy: 0.9474
    Epoch 77/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0136 - accuracy: 1.0000 - val_loss: 0.2041 - val_accuracy: 0.9474
    Epoch 78/150
    4317/4317 [==============================] - 0s 75us/sample - loss: 0.0131 - accuracy: 1.0000 - val_loss: 0.2036 - val_accuracy: 0.9474
    Epoch 79/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0126 - accuracy: 1.0000 - val_loss: 0.1982 - val_accuracy: 0.9518
    Epoch 80/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 0.0122 - accuracy: 1.0000 - val_loss: 0.1977 - val_accuracy: 0.9430
    Epoch 81/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0117 - accuracy: 1.0000 - val_loss: 0.1929 - val_accuracy: 0.9474
    Epoch 82/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 0.0113 - accuracy: 1.0000 - val_loss: 0.1895 - val_accuracy: 0.9518
    Epoch 83/150
    4317/4317 [==============================] - 0s 101us/sample - loss: 0.0109 - accuracy: 1.0000 - val_loss: 0.1879 - val_accuracy: 0.9474
    Epoch 84/150
    4317/4317 [==============================] - 0s 78us/sample - loss: 0.0105 - accuracy: 1.0000 - val_loss: 0.1843 - val_accuracy: 0.9474
    Epoch 85/150
    4317/4317 [==============================] - 0s 72us/sample - loss: 0.0102 - accuracy: 1.0000 - val_loss: 0.1891 - val_accuracy: 0.9474
    Epoch 86/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 0.0098 - accuracy: 1.0000 - val_loss: 0.1777 - val_accuracy: 0.9474
    Epoch 87/150
    4317/4317 [==============================] - 0s 69us/sample - loss: 0.0096 - accuracy: 1.0000 - val_loss: 0.1806 - val_accuracy: 0.9474
    Epoch 88/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 0.0093 - accuracy: 1.0000 - val_loss: 0.1760 - val_accuracy: 0.9474
    Epoch 89/150
    4317/4317 [==============================] - 0s 72us/sample - loss: 0.0089 - accuracy: 1.0000 - val_loss: 0.1787 - val_accuracy: 0.9474
    Epoch 90/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0086 - accuracy: 1.0000 - val_loss: 0.1720 - val_accuracy: 0.9474
    Epoch 91/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0083 - accuracy: 1.0000 - val_loss: 0.1710 - val_accuracy: 0.9474
    Epoch 92/150
    4317/4317 [==============================] - 0s 69us/sample - loss: 0.0081 - accuracy: 1.0000 - val_loss: 0.1697 - val_accuracy: 0.9474
    Epoch 93/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 0.0079 - accuracy: 1.0000 - val_loss: 0.1662 - val_accuracy: 0.9474
    Epoch 94/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 0.0076 - accuracy: 1.0000 - val_loss: 0.1641 - val_accuracy: 0.9474
    Epoch 95/150
    4317/4317 [==============================] - 0s 70us/sample - loss: 0.0074 - accuracy: 1.0000 - val_loss: 0.1645 - val_accuracy: 0.9518
    Epoch 96/150
    4317/4317 [==============================] - 0s 70us/sample - loss: 0.0071 - accuracy: 1.0000 - val_loss: 0.1657 - val_accuracy: 0.9474
    Epoch 97/150
    4317/4317 [==============================] - 0s 75us/sample - loss: 0.0069 - accuracy: 1.0000 - val_loss: 0.1620 - val_accuracy: 0.9474
    Epoch 98/150
    4317/4317 [==============================] - 0s 69us/sample - loss: 0.0067 - accuracy: 1.0000 - val_loss: 0.1634 - val_accuracy: 0.9518
    Epoch 99/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 0.0065 - accuracy: 1.0000 - val_loss: 0.1582 - val_accuracy: 0.9474
    Epoch 100/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0063 - accuracy: 1.0000 - val_loss: 0.1550 - val_accuracy: 0.9518
    Epoch 101/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0061 - accuracy: 1.0000 - val_loss: 0.1540 - val_accuracy: 0.9518
    Epoch 102/150
    4317/4317 [==============================] - 0s 69us/sample - loss: 0.0060 - accuracy: 1.0000 - val_loss: 0.1543 - val_accuracy: 0.9474
    Epoch 103/150
    4317/4317 [==============================] - 0s 74us/sample - loss: 0.0058 - accuracy: 1.0000 - val_loss: 0.1492 - val_accuracy: 0.9518
    Epoch 104/150
    4317/4317 [==============================] - 0s 69us/sample - loss: 0.0056 - accuracy: 1.0000 - val_loss: 0.1495 - val_accuracy: 0.9518
    Epoch 105/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0055 - accuracy: 1.0000 - val_loss: 0.1502 - val_accuracy: 0.9518
    Epoch 106/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0053 - accuracy: 1.0000 - val_loss: 0.1466 - val_accuracy: 0.9518
    Epoch 107/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0052 - accuracy: 1.0000 - val_loss: 0.1467 - val_accuracy: 0.9518
    Epoch 108/150
    4317/4317 [==============================] - 0s 69us/sample - loss: 0.0050 - accuracy: 1.0000 - val_loss: 0.1439 - val_accuracy: 0.9518
    Epoch 109/150
    4317/4317 [==============================] - 0s 71us/sample - loss: 0.0049 - accuracy: 1.0000 - val_loss: 0.1421 - val_accuracy: 0.9518
    Epoch 110/150
    4317/4317 [==============================] - 0s 66us/sample - loss: 0.0047 - accuracy: 1.0000 - val_loss: 0.1427 - val_accuracy: 0.9474
    Epoch 111/150
    4317/4317 [==============================] - 0s 64us/sample - loss: 0.0046 - accuracy: 1.0000 - val_loss: 0.1434 - val_accuracy: 0.9474
    Epoch 112/150
    4317/4317 [==============================] - 0s 66us/sample - loss: 0.0045 - accuracy: 1.0000 - val_loss: 0.1413 - val_accuracy: 0.9518
    Epoch 113/150
    4317/4317 [==============================] - 0s 69us/sample - loss: 0.0043 - accuracy: 1.0000 - val_loss: 0.1397 - val_accuracy: 0.9518
    Epoch 114/150
    4317/4317 [==============================] - 0s 71us/sample - loss: 0.0042 - accuracy: 1.0000 - val_loss: 0.1416 - val_accuracy: 0.9518
    Epoch 115/150
    4317/4317 [==============================] - 0s 71us/sample - loss: 0.0041 - accuracy: 1.0000 - val_loss: 0.1357 - val_accuracy: 0.9518
    Epoch 116/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0040 - accuracy: 1.0000 - val_loss: 0.1359 - val_accuracy: 0.9518
    Epoch 117/150
    4317/4317 [==============================] - 0s 72us/sample - loss: 0.0039 - accuracy: 1.0000 - val_loss: 0.1377 - val_accuracy: 0.9518
    Epoch 118/150
    4317/4317 [==============================] - 0s 75us/sample - loss: 0.0038 - accuracy: 1.0000 - val_loss: 0.1331 - val_accuracy: 0.9518
    Epoch 119/150
    4317/4317 [==============================] - 0s 72us/sample - loss: 0.0037 - accuracy: 1.0000 - val_loss: 0.1327 - val_accuracy: 0.9518
    Epoch 120/150
    4317/4317 [==============================] - 0s 71us/sample - loss: 0.0036 - accuracy: 1.0000 - val_loss: 0.1339 - val_accuracy: 0.9518
    Epoch 121/150
    4317/4317 [==============================] - 0s 72us/sample - loss: 0.0035 - accuracy: 1.0000 - val_loss: 0.1334 - val_accuracy: 0.9518
    Epoch 122/150
    4317/4317 [==============================] - 0s 74us/sample - loss: 0.0034 - accuracy: 1.0000 - val_loss: 0.1301 - val_accuracy: 0.9518
    Epoch 123/150
    4317/4317 [==============================] - 0s 72us/sample - loss: 0.0033 - accuracy: 1.0000 - val_loss: 0.1290 - val_accuracy: 0.9518
    Epoch 124/150
    4317/4317 [==============================] - 0s 71us/sample - loss: 0.0033 - accuracy: 1.0000 - val_loss: 0.1264 - val_accuracy: 0.9518
    Epoch 125/150
    4317/4317 [==============================] - 0s 71us/sample - loss: 0.0032 - accuracy: 1.0000 - val_loss: 0.1277 - val_accuracy: 0.9518
    Epoch 126/150
    4317/4317 [==============================] - 0s 75us/sample - loss: 0.0031 - accuracy: 1.0000 - val_loss: 0.1259 - val_accuracy: 0.9518
    Epoch 127/150
    4317/4317 [==============================] - 0s 77us/sample - loss: 0.0030 - accuracy: 1.0000 - val_loss: 0.1272 - val_accuracy: 0.9518
    Epoch 128/150
    4317/4317 [==============================] - 0s 70us/sample - loss: 0.0029 - accuracy: 1.0000 - val_loss: 0.1249 - val_accuracy: 0.9518
    Epoch 129/150
    4317/4317 [==============================] - 0s 69us/sample - loss: 0.0029 - accuracy: 1.0000 - val_loss: 0.1252 - val_accuracy: 0.9518
    Epoch 130/150
    4317/4317 [==============================] - 0s 73us/sample - loss: 0.0028 - accuracy: 1.0000 - val_loss: 0.1254 - val_accuracy: 0.9518
    Epoch 131/150
    4317/4317 [==============================] - 0s 72us/sample - loss: 0.0027 - accuracy: 1.0000 - val_loss: 0.1207 - val_accuracy: 0.9561
    Epoch 132/150
    4317/4317 [==============================] - 0s 75us/sample - loss: 0.0027 - accuracy: 1.0000 - val_loss: 0.1218 - val_accuracy: 0.9518
    Epoch 133/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 0.0026 - accuracy: 1.0000 - val_loss: 0.1205 - val_accuracy: 0.9518
    Epoch 134/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 0.0025 - accuracy: 1.0000 - val_loss: 0.1175 - val_accuracy: 0.9561
    Epoch 135/150
    4317/4317 [==============================] - 0s 68us/sample - loss: 0.0025 - accuracy: 1.0000 - val_loss: 0.1187 - val_accuracy: 0.9518
    Epoch 136/150
    4317/4317 [==============================] - 0s 67us/sample - loss: 0.0024 - accuracy: 1.0000 - val_loss: 0.1182 - val_accuracy: 0.9518
    Epoch 137/150
    4317/4317 [==============================] - 0s 71us/sample - loss: 0.0024 - accuracy: 1.0000 - val_loss: 0.1177 - val_accuracy: 0.9561
    False Expected: # Predicted: a Input: x Confidence: 0.6288551
    False Expected: # Predicted: m Input: yo Confidence: 0.96041507
    False Expected: # Predicted: a Input: ham Confidence: 0.8889916
    False Expected: # Predicted: o Input: four Confidence: 0.8232217
    False Expected: # Predicted: a Input: f ive Confidence: 0.9190856
    False Expected: # Predicted: z Input: sixsix Confidence: 0.9657479
    False Expected: # Predicted: a Input: seven z Confidence: 0.9826301
    False Expected: # Predicted: z Input: ei ght x Confidence: 0.9266826
    False Expected: # Predicted: a Input: nine hops Confidence: 0.9878347
    False Expected: # Predicted: f Input: ten strike Confidence: 0.7795283
    True Expected: d Predicted: d Input: hello world Confidence: 0.97773224



{% highlight python %}
plt.plot(hist10.history['val_accuracy'], label='Original data set')
plt.plot(hist10_with_size_11.history['val_accuracy'], label='Subset of texts with len 11')
plt.title('Accuracy')
plt.ylabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="best")
axes = plt.gca()
plt.show()

plt.plot(hist10.history['val_loss'], label='Original data set')
plt.plot(hist10_with_size_11.history['val_loss'], label='Subset of texts with len 11')
plt.title('Loss')
plt.ylabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="best")
plt.show()
{% endhighlight %}


![png](/assets/qr-nn/output_44_0.png)



![png](/assets/qr-nn/output_44_1.png)


The trend looks as expected, less data did result in less progress in both accuracy and lost. Let's see how the last 20 epochs behave in both cases:


{% highlight python %}
plt.plot(hist10.history['val_accuracy'][-20:], label='Original data set')
plt.plot(hist10_with_size_11.history['val_accuracy'][-20:], label='Subset of texts with len 11')
plt.title('Accuracy, last 20 epochs')
plt.ylabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="best")
axes = plt.gca()
plt.show()

plt.plot(hist10.history['val_loss'][-20:], label='Original data set')
plt.plot(hist10_with_size_11.history['val_loss'][-20:], label='Subset of texts with len 11')
plt.title('Loss, last 20 epochs')
plt.ylabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="best")
plt.show()
{% endhighlight %}


![png](/assets/qr-nn/output_46_0.png)



![png](/assets/qr-nn/output_46_1.png)


Not good. The set of 11 length texts is too small to achieve the original performance.

To completely proof this idea, a new dataset could be generated. This dataset has twice more samples, comparing to previous runs.


{% highlight python %}
(training_data11, training_labels11) = generate_dataset(int(50000/11*2), min_size=11, max_size=11) # original dataset size was 50000/11 - 50k samples uniformly distributed over sizes 1..11
training_labels11_char10 = make_labels_for_position(training_labels11, 10)
model = define_char_classifier()
hist10_with_size_11_2x = train_model(model, training_data11, training_labels11_char10, epochs=150)
{% endhighlight %}

    Generating
    Done: 10.0 %
    Done: 20.0 %
    Done: 30.0 %
    Done: 40.0 %
    Done: 50.0 %
    Done: 60.0 %
    Done: 70.0 %
    Done: 80.0 %
    Done: 90.0 %
    Done: 100 %
    Train on 8635 samples, validate on 455 samples
    Epoch 1/150
    8635/8635 [==============================] - 1s 117us/sample - loss: 3.3309 - accuracy: 0.0491 - val_loss: 3.3035 - val_accuracy: 0.0440
    Epoch 2/150
    8635/8635 [==============================] - 0s 51us/sample - loss: 3.1605 - accuracy: 0.0990 - val_loss: 3.1738 - val_accuracy: 0.0945
    Epoch 3/150
    8635/8635 [==============================] - 0s 51us/sample - loss: 2.9574 - accuracy: 0.1634 - val_loss: 2.9456 - val_accuracy: 0.1604
    Epoch 4/150
    8635/8635 [==============================] - 0s 55us/sample - loss: 2.6786 - accuracy: 0.2327 - val_loss: 2.7075 - val_accuracy: 0.2154
    Epoch 5/150
    8635/8635 [==============================] - 1s 64us/sample - loss: 2.3789 - accuracy: 0.3032 - val_loss: 2.3894 - val_accuracy: 0.2945
    Epoch 6/150
    8635/8635 [==============================] - 1s 63us/sample - loss: 2.0535 - accuracy: 0.4118 - val_loss: 2.1108 - val_accuracy: 0.3297
    Epoch 7/150
    8635/8635 [==============================] - 1s 62us/sample - loss: 1.7278 - accuracy: 0.5266 - val_loss: 1.8076 - val_accuracy: 0.4571
    Epoch 8/150
    8635/8635 [==============================] - 1s 68us/sample - loss: 1.4237 - accuracy: 0.6483 - val_loss: 1.5176 - val_accuracy: 0.5516
    Epoch 9/150
    8635/8635 [==============================] - 1s 65us/sample - loss: 1.1454 - accuracy: 0.7473 - val_loss: 1.2581 - val_accuracy: 0.6242
    Epoch 10/150
    8635/8635 [==============================] - 1s 64us/sample - loss: 0.9165 - accuracy: 0.8270 - val_loss: 1.0448 - val_accuracy: 0.7253
    Epoch 11/150
    8635/8635 [==============================] - 1s 68us/sample - loss: 0.7312 - accuracy: 0.8774 - val_loss: 0.8624 - val_accuracy: 0.7824
    Epoch 12/150
    8635/8635 [==============================] - 1s 64us/sample - loss: 0.5897 - accuracy: 0.9097 - val_loss: 0.7155 - val_accuracy: 0.8352
    Epoch 13/150
    8635/8635 [==============================] - 1s 71us/sample - loss: 0.4817 - accuracy: 0.9289 - val_loss: 0.6114 - val_accuracy: 0.8571
    Epoch 14/150
    8635/8635 [==============================] - 1s 76us/sample - loss: 0.3994 - accuracy: 0.9433 - val_loss: 0.5373 - val_accuracy: 0.8747
    Epoch 15/150
    8635/8635 [==============================] - 1s 74us/sample - loss: 0.3285 - accuracy: 0.9650 - val_loss: 0.4594 - val_accuracy: 0.8901
    Epoch 16/150
    8635/8635 [==============================] - 1s 70us/sample - loss: 0.2739 - accuracy: 0.9749 - val_loss: 0.3945 - val_accuracy: 0.9033
    Epoch 17/150
    8635/8635 [==============================] - 1s 62us/sample - loss: 0.2320 - accuracy: 0.9844 - val_loss: 0.3555 - val_accuracy: 0.9319
    Epoch 18/150
    8635/8635 [==============================] - 1s 63us/sample - loss: 0.1941 - accuracy: 0.9888 - val_loss: 0.3088 - val_accuracy: 0.9363
    Epoch 19/150
    8635/8635 [==============================] - 1s 60us/sample - loss: 0.1631 - accuracy: 0.9925 - val_loss: 0.2766 - val_accuracy: 0.9495
    Epoch 20/150
    8635/8635 [==============================] - 1s 59us/sample - loss: 0.1396 - accuracy: 0.9950 - val_loss: 0.2399 - val_accuracy: 0.9582
    Epoch 21/150
    8635/8635 [==============================] - 1s 65us/sample - loss: 0.1199 - accuracy: 0.9971 - val_loss: 0.2104 - val_accuracy: 0.9648
    Epoch 22/150
    8635/8635 [==============================] - 1s 60us/sample - loss: 0.1033 - accuracy: 0.9986 - val_loss: 0.1905 - val_accuracy: 0.9780
    Epoch 23/150
    8635/8635 [==============================] - 1s 63us/sample - loss: 0.0899 - accuracy: 0.9985 - val_loss: 0.1720 - val_accuracy: 0.9758
    Epoch 24/150
    8635/8635 [==============================] - 1s 65us/sample - loss: 0.0788 - accuracy: 0.9992 - val_loss: 0.1546 - val_accuracy: 0.9824
    Epoch 25/150
    8635/8635 [==============================] - 1s 62us/sample - loss: 0.0689 - accuracy: 0.9995 - val_loss: 0.1457 - val_accuracy: 0.9758
    Epoch 26/150
    8635/8635 [==============================] - 1s 62us/sample - loss: 0.0608 - accuracy: 0.9998 - val_loss: 0.1289 - val_accuracy: 0.9846
    Epoch 27/150
    8635/8635 [==============================] - 1s 72us/sample - loss: 0.0540 - accuracy: 1.0000 - val_loss: 0.1199 - val_accuracy: 0.9824
    Epoch 28/150
    8635/8635 [==============================] - 1s 73us/sample - loss: 0.0480 - accuracy: 0.9999 - val_loss: 0.1114 - val_accuracy: 0.9846
    Epoch 29/150
    8635/8635 [==============================] - 1s 68us/sample - loss: 0.0432 - accuracy: 1.0000 - val_loss: 0.1023 - val_accuracy: 0.9846
    Epoch 30/150
    8635/8635 [==============================] - 1s 68us/sample - loss: 0.0388 - accuracy: 1.0000 - val_loss: 0.0935 - val_accuracy: 0.9890
    Epoch 31/150
    8635/8635 [==============================] - 1s 63us/sample - loss: 0.0349 - accuracy: 1.0000 - val_loss: 0.0880 - val_accuracy: 0.9868
    Epoch 32/150
    8635/8635 [==============================] - 1s 66us/sample - loss: 0.0315 - accuracy: 1.0000 - val_loss: 0.0799 - val_accuracy: 0.9890
    Epoch 33/150
    8635/8635 [==============================] - 1s 67us/sample - loss: 0.0286 - accuracy: 1.0000 - val_loss: 0.0767 - val_accuracy: 0.9912
    Epoch 34/150
    8635/8635 [==============================] - 1s 60us/sample - loss: 0.0263 - accuracy: 1.0000 - val_loss: 0.0704 - val_accuracy: 0.9934
    Epoch 35/150
    8635/8635 [==============================] - 1s 59us/sample - loss: 0.0240 - accuracy: 1.0000 - val_loss: 0.0646 - val_accuracy: 0.9956
    Epoch 36/150
    8635/8635 [==============================] - 1s 61us/sample - loss: 0.0219 - accuracy: 1.0000 - val_loss: 0.0617 - val_accuracy: 0.9956
    Epoch 37/150
    8635/8635 [==============================] - 1s 59us/sample - loss: 0.0201 - accuracy: 1.0000 - val_loss: 0.0580 - val_accuracy: 0.9956
    Epoch 38/150
    8635/8635 [==============================] - 1s 59us/sample - loss: 0.0185 - accuracy: 1.0000 - val_loss: 0.0549 - val_accuracy: 0.9934
    Epoch 39/150
    8635/8635 [==============================] - 1s 65us/sample - loss: 0.0171 - accuracy: 1.0000 - val_loss: 0.0517 - val_accuracy: 0.9956
    Epoch 40/150
    8635/8635 [==============================] - 1s 65us/sample - loss: 0.0158 - accuracy: 1.0000 - val_loss: 0.0494 - val_accuracy: 0.9956
    Epoch 41/150
    8635/8635 [==============================] - 1s 63us/sample - loss: 0.0146 - accuracy: 1.0000 - val_loss: 0.0469 - val_accuracy: 0.9956
    Epoch 42/150
    8635/8635 [==============================] - 1s 62us/sample - loss: 0.0136 - accuracy: 1.0000 - val_loss: 0.0426 - val_accuracy: 0.9956
    Epoch 43/150
    8635/8635 [==============================] - 1s 60us/sample - loss: 0.0125 - accuracy: 1.0000 - val_loss: 0.0395 - val_accuracy: 0.9956
    Epoch 44/150
    8635/8635 [==============================] - 1s 66us/sample - loss: 0.0117 - accuracy: 1.0000 - val_loss: 0.0383 - val_accuracy: 0.9956
    Epoch 45/150
    8635/8635 [==============================] - 1s 68us/sample - loss: 0.0109 - accuracy: 1.0000 - val_loss: 0.0371 - val_accuracy: 0.9956
    Epoch 46/150
    8635/8635 [==============================] - 1s 71us/sample - loss: 0.0102 - accuracy: 1.0000 - val_loss: 0.0352 - val_accuracy: 0.9956
    Epoch 47/150
    8635/8635 [==============================] - 1s 61us/sample - loss: 0.0094 - accuracy: 1.0000 - val_loss: 0.0338 - val_accuracy: 0.9956
    Epoch 48/150
    8635/8635 [==============================] - 1s 61us/sample - loss: 0.0088 - accuracy: 1.0000 - val_loss: 0.0313 - val_accuracy: 0.9978
    Epoch 49/150
    8635/8635 [==============================] - 1s 58us/sample - loss: 0.0083 - accuracy: 1.0000 - val_loss: 0.0300 - val_accuracy: 0.9956
    Epoch 50/150
    8635/8635 [==============================] - 0s 58us/sample - loss: 0.0077 - accuracy: 1.0000 - val_loss: 0.0283 - val_accuracy: 0.9978
    Epoch 51/150
    8635/8635 [==============================] - 1s 59us/sample - loss: 0.0073 - accuracy: 1.0000 - val_loss: 0.0275 - val_accuracy: 0.9978
    Epoch 52/150
    8635/8635 [==============================] - 0s 56us/sample - loss: 0.0068 - accuracy: 1.0000 - val_loss: 0.0257 - val_accuracy: 0.9978
    Epoch 53/150
    8635/8635 [==============================] - 0s 56us/sample - loss: 0.0064 - accuracy: 1.0000 - val_loss: 0.0245 - val_accuracy: 0.9978
    Epoch 54/150
    8635/8635 [==============================] - 0s 56us/sample - loss: 0.0060 - accuracy: 1.0000 - val_loss: 0.0237 - val_accuracy: 0.9978
    Epoch 55/150
    8635/8635 [==============================] - 1s 59us/sample - loss: 0.0057 - accuracy: 1.0000 - val_loss: 0.0229 - val_accuracy: 0.9978
    Epoch 56/150
    8635/8635 [==============================] - 1s 59us/sample - loss: 0.0054 - accuracy: 1.0000 - val_loss: 0.0221 - val_accuracy: 0.9978
    Epoch 57/150
    8635/8635 [==============================] - 1s 58us/sample - loss: 0.0051 - accuracy: 1.0000 - val_loss: 0.0204 - val_accuracy: 0.9978
    Epoch 58/150
    8635/8635 [==============================] - 1s 62us/sample - loss: 0.0048 - accuracy: 1.0000 - val_loss: 0.0194 - val_accuracy: 0.9978
    Epoch 59/150
    8635/8635 [==============================] - 1s 61us/sample - loss: 0.0045 - accuracy: 1.0000 - val_loss: 0.0192 - val_accuracy: 0.9978
    Epoch 60/150
    8635/8635 [==============================] - 1s 62us/sample - loss: 0.0043 - accuracy: 1.0000 - val_loss: 0.0181 - val_accuracy: 0.9978
    Epoch 61/150
    8635/8635 [==============================] - 1s 59us/sample - loss: 0.0040 - accuracy: 1.0000 - val_loss: 0.0172 - val_accuracy: 1.0000
    Epoch 62/150
    8635/8635 [==============================] - 1s 62us/sample - loss: 0.0038 - accuracy: 1.0000 - val_loss: 0.0169 - val_accuracy: 0.9978
    Epoch 63/150
    8635/8635 [==============================] - 0s 56us/sample - loss: 0.0036 - accuracy: 1.0000 - val_loss: 0.0164 - val_accuracy: 0.9978
    Epoch 64/150
    8635/8635 [==============================] - 0s 56us/sample - loss: 0.0034 - accuracy: 1.0000 - val_loss: 0.0156 - val_accuracy: 0.9978
    Epoch 65/150
    8635/8635 [==============================] - 0s 57us/sample - loss: 0.0032 - accuracy: 1.0000 - val_loss: 0.0152 - val_accuracy: 1.0000
    Epoch 66/150
    8635/8635 [==============================] - 0s 56us/sample - loss: 0.0031 - accuracy: 1.0000 - val_loss: 0.0146 - val_accuracy: 1.0000
    Epoch 67/150
    8635/8635 [==============================] - 1s 58us/sample - loss: 0.0029 - accuracy: 1.0000 - val_loss: 0.0134 - val_accuracy: 1.0000
    Epoch 68/150
    8635/8635 [==============================] - 1s 59us/sample - loss: 0.0028 - accuracy: 1.0000 - val_loss: 0.0131 - val_accuracy: 1.0000
    Epoch 69/150
    8635/8635 [==============================] - 1s 61us/sample - loss: 0.0026 - accuracy: 1.0000 - val_loss: 0.0128 - val_accuracy: 1.0000
    Epoch 70/150
    8635/8635 [==============================] - 0s 57us/sample - loss: 0.0025 - accuracy: 1.0000 - val_loss: 0.0122 - val_accuracy: 1.0000
    Epoch 71/150
    8635/8635 [==============================] - 0s 58us/sample - loss: 0.0024 - accuracy: 1.0000 - val_loss: 0.0124 - val_accuracy: 0.9978
    Epoch 72/150
    8635/8635 [==============================] - 0s 57us/sample - loss: 0.0022 - accuracy: 1.0000 - val_loss: 0.0118 - val_accuracy: 1.0000
    Epoch 73/150
    8635/8635 [==============================] - 1s 59us/sample - loss: 0.0021 - accuracy: 1.0000 - val_loss: 0.0110 - val_accuracy: 1.0000
    Epoch 74/150
    8635/8635 [==============================] - 1s 58us/sample - loss: 0.0020 - accuracy: 1.0000 - val_loss: 0.0109 - val_accuracy: 1.0000
    Epoch 75/150
    8635/8635 [==============================] - 1s 59us/sample - loss: 0.0019 - accuracy: 1.0000 - val_loss: 0.0099 - val_accuracy: 1.0000
    Epoch 76/150
    8635/8635 [==============================] - 1s 58us/sample - loss: 0.0018 - accuracy: 1.0000 - val_loss: 0.0100 - val_accuracy: 1.0000
    Epoch 77/150
    8635/8635 [==============================] - 0s 58us/sample - loss: 0.0018 - accuracy: 1.0000 - val_loss: 0.0095 - val_accuracy: 1.0000
    Epoch 78/150
    8635/8635 [==============================] - 1s 58us/sample - loss: 0.0017 - accuracy: 1.0000 - val_loss: 0.0096 - val_accuracy: 1.0000
    Epoch 79/150
    8635/8635 [==============================] - 0s 58us/sample - loss: 0.0016 - accuracy: 1.0000 - val_loss: 0.0096 - val_accuracy: 1.0000
    Epoch 80/150
    8635/8635 [==============================] - 0s 57us/sample - loss: 0.0015 - accuracy: 1.0000 - val_loss: 0.0092 - val_accuracy: 1.0000
    Epoch 81/150
    8635/8635 [==============================] - 1s 58us/sample - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.0081 - val_accuracy: 1.0000
    Epoch 82/150
    8635/8635 [==============================] - 0s 56us/sample - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.0082 - val_accuracy: 1.0000
    Epoch 83/150
    8635/8635 [==============================] - 0s 58us/sample - loss: 0.0013 - accuracy: 1.0000 - val_loss: 0.0082 - val_accuracy: 1.0000
    Epoch 84/150
    8635/8635 [==============================] - 1s 59us/sample - loss: 0.0012 - accuracy: 1.0000 - val_loss: 0.0078 - val_accuracy: 1.0000
    Epoch 85/150
    8635/8635 [==============================] - 0s 58us/sample - loss: 0.0012 - accuracy: 1.0000 - val_loss: 0.0076 - val_accuracy: 1.0000
    Epoch 86/150
    8635/8635 [==============================] - 0s 58us/sample - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.0074 - val_accuracy: 1.0000
    Epoch 87/150
    8635/8635 [==============================] - 1s 60us/sample - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.0069 - val_accuracy: 1.0000
    Epoch 88/150
    8635/8635 [==============================] - 1s 62us/sample - loss: 0.0010 - accuracy: 1.0000 - val_loss: 0.0067 - val_accuracy: 1.0000
    Epoch 89/150
    8635/8635 [==============================] - 0s 58us/sample - loss: 9.8558e-04 - accuracy: 1.0000 - val_loss: 0.0067 - val_accuracy: 1.0000
    Epoch 90/150
    8635/8635 [==============================] - 1s 59us/sample - loss: 9.4302e-04 - accuracy: 1.0000 - val_loss: 0.0065 - val_accuracy: 1.0000
    Epoch 91/150
    8635/8635 [==============================] - 1s 58us/sample - loss: 8.9881e-04 - accuracy: 1.0000 - val_loss: 0.0063 - val_accuracy: 1.0000
    Epoch 92/150
    8635/8635 [==============================] - 1s 60us/sample - loss: 8.5815e-04 - accuracy: 1.0000 - val_loss: 0.0061 - val_accuracy: 1.0000
    Epoch 93/150
    8635/8635 [==============================] - 0s 57us/sample - loss: 8.1993e-04 - accuracy: 1.0000 - val_loss: 0.0057 - val_accuracy: 1.0000
    Epoch 94/150
    8635/8635 [==============================] - 1s 58us/sample - loss: 7.8299e-04 - accuracy: 1.0000 - val_loss: 0.0056 - val_accuracy: 1.0000
    Epoch 95/150
    8635/8635 [==============================] - 0s 56us/sample - loss: 7.4757e-04 - accuracy: 1.0000 - val_loss: 0.0053 - val_accuracy: 1.0000
    Epoch 96/150
    8635/8635 [==============================] - 1s 59us/sample - loss: 7.1597e-04 - accuracy: 1.0000 - val_loss: 0.0053 - val_accuracy: 1.0000
    Epoch 97/150
    8635/8635 [==============================] - 0s 57us/sample - loss: 6.8339e-04 - accuracy: 1.0000 - val_loss: 0.0052 - val_accuracy: 1.0000
    Epoch 98/150
    8635/8635 [==============================] - 1s 59us/sample - loss: 6.5364e-04 - accuracy: 1.0000 - val_loss: 0.0051 - val_accuracy: 1.0000
    Epoch 99/150
    8635/8635 [==============================] - 1s 58us/sample - loss: 6.2444e-04 - accuracy: 1.0000 - val_loss: 0.0051 - val_accuracy: 1.0000
    Epoch 100/150
    8635/8635 [==============================] - 1s 60us/sample - loss: 5.9848e-04 - accuracy: 1.0000 - val_loss: 0.0053 - val_accuracy: 1.0000
    Epoch 101/150
    8635/8635 [==============================] - 0s 58us/sample - loss: 5.7203e-04 - accuracy: 1.0000 - val_loss: 0.0048 - val_accuracy: 1.0000
    Epoch 102/150
    8635/8635 [==============================] - 1s 65us/sample - loss: 5.4724e-04 - accuracy: 1.0000 - val_loss: 0.0045 - val_accuracy: 1.0000
    Epoch 103/150
    8635/8635 [==============================] - 1s 68us/sample - loss: 5.2431e-04 - accuracy: 1.0000 - val_loss: 0.0044 - val_accuracy: 1.0000
    Epoch 104/150
    8635/8635 [==============================] - 1s 59us/sample - loss: 5.0180e-04 - accuracy: 1.0000 - val_loss: 0.0043 - val_accuracy: 1.0000
    Epoch 105/150
    8635/8635 [==============================] - 1s 59us/sample - loss: 4.8044e-04 - accuracy: 1.0000 - val_loss: 0.0041 - val_accuracy: 1.0000
    Epoch 106/150
    8635/8635 [==============================] - 0s 57us/sample - loss: 4.5914e-04 - accuracy: 1.0000 - val_loss: 0.0040 - val_accuracy: 1.0000
    Epoch 107/150
    8635/8635 [==============================] - 1s 59us/sample - loss: 4.4048e-04 - accuracy: 1.0000 - val_loss: 0.0040 - val_accuracy: 1.0000
    Epoch 108/150
    8635/8635 [==============================] - 1s 67us/sample - loss: 4.2111e-04 - accuracy: 1.0000 - val_loss: 0.0037 - val_accuracy: 1.0000
    Epoch 109/150
    8635/8635 [==============================] - 1s 66us/sample - loss: 4.0344e-04 - accuracy: 1.0000 - val_loss: 0.0036 - val_accuracy: 1.0000
    Epoch 110/150
    8635/8635 [==============================] - 1s 65us/sample - loss: 3.8637e-04 - accuracy: 1.0000 - val_loss: 0.0037 - val_accuracy: 1.0000
    Epoch 111/150
    8635/8635 [==============================] - 1s 62us/sample - loss: 3.7003e-04 - accuracy: 1.0000 - val_loss: 0.0036 - val_accuracy: 1.0000



{% highlight python %}
# plot the original data, the previous set of 11 sized texts and the last one, with twice more data
plt.plot(hist10.history['val_accuracy'], label='Original data set')
plt.plot(hist10_with_size_11.history['val_accuracy'], label='Subset of texts with len 11')
plt.plot(hist10_with_size_11_2x.history['val_accuracy'], label='Subset of texts with len 11, 2X data')
plt.title('Accuracy')
plt.ylabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="best")
plt.show()

plt.plot(hist10.history['val_accuracy'][-20:], label='Original data set')
plt.plot(hist10_with_size_11.history['val_accuracy'][-20:], label='Subset of texts with len 11')
plt.plot(hist10_with_size_11_2x.history['val_accuracy'][-20:], label='Subset of texts with len 11, 2X data')
plt.title('Accuracy, last 20 epochs')
plt.ylabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="best")
plt.show()

plt.plot(hist10.history['val_loss'], label='Original data set')
plt.plot(hist10_with_size_11.history['val_loss'], label='Subset of texts with len 11')
plt.plot(hist10_with_size_11_2x.history['val_loss'], label='Subset of texts with len 11, 2X data')
plt.title('Loss')
plt.ylabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="best")
plt.show()

plt.plot(hist10.history['val_loss'][-20:], label='Original data set')
plt.plot(hist10_with_size_11.history['val_loss'][-20:], label='Subset of texts with len 11')
plt.plot(hist10_with_size_11_2x.history['val_loss'][-20:], label='Subset of texts with len 11, 2X data')
plt.title('Loss, last 20 epochs')
plt.ylabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="best")
plt.show()
{% endhighlight %}


![png](/assets/qr-nn/output_49_0.png)



![png](/assets/qr-nn/output_49_1.png)



![png](/assets/qr-nn/output_49_2.png)



![png](/assets/qr-nn/output_49_3.png)


The performance of the model with twice more data is very good. Time to try a different network design.

### 4.2 Better NN design

The original network has a single hidden layer. The NN may not be flexible enough to recognize all classes correctly. Let's try to add one more hidden layer and train the network bases on the original data set.


{% highlight python %}
# a new model with extra hidden layer
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(21, 21)),
    keras.layers.Dense(21*21, activation='relu'),
    keras.layers.Dense(21*21, activation='relu'), # extra layer
    keras.layers.Dense(len(ALL_CHAR_CLASSES))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

hist10_deep = model.fit(
        training_data, np.asarray(training_labels_char10),
        epochs=150, batch_size=128,
        validation_split=0.05,
        callbacks=[EarlyStopping(monitor='val_loss' , min_delta=0.0001, patience=3)])
{% endhighlight %}

    Train on 47500 samples, validate on 2500 samples
    Epoch 1/150
    47500/47500 [==============================] - 4s 81us/sample - loss: 0.3791 - accuracy: 0.9089 - val_loss: 0.2756 - val_accuracy: 0.9184
    Epoch 2/150
    47500/47500 [==============================] - 3s 72us/sample - loss: 0.2769 - accuracy: 0.9172 - val_loss: 0.2327 - val_accuracy: 0.9264
    Epoch 3/150
    47500/47500 [==============================] - 4s 82us/sample - loss: 0.2136 - accuracy: 0.9278 - val_loss: 0.1631 - val_accuracy: 0.9400
    Epoch 4/150
    47500/47500 [==============================] - 4s 77us/sample - loss: 0.1277 - accuracy: 0.9526 - val_loss: 0.0929 - val_accuracy: 0.9612
    Epoch 5/150
    47500/47500 [==============================] - 4s 82us/sample - loss: 0.0586 - accuracy: 0.9799 - val_loss: 0.0290 - val_accuracy: 0.9944
    Epoch 6/150
    47500/47500 [==============================] - 4s 83us/sample - loss: 0.0131 - accuracy: 0.9983 - val_loss: 0.0410 - val_accuracy: 0.9920
    Epoch 7/150
    47500/47500 [==============================] - 4s 80us/sample - loss: 0.0044 - accuracy: 0.9996 - val_loss: 0.0019 - val_accuracy: 1.0000
    Epoch 8/150
    47500/47500 [==============================] - 4s 80us/sample - loss: 0.0010 - accuracy: 1.0000 - val_loss: 0.0014 - val_accuracy: 1.0000
    Epoch 9/150
    47500/47500 [==============================] - 4s 84us/sample - loss: 6.1877e-04 - accuracy: 1.0000 - val_loss: 9.1512e-04 - val_accuracy: 1.0000
    Epoch 10/150
    47500/47500 [==============================] - 4s 82us/sample - loss: 3.9826e-04 - accuracy: 1.0000 - val_loss: 6.0030e-04 - val_accuracy: 1.0000
    Epoch 11/150
    47500/47500 [==============================] - 4s 81us/sample - loss: 2.7424e-04 - accuracy: 1.0000 - val_loss: 4.3338e-04 - val_accuracy: 1.0000
    Epoch 12/150
    47500/47500 [==============================] - 4s 81us/sample - loss: 1.9417e-04 - accuracy: 1.0000 - val_loss: 3.8792e-04 - val_accuracy: 1.0000
    Epoch 13/150
    47500/47500 [==============================] - 4s 81us/sample - loss: 1.4257e-04 - accuracy: 1.0000 - val_loss: 2.6783e-04 - val_accuracy: 1.0000
    Epoch 14/150
    47500/47500 [==============================] - 4s 79us/sample - loss: 1.0522e-04 - accuracy: 1.0000 - val_loss: 1.9728e-04 - val_accuracy: 1.0000
    Epoch 15/150
    47500/47500 [==============================] - 4s 80us/sample - loss: 7.9755e-05 - accuracy: 1.0000 - val_loss: 1.4277e-04 - val_accuracy: 1.0000
    Epoch 16/150
    47500/47500 [==============================] - 4s 80us/sample - loss: 6.1159e-05 - accuracy: 1.0000 - val_loss: 1.6992e-04 - val_accuracy: 1.0000
    Epoch 17/150
    47500/47500 [==============================] - 4s 80us/sample - loss: 4.6737e-05 - accuracy: 1.0000 - val_loss: 9.8789e-05 - val_accuracy: 1.0000
    Epoch 18/150
    47500/47500 [==============================] - 4s 80us/sample - loss: 3.6435e-05 - accuracy: 1.0000 - val_loss: 9.8796e-05 - val_accuracy: 1.0000



{% highlight python %}
plt.plot(hist0.history['val_accuracy'], label='Char 0')
plt.plot(hist10_deep.history['val_accuracy'], label='Char 10')
plt.title('Accuracy')
plt.ylabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="best")
plt.show()

plt.plot(hist0.history['val_loss'], label='Char 0')
plt.plot(hist10_deep.history['val_loss'], label='Char 10')
plt.title('Loss')
plt.ylabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="best")
plt.show()
{% endhighlight %}


![png](/assets/qr-nn/output_52_0.png)



![png](/assets/qr-nn/output_52_1.png)


Outstanding result. In fact, the new design for the last char performs as good as the old one for the first char.

## 4. Build a set of NNs to decode QR codes

The goal for this part is to build 11 NNs. One for each position. For simplicity, the double hidden layer design is applied for every character.


{% highlight python %}
# function to create and train 11 NNs, one per positions
# output is a set of models and train histories
def make_model_per_position(train, labels, epochs=150):
    models_per_position = []
    histories = []
    for position in range(11):
        print("Training for position:", position)
        train_labels_chars = make_labels_for_position(labels, position) # generate labels for specific position
        model = keras.Sequential([
                                keras.layers.Flatten(input_shape=(21, 21)),
                                keras.layers.Dense(21*21, activation='relu'),
                                keras.layers.Dense(21*21, activation='relu'),
                                keras.layers.Dense(len(ALL_CHAR_CLASSES))
                                ])
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        history = train_model(model, train, train_labels_chars, epochs)
        models_per_position.append(keras.Sequential([model, keras.layers.Softmax()])) # each trained model is augmented with  a softmax layer for easier results interpretation
        histories.append(history)
    return (models_per_position, histories)

# creating 11 NNs using original 50k dataset
(models_per_position, histories) = make_model_per_position(training_data, training_labels)
{% endhighlight %}

    Training for position: 0
    Train on 47500 samples, validate on 2500 samples
    Epoch 1/150
    47500/47500 [==============================] - 4s 79us/sample - loss: 2.1614 - accuracy: 0.3376 - val_loss: 0.3508 - val_accuracy: 0.9200
    Epoch 2/150
    47500/47500 [==============================] - 3s 70us/sample - loss: 0.0541 - accuracy: 0.9957 - val_loss: 0.0078 - val_accuracy: 1.0000
    Epoch 3/150
    47500/47500 [==============================] - 4s 81us/sample - loss: 0.0044 - accuracy: 1.0000 - val_loss: 0.0027 - val_accuracy: 1.0000
    Epoch 4/150
    47500/47500 [==============================] - 4s 82us/sample - loss: 0.0018 - accuracy: 1.0000 - val_loss: 0.0014 - val_accuracy: 1.0000
    Epoch 5/150
    47500/47500 [==============================] - 4s 82us/sample - loss: 9.9950e-04 - accuracy: 1.0000 - val_loss: 8.0182e-04 - val_accuracy: 1.0000
    Epoch 6/150
    47500/47500 [==============================] - 4s 77us/sample - loss: 6.0850e-04 - accuracy: 1.0000 - val_loss: 5.1275e-04 - val_accuracy: 1.0000
    Epoch 7/150
    47500/47500 [==============================] - 4s 83us/sample - loss: 4.0360e-04 - accuracy: 1.0000 - val_loss: 3.4617e-04 - val_accuracy: 1.0000
    Epoch 8/150
    47500/47500 [==============================] - 4s 78us/sample - loss: 2.7971e-04 - accuracy: 1.0000 - val_loss: 2.4629e-04 - val_accuracy: 1.0000
    Epoch 9/150
    47500/47500 [==============================] - 4s 80us/sample - loss: 2.0063e-04 - accuracy: 1.0000 - val_loss: 1.8029e-04 - val_accuracy: 1.0000
    Epoch 10/150
    47500/47500 [==============================] - 4s 82us/sample - loss: 1.4767e-04 - accuracy: 1.0000 - val_loss: 1.3338e-04 - val_accuracy: 1.0000
    Epoch 11/150
    47500/47500 [==============================] - 4s 78us/sample - loss: 1.1045e-04 - accuracy: 1.0000 - val_loss: 1.0356e-04 - val_accuracy: 1.0000
    Epoch 12/150
    47500/47500 [==============================] - 4s 80us/sample - loss: 8.4273e-05 - accuracy: 1.0000 - val_loss: 7.7155e-05 - val_accuracy: 1.0000
    Epoch 13/150
    47500/47500 [==============================] - 4s 80us/sample - loss: 6.4645e-05 - accuracy: 1.0000 - val_loss: 5.9816e-05 - val_accuracy: 1.0000
    Epoch 14/150
    47500/47500 [==============================] - 4s 78us/sample - loss: 5.0262e-05 - accuracy: 1.0000 - val_loss: 4.6618e-05 - val_accuracy: 1.0000
    Epoch 15/150
    47500/47500 [==============================] - 4s 78us/sample - loss: 3.9239e-05 - accuracy: 1.0000 - val_loss: 3.6798e-05 - val_accuracy: 1.0000
    Training for position: 1
    Train on 47500 samples, validate on 2500 samples
    Epoch 1/150
    47500/47500 [==============================] - 4s 94us/sample - loss: 1.7563 - accuracy: 0.5060 - val_loss: 0.1256 - val_accuracy: 0.9944
    Epoch 2/150
    47500/47500 [==============================] - 4s 88us/sample - loss: 0.0263 - accuracy: 0.9997 - val_loss: 0.0070 - val_accuracy: 1.0000
    Epoch 3/150
    47500/47500 [==============================] - 4s 84us/sample - loss: 0.0040 - accuracy: 1.0000 - val_loss: 0.0025 - val_accuracy: 1.0000
    Epoch 4/150
    47500/47500 [==============================] - 4s 81us/sample - loss: 0.0017 - accuracy: 1.0000 - val_loss: 0.0013 - val_accuracy: 1.0000
    Epoch 5/150
    47500/47500 [==============================] - 4s 88us/sample - loss: 9.0345e-04 - accuracy: 1.0000 - val_loss: 7.2286e-04 - val_accuracy: 1.0000
    Epoch 6/150
    47500/47500 [==============================] - 4s 80us/sample - loss: 5.4534e-04 - accuracy: 1.0000 - val_loss: 4.5457e-04 - val_accuracy: 1.0000
    Epoch 7/150
    47500/47500 [==============================] - 4s 83us/sample - loss: 3.5370e-04 - accuracy: 1.0000 - val_loss: 3.0630e-04 - val_accuracy: 1.0000
    Epoch 8/150
    47500/47500 [==============================] - 4s 81us/sample - loss: 2.4187e-04 - accuracy: 1.0000 - val_loss: 2.1389e-04 - val_accuracy: 1.0000
    Epoch 9/150
    47500/47500 [==============================] - 4s 80us/sample - loss: 1.7154e-04 - accuracy: 1.0000 - val_loss: 1.5400e-04 - val_accuracy: 1.0000
    Epoch 10/150
    47500/47500 [==============================] - 4s 81us/sample - loss: 1.2506e-04 - accuracy: 1.0000 - val_loss: 1.1372e-04 - val_accuracy: 1.0000
    Epoch 11/150
    47500/47500 [==============================] - 4s 78us/sample - loss: 9.3141e-05 - accuracy: 1.0000 - val_loss: 8.5570e-05 - val_accuracy: 1.0000
    Epoch 12/150
    47500/47500 [==============================] - 4s 80us/sample - loss: 7.0489e-05 - accuracy: 1.0000 - val_loss: 6.5685e-05 - val_accuracy: 1.0000
    Training for position: 2
    Train on 47500 samples, validate on 2500 samples
    Epoch 1/150
    47500/47500 [==============================] - 4s 89us/sample - loss: 2.1706 - accuracy: 0.3623 - val_loss: 0.3844 - val_accuracy: 0.9660
    Epoch 2/150
    47500/47500 [==============================] - 4s 78us/sample - loss: 0.0521 - accuracy: 0.9985 - val_loss: 0.0076 - val_accuracy: 1.0000
    Epoch 3/150
    47500/47500 [==============================] - 4s 77us/sample - loss: 0.0043 - accuracy: 1.0000 - val_loss: 0.0026 - val_accuracy: 1.0000
    Epoch 4/150
    47500/47500 [==============================] - 4s 81us/sample - loss: 0.0017 - accuracy: 1.0000 - val_loss: 0.0013 - val_accuracy: 1.0000
    Epoch 5/150
    47500/47500 [==============================] - 4s 79us/sample - loss: 9.2805e-04 - accuracy: 1.0000 - val_loss: 7.2975e-04 - val_accuracy: 1.0000
    Epoch 6/150
    47500/47500 [==============================] - 4s 80us/sample - loss: 5.6367e-04 - accuracy: 1.0000 - val_loss: 4.6482e-04 - val_accuracy: 1.0000
    Epoch 7/150
    47500/47500 [==============================] - 4s 78us/sample - loss: 3.6875e-04 - accuracy: 1.0000 - val_loss: 3.1296e-04 - val_accuracy: 1.0000
    Epoch 8/150
    47500/47500 [==============================] - 4s 84us/sample - loss: 2.5332e-04 - accuracy: 1.0000 - val_loss: 2.1871e-04 - val_accuracy: 1.0000
    Epoch 9/150
    47500/47500 [==============================] - 4s 88us/sample - loss: 1.8008e-04 - accuracy: 1.0000 - val_loss: 1.5877e-04 - val_accuracy: 1.0000
    Epoch 10/150
    47500/47500 [==============================] - 4s 79us/sample - loss: 1.3123e-04 - accuracy: 1.0000 - val_loss: 1.1699e-04 - val_accuracy: 1.0000
    Epoch 11/150
    47500/47500 [==============================] - 4s 86us/sample - loss: 9.7421e-05 - accuracy: 1.0000 - val_loss: 8.7502e-05 - val_accuracy: 1.0000
    Epoch 12/150
    47500/47500 [==============================] - 4s 78us/sample - loss: 7.3448e-05 - accuracy: 1.0000 - val_loss: 6.6138e-05 - val_accuracy: 1.0000
    Training for position: 3
    Train on 47500 samples, validate on 2500 samples
    Epoch 1/150
    47500/47500 [==============================] - 4s 92us/sample - loss: 1.9258 - accuracy: 0.3888 - val_loss: 1.0133 - val_accuracy: 0.6112
    Epoch 2/150
    47500/47500 [==============================] - 4s 89us/sample - loss: 0.2257 - accuracy: 0.9491 - val_loss: 0.0154 - val_accuracy: 1.0000
    Epoch 3/150
    47500/47500 [==============================] - 4s 87us/sample - loss: 0.0076 - accuracy: 1.0000 - val_loss: 0.0042 - val_accuracy: 1.0000
    Epoch 4/150
    47500/47500 [==============================] - 4s 90us/sample - loss: 0.0027 - accuracy: 1.0000 - val_loss: 0.0020 - val_accuracy: 1.0000
    Epoch 5/150
    47500/47500 [==============================] - 4s 88us/sample - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.0011 - val_accuracy: 1.0000
    Epoch 6/150
    47500/47500 [==============================] - 4s 80us/sample - loss: 8.3010e-04 - accuracy: 1.0000 - val_loss: 7.0213e-04 - val_accuracy: 1.0000
    Epoch 7/150
    47500/47500 [==============================] - 4s 83us/sample - loss: 5.3635e-04 - accuracy: 1.0000 - val_loss: 4.7197e-04 - val_accuracy: 1.0000
    Epoch 8/150
    47500/47500 [==============================] - 4s 84us/sample - loss: 3.6697e-04 - accuracy: 1.0000 - val_loss: 3.2854e-04 - val_accuracy: 1.0000
    Epoch 9/150
    47500/47500 [==============================] - 4s 87us/sample - loss: 2.5955e-04 - accuracy: 1.0000 - val_loss: 2.3703e-04 - val_accuracy: 1.0000
    Epoch 10/150
    47500/47500 [==============================] - 4s 82us/sample - loss: 1.8888e-04 - accuracy: 1.0000 - val_loss: 1.7813e-04 - val_accuracy: 1.0000
    Epoch 11/150
    47500/47500 [==============================] - 4s 88us/sample - loss: 1.4049e-04 - accuracy: 1.0000 - val_loss: 1.3249e-04 - val_accuracy: 1.0000
    Epoch 12/150
    47500/47500 [==============================] - 4s 87us/sample - loss: 1.0633e-04 - accuracy: 1.0000 - val_loss: 1.0113e-04 - val_accuracy: 1.0000
    Epoch 13/150
    47500/47500 [==============================] - 4s 84us/sample - loss: 8.0983e-05 - accuracy: 1.0000 - val_loss: 7.8894e-05 - val_accuracy: 1.0000
    Training for position: 4
    Train on 47500 samples, validate on 2500 samples
    Epoch 1/150
    47500/47500 [==============================] - 5s 105us/sample - loss: 1.8853 - accuracy: 0.4458 - val_loss: 0.8823 - val_accuracy: 0.6884
    Epoch 2/150
    47500/47500 [==============================] - 5s 96us/sample - loss: 0.1616 - accuracy: 0.9656 - val_loss: 0.0104 - val_accuracy: 1.0000
    Epoch 3/150
    47500/47500 [==============================] - 4s 93us/sample - loss: 0.0054 - accuracy: 1.0000 - val_loss: 0.0032 - val_accuracy: 1.0000
    Epoch 4/150
    47500/47500 [==============================] - 4s 82us/sample - loss: 0.0020 - accuracy: 1.0000 - val_loss: 0.0015 - val_accuracy: 1.0000
    Epoch 5/150
    47500/47500 [==============================] - 4s 84us/sample - loss: 0.0011 - accuracy: 1.0000 - val_loss: 8.7310e-04 - val_accuracy: 1.0000
    Epoch 6/150
    47500/47500 [==============================] - 4s 82us/sample - loss: 6.3733e-04 - accuracy: 1.0000 - val_loss: 5.5097e-04 - val_accuracy: 1.0000
    Epoch 7/150
    47500/47500 [==============================] - 4s 85us/sample - loss: 4.1259e-04 - accuracy: 1.0000 - val_loss: 3.7473e-04 - val_accuracy: 1.0000
    Epoch 8/150
    47500/47500 [==============================] - 4s 84us/sample - loss: 2.8192e-04 - accuracy: 1.0000 - val_loss: 2.6267e-04 - val_accuracy: 1.0000
    Epoch 9/150
    47500/47500 [==============================] - 4s 81us/sample - loss: 1.9983e-04 - accuracy: 1.0000 - val_loss: 1.9219e-04 - val_accuracy: 1.0000
    Epoch 10/150
    47500/47500 [==============================] - 4s 77us/sample - loss: 1.4552e-04 - accuracy: 1.0000 - val_loss: 1.4196e-04 - val_accuracy: 1.0000
    Epoch 11/150
    47500/47500 [==============================] - 4s 81us/sample - loss: 1.0801e-04 - accuracy: 1.0000 - val_loss: 1.1229e-04 - val_accuracy: 1.0000
    Epoch 12/150
    47500/47500 [==============================] - 4s 86us/sample - loss: 8.1355e-05 - accuracy: 1.0000 - val_loss: 8.4408e-05 - val_accuracy: 1.0000
    Epoch 13/150
    47500/47500 [==============================] - 4s 89us/sample - loss: 6.2108e-05 - accuracy: 1.0000 - val_loss: 6.3716e-05 - val_accuracy: 1.0000
    Training for position: 5
    Train on 47500 samples, validate on 2500 samples
    Epoch 1/150
    47500/47500 [==============================] - 5s 97us/sample - loss: 1.7461 - accuracy: 0.4984 - val_loss: 1.1646 - val_accuracy: 0.5784
    Epoch 2/150
    47500/47500 [==============================] - 4s 85us/sample - loss: 0.4023 - accuracy: 0.8809 - val_loss: 0.0177 - val_accuracy: 1.0000
    Epoch 3/150
    47500/47500 [==============================] - 4s 89us/sample - loss: 0.0077 - accuracy: 1.0000 - val_loss: 0.0040 - val_accuracy: 1.0000
    Epoch 4/150
    47500/47500 [==============================] - 4s 83us/sample - loss: 0.0024 - accuracy: 1.0000 - val_loss: 0.0017 - val_accuracy: 1.0000
    Epoch 5/150
    47500/47500 [==============================] - 4s 86us/sample - loss: 0.0012 - accuracy: 1.0000 - val_loss: 9.6495e-04 - val_accuracy: 1.0000
    Epoch 6/150
    47500/47500 [==============================] - 4s 82us/sample - loss: 6.9643e-04 - accuracy: 1.0000 - val_loss: 5.9882e-04 - val_accuracy: 1.0000
    Epoch 7/150
    47500/47500 [==============================] - 4s 86us/sample - loss: 4.4290e-04 - accuracy: 1.0000 - val_loss: 3.9365e-04 - val_accuracy: 1.0000
    Epoch 8/150
    47500/47500 [==============================] - 4s 87us/sample - loss: 2.9888e-04 - accuracy: 1.0000 - val_loss: 2.7340e-04 - val_accuracy: 1.0000
    Epoch 9/150
    47500/47500 [==============================] - 4s 81us/sample - loss: 2.0967e-04 - accuracy: 1.0000 - val_loss: 1.9435e-04 - val_accuracy: 1.0000
    Epoch 10/150
    47500/47500 [==============================] - 4s 84us/sample - loss: 1.5149e-04 - accuracy: 1.0000 - val_loss: 1.4222e-04 - val_accuracy: 1.0000
    Epoch 11/150
    47500/47500 [==============================] - 4s 80us/sample - loss: 1.1183e-04 - accuracy: 1.0000 - val_loss: 1.0723e-04 - val_accuracy: 1.0000
    Epoch 12/150
    47500/47500 [==============================] - 4s 85us/sample - loss: 8.3954e-05 - accuracy: 1.0000 - val_loss: 8.0602e-05 - val_accuracy: 1.0000
    Epoch 13/150
    47500/47500 [==============================] - 4s 91us/sample - loss: 6.3884e-05 - accuracy: 1.0000 - val_loss: 6.2407e-05 - val_accuracy: 1.0000
    Training for position: 6
    Train on 47500 samples, validate on 2500 samples
    Epoch 1/150
    47500/47500 [==============================] - 4s 95us/sample - loss: 1.5618 - accuracy: 0.5662 - val_loss: 1.2588 - val_accuracy: 0.6124
    Epoch 2/150
    47500/47500 [==============================] - 4s 85us/sample - loss: 0.5565 - accuracy: 0.8405 - val_loss: 0.0295 - val_accuracy: 1.0000
    Epoch 3/150
    47500/47500 [==============================] - 4s 86us/sample - loss: 0.0112 - accuracy: 1.0000 - val_loss: 0.0048 - val_accuracy: 1.0000
    Epoch 4/150
    47500/47500 [==============================] - 4s 91us/sample - loss: 0.0030 - accuracy: 1.0000 - val_loss: 0.0020 - val_accuracy: 1.0000
    Epoch 5/150
    47500/47500 [==============================] - 4s 87us/sample - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.0011 - val_accuracy: 1.0000
    Epoch 6/150
    47500/47500 [==============================] - 4s 81us/sample - loss: 8.0396e-04 - accuracy: 1.0000 - val_loss: 6.6578e-04 - val_accuracy: 1.0000
    Epoch 7/150
    47500/47500 [==============================] - 4s 82us/sample - loss: 5.0718e-04 - accuracy: 1.0000 - val_loss: 4.3710e-04 - val_accuracy: 1.0000
    Epoch 8/150
    47500/47500 [==============================] - 4s 84us/sample - loss: 3.4046e-04 - accuracy: 1.0000 - val_loss: 3.0530e-04 - val_accuracy: 1.0000
    Epoch 9/150
    47500/47500 [==============================] - 4s 88us/sample - loss: 2.3848e-04 - accuracy: 1.0000 - val_loss: 2.1882e-04 - val_accuracy: 1.0000
    Epoch 10/150
    47500/47500 [==============================] - 4s 84us/sample - loss: 1.7240e-04 - accuracy: 1.0000 - val_loss: 1.5890e-04 - val_accuracy: 1.0000
    Epoch 11/150
    47500/47500 [==============================] - 4s 84us/sample - loss: 1.2726e-04 - accuracy: 1.0000 - val_loss: 1.2122e-04 - val_accuracy: 1.0000
    Epoch 12/150
    47500/47500 [==============================] - 4s 83us/sample - loss: 9.5447e-05 - accuracy: 1.0000 - val_loss: 9.0965e-05 - val_accuracy: 1.0000
    Epoch 13/150
    47500/47500 [==============================] - 4s 90us/sample - loss: 7.2583e-05 - accuracy: 1.0000 - val_loss: 7.0279e-05 - val_accuracy: 1.0000
    Training for position: 7
    Train on 47500 samples, validate on 2500 samples
    Epoch 1/150
    47500/47500 [==============================] - 5s 101us/sample - loss: 1.2598 - accuracy: 0.6543 - val_loss: 0.9875 - val_accuracy: 0.6940
    Epoch 2/150
    47500/47500 [==============================] - 4s 83us/sample - loss: 0.5290 - accuracy: 0.8304 - val_loss: 0.0681 - val_accuracy: 0.9984
    Epoch 3/150
    47500/47500 [==============================] - 4s 90us/sample - loss: 0.0179 - accuracy: 1.0000 - val_loss: 0.0055 - val_accuracy: 1.0000
    Epoch 4/150
    47500/47500 [==============================] - 4s 92us/sample - loss: 0.0033 - accuracy: 1.0000 - val_loss: 0.0021 - val_accuracy: 1.0000
    Epoch 5/150
    47500/47500 [==============================] - 4s 83us/sample - loss: 0.0015 - accuracy: 1.0000 - val_loss: 0.0011 - val_accuracy: 1.0000
    Epoch 6/150
    47500/47500 [==============================] - 4s 86us/sample - loss: 8.1862e-04 - accuracy: 1.0000 - val_loss: 6.8945e-04 - val_accuracy: 1.0000
    Epoch 7/150
    47500/47500 [==============================] - 4s 84us/sample - loss: 5.1239e-04 - accuracy: 1.0000 - val_loss: 4.5056e-04 - val_accuracy: 1.0000
    Epoch 8/150
    47500/47500 [==============================] - 4s 86us/sample - loss: 3.4253e-04 - accuracy: 1.0000 - val_loss: 3.0583e-04 - val_accuracy: 1.0000
    Epoch 9/150
    47500/47500 [==============================] - 4s 84us/sample - loss: 2.3808e-04 - accuracy: 1.0000 - val_loss: 2.1453e-04 - val_accuracy: 1.0000
    Epoch 10/150
    47500/47500 [==============================] - 4s 80us/sample - loss: 1.7101e-04 - accuracy: 1.0000 - val_loss: 1.5770e-04 - val_accuracy: 1.0000
    Epoch 11/150
    47500/47500 [==============================] - 4s 84us/sample - loss: 1.2567e-04 - accuracy: 1.0000 - val_loss: 1.1817e-04 - val_accuracy: 1.0000
    Epoch 12/150
    47500/47500 [==============================] - 4s 85us/sample - loss: 9.3803e-05 - accuracy: 1.0000 - val_loss: 8.7954e-05 - val_accuracy: 1.0000
    Epoch 13/150
    47500/47500 [==============================] - 4s 89us/sample - loss: 7.1097e-05 - accuracy: 1.0000 - val_loss: 6.8511e-05 - val_accuracy: 1.0000
    Training for position: 8
    Train on 47500 samples, validate on 2500 samples
    Epoch 1/150
    47500/47500 [==============================] - 4s 91us/sample - loss: 1.0001 - accuracy: 0.7367 - val_loss: 0.7693 - val_accuracy: 0.7756
    Epoch 2/150
    47500/47500 [==============================] - 4s 80us/sample - loss: 0.5260 - accuracy: 0.8290 - val_loss: 0.2538 - val_accuracy: 0.9044
    Epoch 3/150
    47500/47500 [==============================] - 4s 92us/sample - loss: 0.0847 - accuracy: 0.9803 - val_loss: 0.0101 - val_accuracy: 1.0000
    Epoch 4/150
    47500/47500 [==============================] - 5s 95us/sample - loss: 0.0050 - accuracy: 1.0000 - val_loss: 0.0028 - val_accuracy: 1.0000
    Epoch 5/150
    47500/47500 [==============================] - 6s 117us/sample - loss: 0.0018 - accuracy: 1.0000 - val_loss: 0.0014 - val_accuracy: 1.0000
    Epoch 6/150
    47500/47500 [==============================] - 5s 103us/sample - loss: 9.5133e-04 - accuracy: 1.0000 - val_loss: 7.8549e-04 - val_accuracy: 1.0000
    Epoch 7/150
    47500/47500 [==============================] - 5s 96us/sample - loss: 5.7835e-04 - accuracy: 1.0000 - val_loss: 5.1678e-04 - val_accuracy: 1.0000
    Epoch 8/150
    47500/47500 [==============================] - 4s 89us/sample - loss: 3.8172e-04 - accuracy: 1.0000 - val_loss: 3.5420e-04 - val_accuracy: 1.0000
    Epoch 9/150
    47500/47500 [==============================] - 4s 82us/sample - loss: 2.6327e-04 - accuracy: 1.0000 - val_loss: 2.4633e-04 - val_accuracy: 1.0000
    Epoch 10/150
    47500/47500 [==============================] - 4s 87us/sample - loss: 1.8893e-04 - accuracy: 1.0000 - val_loss: 1.8116e-04 - val_accuracy: 1.0000
    Epoch 11/150
    47500/47500 [==============================] - 4s 85us/sample - loss: 1.3920e-04 - accuracy: 1.0000 - val_loss: 1.3508e-04 - val_accuracy: 1.0000
    Epoch 12/150
    47500/47500 [==============================] - 4s 86us/sample - loss: 1.0405e-04 - accuracy: 1.0000 - val_loss: 1.0289e-04 - val_accuracy: 1.0000
    Epoch 13/150
    47500/47500 [==============================] - 4s 82us/sample - loss: 7.9075e-05 - accuracy: 1.0000 - val_loss: 7.8687e-05 - val_accuracy: 1.0000
    Epoch 14/150
    47500/47500 [==============================] - 4s 80us/sample - loss: 6.0809e-05 - accuracy: 1.0000 - val_loss: 6.1717e-05 - val_accuracy: 1.0000
    Training for position: 9
    Train on 47500 samples, validate on 2500 samples
    Epoch 1/150
    47500/47500 [==============================] - 4s 91us/sample - loss: 0.6989 - accuracy: 0.8207 - val_loss: 0.5415 - val_accuracy: 0.8448
    Epoch 2/150
    47500/47500 [==============================] - 4s 82us/sample - loss: 0.5307 - accuracy: 0.8392 - val_loss: 0.3950 - val_accuracy: 0.8712
    Epoch 3/150
    47500/47500 [==============================] - 4s 87us/sample - loss: 0.2265 - accuracy: 0.9318 - val_loss: 0.0674 - val_accuracy: 0.9868
    Epoch 4/150
    47500/47500 [==============================] - 4s 81us/sample - loss: 0.0167 - accuracy: 0.9990 - val_loss: 0.0054 - val_accuracy: 1.0000
    Epoch 5/150
    47500/47500 [==============================] - 4s 86us/sample - loss: 0.0029 - accuracy: 1.0000 - val_loss: 0.0022 - val_accuracy: 1.0000
    Epoch 6/150
    47500/47500 [==============================] - 5s 96us/sample - loss: 0.0013 - accuracy: 1.0000 - val_loss: 0.0012 - val_accuracy: 1.0000
    Epoch 7/150
    47500/47500 [==============================] - 4s 88us/sample - loss: 7.7924e-04 - accuracy: 1.0000 - val_loss: 7.6629e-04 - val_accuracy: 1.0000
    Epoch 8/150
    47500/47500 [==============================] - 4s 89us/sample - loss: 4.9838e-04 - accuracy: 1.0000 - val_loss: 4.9883e-04 - val_accuracy: 1.0000
    Epoch 9/150
    47500/47500 [==============================] - 4s 83us/sample - loss: 3.3783e-04 - accuracy: 1.0000 - val_loss: 3.4043e-04 - val_accuracy: 1.0000
    Epoch 10/150
    47500/47500 [==============================] - 4s 89us/sample - loss: 2.3909e-04 - accuracy: 1.0000 - val_loss: 2.5698e-04 - val_accuracy: 1.0000
    Epoch 11/150
    47500/47500 [==============================] - 4s 87us/sample - loss: 1.7413e-04 - accuracy: 1.0000 - val_loss: 1.8301e-04 - val_accuracy: 1.0000
    Epoch 12/150
    47500/47500 [==============================] - 4s 81us/sample - loss: 1.2892e-04 - accuracy: 1.0000 - val_loss: 1.3994e-04 - val_accuracy: 1.0000
    Epoch 13/150
    47500/47500 [==============================] - 4s 82us/sample - loss: 9.7669e-05 - accuracy: 1.0000 - val_loss: 1.0528e-04 - val_accuracy: 1.0000
    Epoch 14/150
    47500/47500 [==============================] - 4s 86us/sample - loss: 7.4483e-05 - accuracy: 1.0000 - val_loss: 8.1553e-05 - val_accuracy: 1.0000
    Epoch 15/150
    47500/47500 [==============================] - 4s 83us/sample - loss: 5.7405e-05 - accuracy: 1.0000 - val_loss: 6.3936e-05 - val_accuracy: 1.0000
    Epoch 16/150
    47500/47500 [==============================] - 4s 81us/sample - loss: 4.4750e-05 - accuracy: 1.0000 - val_loss: 5.0525e-05 - val_accuracy: 1.0000
    Epoch 17/150
    47500/47500 [==============================] - 4s 79us/sample - loss: 3.4935e-05 - accuracy: 1.0000 - val_loss: 3.9022e-05 - val_accuracy: 1.0000
    Training for position: 10
    Train on 47500 samples, validate on 2500 samples
    Epoch 1/150
    47500/47500 [==============================] - 4s 91us/sample - loss: 0.3802 - accuracy: 0.9094 - val_loss: 0.2696 - val_accuracy: 0.9216
    Epoch 2/150
    47500/47500 [==============================] - 4s 79us/sample - loss: 0.2740 - accuracy: 0.9170 - val_loss: 0.2291 - val_accuracy: 0.9248
    Epoch 3/150
    47500/47500 [==============================] - 4s 89us/sample - loss: 0.2142 - accuracy: 0.9276 - val_loss: 0.1679 - val_accuracy: 0.9388
    Epoch 4/150
    47500/47500 [==============================] - 5s 98us/sample - loss: 0.1324 - accuracy: 0.9525 - val_loss: 0.0742 - val_accuracy: 0.9792
    Epoch 5/150
    47500/47500 [==============================] - 4s 82us/sample - loss: 0.0423 - accuracy: 0.9898 - val_loss: 0.0149 - val_accuracy: 0.9988
    Epoch 6/150
    47500/47500 [==============================] - 4s 87us/sample - loss: 0.0068 - accuracy: 0.9999 - val_loss: 0.0038 - val_accuracy: 1.0000
    Epoch 7/150
    47500/47500 [==============================] - 4s 88us/sample - loss: 0.0021 - accuracy: 1.0000 - val_loss: 0.0020 - val_accuracy: 0.9996
    Epoch 8/150
    47500/47500 [==============================] - 4s 88us/sample - loss: 0.0010 - accuracy: 1.0000 - val_loss: 0.0012 - val_accuracy: 1.0000
    Epoch 9/150
    47500/47500 [==============================] - 4s 86us/sample - loss: 6.1536e-04 - accuracy: 1.0000 - val_loss: 7.1127e-04 - val_accuracy: 1.0000
    Epoch 10/150
    47500/47500 [==============================] - 4s 87us/sample - loss: 4.0613e-04 - accuracy: 1.0000 - val_loss: 6.3158e-04 - val_accuracy: 1.0000
    Epoch 11/150
    47500/47500 [==============================] - 4s 87us/sample - loss: 2.8448e-04 - accuracy: 1.0000 - val_loss: 4.0149e-04 - val_accuracy: 1.0000
    Epoch 12/150
    47500/47500 [==============================] - 4s 85us/sample - loss: 2.0404e-04 - accuracy: 1.0000 - val_loss: 3.1312e-04 - val_accuracy: 1.0000
    Epoch 13/150
    47500/47500 [==============================] - 4s 81us/sample - loss: 1.5106e-04 - accuracy: 1.0000 - val_loss: 2.2407e-04 - val_accuracy: 1.0000
    Epoch 14/150
    47500/47500 [==============================] - 4s 81us/sample - loss: 1.1373e-04 - accuracy: 1.0000 - val_loss: 1.7730e-04 - val_accuracy: 1.0000
    Epoch 15/150
    47500/47500 [==============================] - 4s 82us/sample - loss: 8.6037e-05 - accuracy: 1.0000 - val_loss: 1.3810e-04 - val_accuracy: 1.0000
    Epoch 16/150
    47500/47500 [==============================] - 4s 84us/sample - loss: 6.5504e-05 - accuracy: 1.0000 - val_loss: 1.5013e-04 - val_accuracy: 1.0000



{% highlight python %}
# plot every loss and accuracy
pos = 0
for h in histories:
    plt.plot(h.history['val_accuracy'], label=str(pos))
    pos += 1
plt.title("Accuracy in range [.98,1]")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="lower right")
plt.gca().axes.set_ylim([.98,1.001]) # set limits to zoom in into meaningful data
plt.show()

pos = 0
for h in histories:
    plt.plot(h.history['val_loss'], label=str(pos))
    pos += 1
plt.title('Loss in range [.0,.01]')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="upper right")
plt.gca().axes.set_ylim([.0,.01]) # set limits to zoom in into meaningful data
plt.show()
{% endhighlight %}


![png](/assets/qr-nn/output_55_0.png)



![png](/assets/qr-nn/output_55_1.png)


As expected, every larger position performs a bit worse, since the dataset has less and less well-distributed samples per position. Let's try to decide a few QR codes and see how accuracy and confidence behaves:


{% highlight python %}
# print each letter from a QR code with confidence level
def predict_qr(s, qr_code, models):
    print("--------------")
    for i in range(11):
        expected_char = s[i] if i < len(s) else EOI
        (c, confidence, all) = get_letter(qr_code, models[i])
        print(c == expected_char, "Predicted:", c, "Expected:", expected_char, "Confidence:", confidence)

[predict_qr(s, make_qr(s), models_per_position) for s in ["hello world"]]
[predict_qr(s, make_qr(s), models_per_position) for s in ["machine ai"]]
[predict_qr(s, make_qr(s), models_per_position) for s in ["joe"]]
[predict_qr(s, make_qr(s), models_per_position) for s in ["iddqd"]]
[predict_qr(s, make_qr(s), models_per_position) for s in ["longer str"]]
[predict_qr(s, make_qr(s), models_per_position) for s in ["may fail at"]]
[predict_qr(s, make_qr(s), models_per_position) for s in ["data master"]]
[predict_qr(s, make_qr(s), models_per_position) for s in ["iddqd quake"]]
[predict_qr(s, make_qr(s), models_per_position) for s in ["datamatters"]]
{% endhighlight %}

    --------------
    True Predicted: h Expected: h Confidence: 0.99994385
    True Predicted: e Expected: e Confidence: 0.9999453
    True Predicted: l Expected: l Confidence: 0.99992955
    True Predicted: l Expected: l Confidence: 0.99993896
    True Predicted: o Expected: o Confidence: 0.99990535
    True Predicted:   Expected:   Confidence: 0.9999378
    True Predicted: w Expected: w Confidence: 0.99981326
    True Predicted: o Expected: o Confidence: 0.999902
    True Predicted: r Expected: r Confidence: 0.99974626
    True Predicted: l Expected: l Confidence: 0.9998753
    True Predicted: d Expected: d Confidence: 0.9994261
    --------------
    True Predicted: m Expected: m Confidence: 0.9999678
    True Predicted: a Expected: a Confidence: 0.99994326
    True Predicted: c Expected: c Confidence: 0.9999323
    True Predicted: h Expected: h Confidence: 0.99995327
    True Predicted: i Expected: i Confidence: 0.99989593
    True Predicted: n Expected: n Confidence: 0.999923
    True Predicted: e Expected: e Confidence: 0.9999316
    True Predicted:   Expected:   Confidence: 0.999879
    True Predicted: a Expected: a Confidence: 0.9998983
    True Predicted: i Expected: i Confidence: 0.9997764
    True Predicted: # Expected: # Confidence: 1.0
    --------------
    True Predicted: j Expected: j Confidence: 0.9999777
    True Predicted: o Expected: o Confidence: 0.99997175
    True Predicted: e Expected: e Confidence: 0.9999361
    True Predicted: # Expected: # Confidence: 0.9999951
    True Predicted: # Expected: # Confidence: 1.0
    True Predicted: # Expected: # Confidence: 1.0
    True Predicted: # Expected: # Confidence: 1.0
    True Predicted: # Expected: # Confidence: 1.0
    True Predicted: # Expected: # Confidence: 1.0
    True Predicted: # Expected: # Confidence: 1.0
    True Predicted: # Expected: # Confidence: 1.0
    --------------
    True Predicted: i Expected: i Confidence: 0.99997306
    True Predicted: d Expected: d Confidence: 0.9999449
    True Predicted: d Expected: d Confidence: 0.9999641
    True Predicted: q Expected: q Confidence: 0.9998965
    True Predicted: d Expected: d Confidence: 0.99992144
    True Predicted: # Expected: # Confidence: 0.9999989
    True Predicted: # Expected: # Confidence: 1.0
    True Predicted: # Expected: # Confidence: 1.0
    True Predicted: # Expected: # Confidence: 1.0
    True Predicted: # Expected: # Confidence: 1.0
    True Predicted: # Expected: # Confidence: 1.0
    --------------
    True Predicted: l Expected: l Confidence: 0.999966
    True Predicted: o Expected: o Confidence: 0.9999641
    True Predicted: n Expected: n Confidence: 0.9999455
    True Predicted: g Expected: g Confidence: 0.9999435
    True Predicted: e Expected: e Confidence: 0.99991107
    True Predicted: r Expected: r Confidence: 0.9999573
    True Predicted:   Expected:   Confidence: 0.9998803
    True Predicted: s Expected: s Confidence: 0.9997596
    True Predicted: t Expected: t Confidence: 0.99989533
    True Predicted: r Expected: r Confidence: 0.99983525
    True Predicted: # Expected: # Confidence: 1.0
    --------------
    True Predicted: m Expected: m Confidence: 0.9999577
    True Predicted: a Expected: a Confidence: 0.9999337
    True Predicted: y Expected: y Confidence: 0.9999318
    True Predicted:   Expected:   Confidence: 0.99995625
    True Predicted: f Expected: f Confidence: 0.9997105
    True Predicted: a Expected: a Confidence: 0.9998411
    True Predicted: i Expected: i Confidence: 0.99984396
    True Predicted: l Expected: l Confidence: 0.9996667
    True Predicted:   Expected:   Confidence: 0.99975556
    True Predicted: a Expected: a Confidence: 0.9977695
    True Predicted: t Expected: t Confidence: 0.99968326
    --------------
    True Predicted: d Expected: d Confidence: 0.9999651
    True Predicted: a Expected: a Confidence: 0.9999691
    True Predicted: t Expected: t Confidence: 0.9999416
    True Predicted: a Expected: a Confidence: 0.99997354
    True Predicted:   Expected:   Confidence: 0.99991894
    True Predicted: m Expected: m Confidence: 0.99988115
    True Predicted: a Expected: a Confidence: 0.9998586
    True Predicted: s Expected: s Confidence: 0.9998909
    True Predicted: t Expected: t Confidence: 0.99995196
    True Predicted: e Expected: e Confidence: 0.99984646
    True Predicted: r Expected: r Confidence: 0.9992467
    --------------
    True Predicted: i Expected: i Confidence: 0.9999603
    True Predicted: d Expected: d Confidence: 0.9998685
    True Predicted: d Expected: d Confidence: 0.99988675
    True Predicted: q Expected: q Confidence: 0.9997856
    True Predicted: d Expected: d Confidence: 0.99993145
    True Predicted:   Expected:   Confidence: 0.9999727
    True Predicted: q Expected: q Confidence: 0.9999138
    True Predicted: u Expected: u Confidence: 0.99969554
    True Predicted: a Expected: a Confidence: 0.9997893
    True Predicted: k Expected: k Confidence: 0.99917233
    True Predicted: e Expected: e Confidence: 0.9989573
    --------------
    True Predicted: d Expected: d Confidence: 0.9999448
    True Predicted: a Expected: a Confidence: 0.99996173
    True Predicted: t Expected: t Confidence: 0.9999517
    True Predicted: a Expected: a Confidence: 0.9997781
    True Predicted: m Expected: m Confidence: 0.9999137
    True Predicted: a Expected: a Confidence: 0.9999112
    True Predicted: t Expected: t Confidence: 0.99995315
    True Predicted: t Expected: t Confidence: 0.99987435
    True Predicted: e Expected: e Confidence: 0.99965656
    True Predicted: r Expected: r Confidence: 0.9997918
    True Predicted: s Expected: s Confidence: 0.9981238





    [None]



To run one more test, we can take 10k of the most popular English words and try NNs on them. Get this repo: https://github.com/first20hours/google-10000-english and update the path below. The code will load each word and trim longer ones to have the first 11 characters.


{% highlight python %}
GOOGLE_10000_WORDS = 'google-10000-english-no-swears.txt' # update path to the file
with open(GOOGLE_10000_WORDS, 'r') as f:
    words10000 = f.readlines()

words11_data = [] # QR codes
words11 = [] # text labels
for w in words10000:
    label = w.strip()[:11]
    if len(label) == 0:
        continue
    words11.append(label)
    img = make_qr(label)
    assert img.size == (21, 21)
    words11_data.append(np.asarray(img, dtype='float'))
words11_data = np.asarray(words11_data)
{% endhighlight %}


{% highlight python %}
# run the NN against all QR codes and remember texts with errors
# since we have separate networks per position, results are also per position
def run_words_with_individual_models():
    wrong_words = []
    for pos in range(11): # apply one model per position
        smallest_confidence = 1
        model = models_per_position[pos]
        output = model.predict(words11_data)
        correct = 0
        for i in range(len(output)):
            expected_char = words11[i][pos] if pos < len(words11[i]) else EOI # if text is shorter than 11, tail characters are '#'
            largest_index = np.argmax(output[i], axis=0)
            smallest_confidence = min(smallest_confidence, output[i][largest_index])
            predicred_char = ALL_CHAR_CLASSES[largest_index]
            if predicred_char == expected_char:
                correct += 1
            else:
                wrong_words.append(words11[i])
        print("Pos", pos, "Correct", correct/len(words11_data), "Mis. count:", len(words11_data) - correct, "Smallest confidence:", smallest_confidence)
    return wrong_words
wrong_words = run_words_with_individual_models()
{% endhighlight %}

    Pos 0 Correct 1.0 Mis. count: 0 Smallest confidence: 0.9994931
    Pos 1 Correct 1.0 Mis. count: 0 Smallest confidence: 0.99438184
    Pos 2 Correct 1.0 Mis. count: 0 Smallest confidence: 0.9983413
    Pos 3 Correct 1.0 Mis. count: 0 Smallest confidence: 0.9883525
    Pos 4 Correct 1.0 Mis. count: 0 Smallest confidence: 0.98402286
    Pos 5 Correct 1.0 Mis. count: 0 Smallest confidence: 0.98981196
    Pos 6 Correct 1.0 Mis. count: 0 Smallest confidence: 0.9919063
    Pos 7 Correct 1.0 Mis. count: 0 Smallest confidence: 0.9950441
    Pos 8 Correct 1.0 Mis. count: 0 Smallest confidence: 0.9915815
    Pos 9 Correct 1.0 Mis. count: 0 Smallest confidence: 0.97259146
    Pos 10 Correct 1.0 Mis. count: 0 Smallest confidence: 0.97291905


At this time we should have 100% accuracy. The last column is the smallest confidence per position. A high number is great.

## 5. Build single NN to decode QR codes

So far we have trained eleven individual NNs to decode symbols of a QR code one by one. Seems like a lot of networks to manage. It makes sense to join NNs to one.

The intent is to build one NN with 11 outputs:


{% highlight python %}
# first recreate labels for all positions
training_label_sizes = list(map(lambda x: [len(x) - 1], training_labels))
training_labels_char0 = make_labels_for_position(training_labels, 0)
training_labels_char1 = make_labels_for_position(training_labels, 1)
training_labels_char2 = make_labels_for_position(training_labels, 2)
training_labels_char3 = make_labels_for_position(training_labels, 3)
training_labels_char4 = make_labels_for_position(training_labels, 4)
training_labels_char5 = make_labels_for_position(training_labels, 5)
training_labels_char6 = make_labels_for_position(training_labels, 6)
training_labels_char7 = make_labels_for_position(training_labels, 7)
training_labels_char8 = make_labels_for_position(training_labels, 8)
training_labels_char9 = make_labels_for_position(training_labels, 9)
training_labels_char10 = make_labels_for_position(training_labels, 10)
{% endhighlight %}


{% highlight python %}
#                                                     |-> char 0 output
# INPUT -> FLATTEN -> HIDDEN LAYER1 -> HIDDEN LAYER2 -|-> ...
#                                                     |-> char 10 output
#
#
def define_multi_output_model():
    input_layer = keras.layers.Input(shape=(21,21,), dtype='float', name='input_qr')
    flatten = keras.layers.Flatten(input_shape=(21, 21), name='flatten')(input_layer)    
    hidden_chars1 = keras.layers.Dense(21*21, activation='relu', name='hidden_chars1')(flatten)
    hidden_chars2 = keras.layers.Dense(21*21, activation='relu', name='hidden_chars2')(hidden_chars1)

    outputs = []
    for i in range(11):
        char_output = keras.layers.Dense(len(ALL_CHAR_CLASSES), name='char' + str(i))(hidden_chars2)
        outputs.append(char_output)

    multi_output_model = keras.Model(inputs=[input_layer], outputs=outputs)

    multi_output_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return multi_output_model

mo_model = define_multi_output_model()
mo_model.summary()
mo_hist = mo_model.fit(
        training_data, [
            np.asarray(training_labels_char0),
            np.asarray(training_labels_char1),
            np.asarray(training_labels_char2),
            np.asarray(training_labels_char3),
            np.asarray(training_labels_char4),
            np.asarray(training_labels_char5),
            np.asarray(training_labels_char6),
            np.asarray(training_labels_char7),
            np.asarray(training_labels_char8),
            np.asarray(training_labels_char9),
            np.asarray(training_labels_char10)
        ],
        epochs=150, batch_size=128,
        validation_split=0.05,
        callbacks=[EarlyStopping(monitor='val_loss' , min_delta=0.0001, patience=3)]
    )
{% endhighlight %}

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_qr (InputLayer)           [(None, 21, 21)]     0                                            
    __________________________________________________________________________________________________
    flatten (Flatten)               (None, 441)          0           input_qr[0][0]                   
    __________________________________________________________________________________________________
    hidden_chars1 (Dense)           (None, 441)          194922      flatten[0][0]                    
    __________________________________________________________________________________________________
    hidden_chars2 (Dense)           (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    char0 (Dense)                   (None, 28)           12376       hidden_chars2[0][0]              
    __________________________________________________________________________________________________
    char1 (Dense)                   (None, 28)           12376       hidden_chars2[0][0]              
    __________________________________________________________________________________________________
    char2 (Dense)                   (None, 28)           12376       hidden_chars2[0][0]              
    __________________________________________________________________________________________________
    char3 (Dense)                   (None, 28)           12376       hidden_chars2[0][0]              
    __________________________________________________________________________________________________
    char4 (Dense)                   (None, 28)           12376       hidden_chars2[0][0]              
    __________________________________________________________________________________________________
    char5 (Dense)                   (None, 28)           12376       hidden_chars2[0][0]              
    __________________________________________________________________________________________________
    char6 (Dense)                   (None, 28)           12376       hidden_chars2[0][0]              
    __________________________________________________________________________________________________
    char7 (Dense)                   (None, 28)           12376       hidden_chars2[0][0]              
    __________________________________________________________________________________________________
    char8 (Dense)                   (None, 28)           12376       hidden_chars2[0][0]              
    __________________________________________________________________________________________________
    char9 (Dense)                   (None, 28)           12376       hidden_chars2[0][0]              
    __________________________________________________________________________________________________
    char10 (Dense)                  (None, 28)           12376       hidden_chars2[0][0]              
    ==================================================================================================
    Total params: 525,980
    Trainable params: 525,980
    Non-trainable params: 0
    __________________________________________________________________________________________________
    Train on 47500 samples, validate on 2500 samples
    Epoch 1/150
    47500/47500 [==============================] - 12s 259us/sample - loss: 20.5153 - char0_loss: 3.0221 - char1_loss: 2.8597 - char2_loss: 2.7826 - char3_loss: 2.4774 - char4_loss: 2.2455 - char5_loss: 1.9359 - char6_loss: 1.6435 - char7_loss: 1.3467 - char8_loss: 1.0525 - char9_loss: 0.7462 - char10_loss: 0.4066 - char0_accuracy: 0.1306 - char1_accuracy: 0.1908 - char2_accuracy: 0.2152 - char3_accuracy: 0.3084 - char4_accuracy: 0.3820 - char5_accuracy: 0.4723 - char6_accuracy: 0.5563 - char7_accuracy: 0.6406 - char8_accuracy: 0.7283 - char9_accuracy: 0.8169 - char10_accuracy: 0.9069 - val_loss: 18.0820 - val_char0_loss: 2.5901 - val_char1_loss: 2.4234 - val_char2_loss: 2.5413 - val_char3_loss: 2.2038 - val_char4_loss: 2.0920 - val_char5_loss: 1.7979 - val_char6_loss: 1.4903 - val_char7_loss: 1.1607 - val_char8_loss: 0.8801 - val_char9_loss: 0.5803 - val_char10_loss: 0.2955 - val_char0_accuracy: 0.2124 - val_char1_accuracy: 0.3104 - val_char2_accuracy: 0.2588 - val_char3_accuracy: 0.3600 - val_char4_accuracy: 0.4028 - val_char5_accuracy: 0.4892 - val_char6_accuracy: 0.5748 - val_char7_accuracy: 0.6728 - val_char8_accuracy: 0.7544 - val_char9_accuracy: 0.8380 - val_char10_accuracy: 0.9196
    Epoch 2/150
    47500/47500 [==============================] - 9s 187us/sample - loss: 16.7026 - char0_loss: 2.2007 - char1_loss: 1.8370 - char2_loss: 2.3287 - char3_loss: 1.9822 - char4_loss: 1.9956 - char5_loss: 1.7533 - char6_loss: 1.5108 - char7_loss: 1.2115 - char8_loss: 0.9218 - char9_loss: 0.6284 - char10_loss: 0.3185 - char0_accuracy: 0.2613 - char1_accuracy: 0.4565 - char2_accuracy: 0.2785 - char3_accuracy: 0.3751 - char4_accuracy: 0.4188 - char5_accuracy: 0.4945 - char6_accuracy: 0.5686 - char7_accuracy: 0.6555 - char8_accuracy: 0.7387 - char9_accuracy: 0.8243 - char10_accuracy: 0.9122 - val_loss: 14.6173 - val_char0_loss: 1.7738 - val_char1_loss: 1.1651 - val_char2_loss: 2.0345 - val_char3_loss: 1.7399 - val_char4_loss: 1.8680 - val_char5_loss: 1.6859 - val_char6_loss: 1.4545 - val_char7_loss: 1.1429 - val_char8_loss: 0.8634 - val_char9_loss: 0.5778 - val_char10_loss: 0.2867 - val_char0_accuracy: 0.3428 - val_char1_accuracy: 0.6348 - val_char2_accuracy: 0.3276 - val_char3_accuracy: 0.4120 - val_char4_accuracy: 0.4360 - val_char5_accuracy: 0.5052 - val_char6_accuracy: 0.5844 - val_char7_accuracy: 0.6764 - val_char8_accuracy: 0.7552 - val_char9_accuracy: 0.8412 - val_char10_accuracy: 0.9192
    Epoch 3/150
    47500/47500 [==============================] - 8s 170us/sample - loss: 12.4719 - char0_loss: 1.2098 - char1_loss: 0.6857 - char2_loss: 1.4700 - char3_loss: 1.4708 - char4_loss: 1.6247 - char5_loss: 1.5792 - char6_loss: 1.4418 - char7_loss: 1.1583 - char8_loss: 0.8874 - char9_loss: 0.6250 - char10_loss: 0.3132 - char0_accuracy: 0.5968 - char1_accuracy: 0.8009 - char2_accuracy: 0.5151 - char3_accuracy: 0.4797 - char4_accuracy: 0.4753 - char5_accuracy: 0.5244 - char6_accuracy: 0.5840 - char7_accuracy: 0.6676 - char8_accuracy: 0.7473 - char9_accuracy: 0.8273 - char10_accuracy: 0.9136 - val_loss: 9.9856 - val_char0_loss: 0.5615 - val_char1_loss: 0.3421 - val_char2_loss: 0.8660 - val_char3_loss: 1.1863 - val_char4_loss: 1.3848 - val_char5_loss: 1.4758 - val_char6_loss: 1.3566 - val_char7_loss: 1.0793 - val_char8_loss: 0.8311 - val_char9_loss: 0.5861 - val_char10_loss: 0.2950 - val_char0_accuracy: 0.8820 - val_char1_accuracy: 0.9336 - val_char2_accuracy: 0.7228 - val_char3_accuracy: 0.5664 - val_char4_accuracy: 0.5144 - val_char5_accuracy: 0.5468 - val_char6_accuracy: 0.5980 - val_char7_accuracy: 0.6820 - val_char8_accuracy: 0.7588 - val_char9_accuracy: 0.8448 - val_char10_accuracy: 0.9216
    Epoch 4/150
    47500/47500 [==============================] - 9s 179us/sample - loss: 8.3028 - char0_loss: 0.2990 - char1_loss: 0.1868 - char2_loss: 0.4961 - char3_loss: 0.7982 - char4_loss: 1.0656 - char5_loss: 1.3016 - char6_loss: 1.3242 - char7_loss: 1.0429 - char8_loss: 0.8364 - char9_loss: 0.6304 - char10_loss: 0.3169 - char0_accuracy: 0.9580 - char1_accuracy: 0.9810 - char2_accuracy: 0.8748 - char3_accuracy: 0.7322 - char4_accuracy: 0.6291 - char5_accuracy: 0.5828 - char6_accuracy: 0.6065 - char7_accuracy: 0.6885 - char8_accuracy: 0.7594 - char9_accuracy: 0.8289 - char10_accuracy: 0.9153 - val_loss: 7.1850 - val_char0_loss: 0.2392 - val_char1_loss: 0.1831 - val_char2_loss: 0.3091 - val_char3_loss: 0.5474 - val_char4_loss: 0.8388 - val_char5_loss: 1.1887 - val_char6_loss: 1.2015 - val_char7_loss: 0.9535 - val_char8_loss: 0.7862 - val_char9_loss: 0.6181 - val_char10_loss: 0.3050 - val_char0_accuracy: 0.9716 - val_char1_accuracy: 0.9804 - val_char2_accuracy: 0.9556 - val_char3_accuracy: 0.8600 - val_char4_accuracy: 0.7216 - val_char5_accuracy: 0.6012 - val_char6_accuracy: 0.6268 - val_char7_accuracy: 0.7064 - val_char8_accuracy: 0.7692 - val_char9_accuracy: 0.8408 - val_char10_accuracy: 0.9192
    Epoch 5/150
    47500/47500 [==============================] - 7s 153us/sample - loss: 5.5195 - char0_loss: 0.1162 - char1_loss: 0.0774 - char2_loss: 0.1401 - char3_loss: 0.2541 - char4_loss: 0.4727 - char5_loss: 0.8931 - char6_loss: 1.0755 - char7_loss: 0.8669 - char8_loss: 0.7226 - char9_loss: 0.5985 - char10_loss: 0.3014 - char0_accuracy: 0.9916 - char1_accuracy: 0.9963 - char2_accuracy: 0.9918 - char3_accuracy: 0.9658 - char4_accuracy: 0.8771 - char5_accuracy: 0.6925 - char6_accuracy: 0.6536 - char7_accuracy: 0.7222 - char8_accuracy: 0.7841 - char9_accuracy: 0.8339 - char10_accuracy: 0.9176 - val_loss: 4.7243 - val_char0_loss: 0.0966 - val_char1_loss: 0.0602 - val_char2_loss: 0.1004 - val_char3_loss: 0.1447 - val_char4_loss: 0.2972 - val_char5_loss: 0.7358 - val_char6_loss: 0.9529 - val_char7_loss: 0.8210 - val_char8_loss: 0.6579 - val_char9_loss: 0.5716 - val_char10_loss: 0.2795 - val_char0_accuracy: 0.9880 - val_char1_accuracy: 0.9968 - val_char2_accuracy: 0.9940 - val_char3_accuracy: 0.9884 - val_char4_accuracy: 0.9416 - val_char5_accuracy: 0.7404 - val_char6_accuracy: 0.6808 - val_char7_accuracy: 0.7368 - val_char8_accuracy: 0.8036 - val_char9_accuracy: 0.8468 - val_char10_accuracy: 0.9240
    Epoch 6/150
    47500/47500 [==============================] - 7s 151us/sample - loss: 3.8533 - char0_loss: 0.0655 - char1_loss: 0.0452 - char2_loss: 0.0677 - char3_loss: 0.1031 - char4_loss: 0.1671 - char5_loss: 0.4720 - char6_loss: 0.7693 - char7_loss: 0.6834 - char8_loss: 0.6097 - char9_loss: 0.5789 - char10_loss: 0.2893 - char0_accuracy: 0.9964 - char1_accuracy: 0.9985 - char2_accuracy: 0.9981 - char3_accuracy: 0.9941 - char4_accuracy: 0.9854 - char5_accuracy: 0.8636 - char6_accuracy: 0.7413 - char7_accuracy: 0.7729 - char8_accuracy: 0.8139 - char9_accuracy: 0.8372 - char10_accuracy: 0.9199 - val_loss: 3.3035 - val_char0_loss: 0.0612 - val_char1_loss: 0.0448 - val_char2_loss: 0.0580 - val_char3_loss: 0.0804 - val_char4_loss: 0.1230 - val_char5_loss: 0.3277 - val_char6_loss: 0.6041 - val_char7_loss: 0.6086 - val_char8_loss: 0.5376 - val_char9_loss: 0.5594 - val_char10_loss: 0.2923 - val_char0_accuracy: 0.9940 - val_char1_accuracy: 0.9964 - val_char2_accuracy: 0.9972 - val_char3_accuracy: 0.9948 - val_char4_accuracy: 0.9908 - val_char5_accuracy: 0.9232 - val_char6_accuracy: 0.8008 - val_char7_accuracy: 0.7872 - val_char8_accuracy: 0.8368 - val_char9_accuracy: 0.8460 - val_char10_accuracy: 0.9224
    Epoch 7/150
    47500/47500 [==============================] - 7s 152us/sample - loss: 2.6779 - char0_loss: 0.0446 - char1_loss: 0.0309 - char2_loss: 0.0426 - char3_loss: 0.0569 - char4_loss: 0.0818 - char5_loss: 0.2080 - char6_loss: 0.4290 - char7_loss: 0.4629 - char8_loss: 0.4889 - char9_loss: 0.5507 - char10_loss: 0.2818 - char0_accuracy: 0.9975 - char1_accuracy: 0.9991 - char2_accuracy: 0.9989 - char3_accuracy: 0.9978 - char4_accuracy: 0.9960 - char5_accuracy: 0.9688 - char6_accuracy: 0.8739 - char7_accuracy: 0.8459 - char8_accuracy: 0.8448 - char9_accuracy: 0.8420 - char10_accuracy: 0.9216 - val_loss: 2.3917 - val_char0_loss: 0.0448 - val_char1_loss: 0.0326 - val_char2_loss: 0.0389 - val_char3_loss: 0.0574 - val_char4_loss: 0.0641 - val_char5_loss: 0.1601 - val_char6_loss: 0.2937 - val_char7_loss: 0.3917 - val_char8_loss: 0.4400 - val_char9_loss: 0.5749 - val_char10_loss: 0.2888 - val_char0_accuracy: 0.9960 - val_char1_accuracy: 0.9988 - val_char2_accuracy: 0.9972 - val_char3_accuracy: 0.9968 - val_char4_accuracy: 0.9976 - val_char5_accuracy: 0.9792 - val_char6_accuracy: 0.9292 - val_char7_accuracy: 0.8612 - val_char8_accuracy: 0.8528 - val_char9_accuracy: 0.8492 - val_char10_accuracy: 0.9236
    Epoch 8/150
    47500/47500 [==============================] - 7s 153us/sample - loss: 1.8565 - char0_loss: 0.0302 - char1_loss: 0.0212 - char2_loss: 0.0297 - char3_loss: 0.0360 - char4_loss: 0.0480 - char5_loss: 0.1004 - char6_loss: 0.1904 - char7_loss: 0.2531 - char8_loss: 0.3594 - char9_loss: 0.5156 - char10_loss: 0.2727 - char0_accuracy: 0.9987 - char1_accuracy: 0.9993 - char2_accuracy: 0.9990 - char3_accuracy: 0.9991 - char4_accuracy: 0.9985 - char5_accuracy: 0.9930 - char6_accuracy: 0.9661 - char7_accuracy: 0.9287 - char8_accuracy: 0.8826 - char9_accuracy: 0.8496 - char10_accuracy: 0.9237 - val_loss: 1.7242 - val_char0_loss: 0.0340 - val_char1_loss: 0.0266 - val_char2_loss: 0.0267 - val_char3_loss: 0.0343 - val_char4_loss: 0.0404 - val_char5_loss: 0.0821 - val_char6_loss: 0.1350 - val_char7_loss: 0.1998 - val_char8_loss: 0.3168 - val_char9_loss: 0.5328 - val_char10_loss: 0.2956 - val_char0_accuracy: 0.9968 - val_char1_accuracy: 0.9976 - val_char2_accuracy: 0.9988 - val_char3_accuracy: 0.9980 - val_char4_accuracy: 0.9988 - val_char5_accuracy: 0.9948 - val_char6_accuracy: 0.9860 - val_char7_accuracy: 0.9452 - val_char8_accuracy: 0.8876 - val_char9_accuracy: 0.8524 - val_char10_accuracy: 0.9216
    Epoch 9/150
    47500/47500 [==============================] - 7s 157us/sample - loss: 1.3403 - char0_loss: 0.0210 - char1_loss: 0.0151 - char2_loss: 0.0200 - char3_loss: 0.0245 - char4_loss: 0.0298 - char5_loss: 0.0558 - char6_loss: 0.0876 - char7_loss: 0.1246 - char8_loss: 0.2279 - char9_loss: 0.4726 - char10_loss: 0.2641 - char0_accuracy: 0.9992 - char1_accuracy: 0.9996 - char2_accuracy: 0.9996 - char3_accuracy: 0.9995 - char4_accuracy: 0.9992 - char5_accuracy: 0.9981 - char6_accuracy: 0.9936 - char7_accuracy: 0.9772 - char8_accuracy: 0.9278 - char9_accuracy: 0.8568 - char10_accuracy: 0.9254 - val_loss: 1.3036 - val_char0_loss: 0.0209 - val_char1_loss: 0.0153 - val_char2_loss: 0.0207 - val_char3_loss: 0.0234 - val_char4_loss: 0.0307 - val_char5_loss: 0.0533 - val_char6_loss: 0.0683 - val_char7_loss: 0.1120 - val_char8_loss: 0.2031 - val_char9_loss: 0.4708 - val_char10_loss: 0.2859 - val_char0_accuracy: 0.9992 - val_char1_accuracy: 0.9988 - val_char2_accuracy: 0.9984 - val_char3_accuracy: 0.9992 - val_char4_accuracy: 0.9988 - val_char5_accuracy: 0.9992 - val_char6_accuracy: 0.9968 - val_char7_accuracy: 0.9812 - val_char8_accuracy: 0.9280 - val_char9_accuracy: 0.8600 - val_char10_accuracy: 0.9276
    Epoch 10/150
    47500/47500 [==============================] - 8s 162us/sample - loss: 1.0267 - char0_loss: 0.0148 - char1_loss: 0.0116 - char2_loss: 0.0150 - char3_loss: 0.0175 - char4_loss: 0.0202 - char5_loss: 0.0367 - char6_loss: 0.0481 - char7_loss: 0.0653 - char8_loss: 0.1231 - char9_loss: 0.4246 - char10_loss: 0.2482 - char0_accuracy: 0.9995 - char1_accuracy: 0.9997 - char2_accuracy: 0.9996 - char3_accuracy: 0.9997 - char4_accuracy: 0.9999 - char5_accuracy: 0.9990 - char6_accuracy: 0.9983 - char7_accuracy: 0.9939 - char8_accuracy: 0.9679 - char9_accuracy: 0.8675 - char10_accuracy: 0.9278 - val_loss: 1.0592 - val_char0_loss: 0.0167 - val_char1_loss: 0.0155 - val_char2_loss: 0.0142 - val_char3_loss: 0.0177 - val_char4_loss: 0.0192 - val_char5_loss: 0.0320 - val_char6_loss: 0.0424 - val_char7_loss: 0.0702 - val_char8_loss: 0.1092 - val_char9_loss: 0.4486 - val_char10_loss: 0.2744 - val_char0_accuracy: 0.9992 - val_char1_accuracy: 0.9980 - val_char2_accuracy: 0.9992 - val_char3_accuracy: 0.9996 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 0.9988 - val_char6_accuracy: 0.9992 - val_char7_accuracy: 0.9912 - val_char8_accuracy: 0.9728 - val_char9_accuracy: 0.8660 - val_char10_accuracy: 0.9280
    Epoch 11/150
    47500/47500 [==============================] - 8s 169us/sample - loss: 0.8040 - char0_loss: 0.0111 - char1_loss: 0.0092 - char2_loss: 0.0111 - char3_loss: 0.0130 - char4_loss: 0.0144 - char5_loss: 0.0239 - char6_loss: 0.0309 - char7_loss: 0.0392 - char8_loss: 0.0655 - char9_loss: 0.3551 - char10_loss: 0.2292 - char0_accuracy: 0.9998 - char1_accuracy: 0.9997 - char2_accuracy: 0.9998 - char3_accuracy: 0.9997 - char4_accuracy: 0.9999 - char5_accuracy: 0.9997 - char6_accuracy: 0.9995 - char7_accuracy: 0.9982 - char8_accuracy: 0.9888 - char9_accuracy: 0.8853 - char10_accuracy: 0.9318 - val_loss: 0.8542 - val_char0_loss: 0.0182 - val_char1_loss: 0.0104 - val_char2_loss: 0.0103 - val_char3_loss: 0.0152 - val_char4_loss: 0.0146 - val_char5_loss: 0.0282 - val_char6_loss: 0.0315 - val_char7_loss: 0.0463 - val_char8_loss: 0.0677 - val_char9_loss: 0.3647 - val_char10_loss: 0.2470 - val_char0_accuracy: 0.9968 - val_char1_accuracy: 0.9996 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 0.9984 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 0.9984 - val_char6_accuracy: 0.9984 - val_char7_accuracy: 0.9964 - val_char8_accuracy: 0.9852 - val_char9_accuracy: 0.8812 - val_char10_accuracy: 0.9320
    Epoch 12/150
    47500/47500 [==============================] - 7s 157us/sample - loss: 0.6304 - char0_loss: 0.0087 - char1_loss: 0.0075 - char2_loss: 0.0091 - char3_loss: 0.0102 - char4_loss: 0.0114 - char5_loss: 0.0186 - char6_loss: 0.0216 - char7_loss: 0.0265 - char8_loss: 0.0389 - char9_loss: 0.2684 - char10_loss: 0.2090 - char0_accuracy: 0.9999 - char1_accuracy: 0.9998 - char2_accuracy: 0.9998 - char3_accuracy: 0.9998 - char4_accuracy: 1.0000 - char5_accuracy: 0.9996 - char6_accuracy: 0.9997 - char7_accuracy: 0.9990 - char8_accuracy: 0.9958 - char9_accuracy: 0.9105 - char10_accuracy: 0.9359 - val_loss: 0.7005 - val_char0_loss: 0.0136 - val_char1_loss: 0.0107 - val_char2_loss: 0.0115 - val_char3_loss: 0.0114 - val_char4_loss: 0.0151 - val_char5_loss: 0.0210 - val_char6_loss: 0.0234 - val_char7_loss: 0.0295 - val_char8_loss: 0.0550 - val_char9_loss: 0.2784 - val_char10_loss: 0.2324 - val_char0_accuracy: 0.9996 - val_char1_accuracy: 0.9992 - val_char2_accuracy: 0.9996 - val_char3_accuracy: 0.9992 - val_char4_accuracy: 0.9996 - val_char5_accuracy: 0.9988 - val_char6_accuracy: 0.9996 - val_char7_accuracy: 0.9980 - val_char8_accuracy: 0.9904 - val_char9_accuracy: 0.9004 - val_char10_accuracy: 0.9324
    Epoch 13/150
    47500/47500 [==============================] - 7s 156us/sample - loss: 0.4925 - char0_loss: 0.0069 - char1_loss: 0.0062 - char2_loss: 0.0078 - char3_loss: 0.0075 - char4_loss: 0.0088 - char5_loss: 0.0141 - char6_loss: 0.0161 - char7_loss: 0.0192 - char8_loss: 0.0266 - char9_loss: 0.1889 - char10_loss: 0.1896 - char0_accuracy: 0.9998 - char1_accuracy: 0.9999 - char2_accuracy: 0.9998 - char3_accuracy: 1.0000 - char4_accuracy: 0.9999 - char5_accuracy: 0.9998 - char6_accuracy: 0.9998 - char7_accuracy: 0.9995 - char8_accuracy: 0.9976 - char9_accuracy: 0.9389 - char10_accuracy: 0.9402 - val_loss: 0.5514 - val_char0_loss: 0.0091 - val_char1_loss: 0.0097 - val_char2_loss: 0.0067 - val_char3_loss: 0.0084 - val_char4_loss: 0.0096 - val_char5_loss: 0.0178 - val_char6_loss: 0.0162 - val_char7_loss: 0.0236 - val_char8_loss: 0.0360 - val_char9_loss: 0.1921 - val_char10_loss: 0.2249 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 0.9984 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 0.9988 - val_char4_accuracy: 0.9996 - val_char5_accuracy: 0.9992 - val_char6_accuracy: 0.9996 - val_char7_accuracy: 0.9984 - val_char8_accuracy: 0.9940 - val_char9_accuracy: 0.9352 - val_char10_accuracy: 0.9328
    Epoch 14/150
    47500/47500 [==============================] - 8s 159us/sample - loss: 0.3742 - char0_loss: 0.0064 - char1_loss: 0.0067 - char2_loss: 0.0057 - char3_loss: 0.0064 - char4_loss: 0.0074 - char5_loss: 0.0123 - char6_loss: 0.0134 - char7_loss: 0.0140 - char8_loss: 0.0191 - char9_loss: 0.1172 - char10_loss: 0.1652 - char0_accuracy: 0.9997 - char1_accuracy: 0.9994 - char2_accuracy: 0.9999 - char3_accuracy: 0.9999 - char4_accuracy: 0.9999 - char5_accuracy: 0.9997 - char6_accuracy: 0.9997 - char7_accuracy: 0.9997 - char8_accuracy: 0.9988 - char9_accuracy: 0.9655 - char10_accuracy: 0.9470 - val_loss: 0.4571 - val_char0_loss: 0.0132 - val_char1_loss: 0.0054 - val_char2_loss: 0.0092 - val_char3_loss: 0.0073 - val_char4_loss: 0.0105 - val_char5_loss: 0.0114 - val_char6_loss: 0.0135 - val_char7_loss: 0.0179 - val_char8_loss: 0.0253 - val_char9_loss: 0.1371 - val_char10_loss: 0.2082 - val_char0_accuracy: 0.9984 - val_char1_accuracy: 0.9988 - val_char2_accuracy: 0.9988 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 0.9988 - val_char5_accuracy: 0.9996 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 0.9988 - val_char8_accuracy: 0.9952 - val_char9_accuracy: 0.9552 - val_char10_accuracy: 0.9388
    Epoch 15/150
    47500/47500 [==============================] - 8s 165us/sample - loss: 0.2820 - char0_loss: 0.0057 - char1_loss: 0.0039 - char2_loss: 0.0056 - char3_loss: 0.0058 - char4_loss: 0.0065 - char5_loss: 0.0089 - char6_loss: 0.0094 - char7_loss: 0.0105 - char8_loss: 0.0141 - char9_loss: 0.0719 - char10_loss: 0.1389 - char0_accuracy: 0.9997 - char1_accuracy: 0.9999 - char2_accuracy: 0.9999 - char3_accuracy: 0.9999 - char4_accuracy: 0.9998 - char5_accuracy: 0.9997 - char6_accuracy: 0.9999 - char7_accuracy: 0.9998 - char8_accuracy: 0.9989 - char9_accuracy: 0.9822 - char10_accuracy: 0.9544 - val_loss: 0.3646 - val_char0_loss: 0.0048 - val_char1_loss: 0.0067 - val_char2_loss: 0.0077 - val_char3_loss: 0.0095 - val_char4_loss: 0.0072 - val_char5_loss: 0.0098 - val_char6_loss: 0.0116 - val_char7_loss: 0.0143 - val_char8_loss: 0.0237 - val_char9_loss: 0.0900 - val_char10_loss: 0.1836 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 0.9988 - val_char2_accuracy: 0.9992 - val_char3_accuracy: 0.9988 - val_char4_accuracy: 0.9996 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 0.9996 - val_char7_accuracy: 0.9996 - val_char8_accuracy: 0.9964 - val_char9_accuracy: 0.9744 - val_char10_accuracy: 0.9420
    Epoch 16/150
    47500/47500 [==============================] - 7s 148us/sample - loss: 0.7107 - char0_loss: 0.0274 - char1_loss: 0.0319 - char2_loss: 0.0517 - char3_loss: 0.0531 - char4_loss: 0.0535 - char5_loss: 0.0598 - char6_loss: 0.0636 - char7_loss: 0.0674 - char8_loss: 0.0658 - char9_loss: 0.0902 - char10_loss: 0.1447 - char0_accuracy: 0.9918 - char1_accuracy: 0.9912 - char2_accuracy: 0.9871 - char3_accuracy: 0.9871 - char4_accuracy: 0.9870 - char5_accuracy: 0.9872 - char6_accuracy: 0.9870 - char7_accuracy: 0.9861 - char8_accuracy: 0.9867 - char9_accuracy: 0.9807 - char10_accuracy: 0.9570 - val_loss: 0.3271 - val_char0_loss: 0.0089 - val_char1_loss: 0.0045 - val_char2_loss: 0.0048 - val_char3_loss: 0.0053 - val_char4_loss: 0.0052 - val_char5_loss: 0.0074 - val_char6_loss: 0.0110 - val_char7_loss: 0.0130 - val_char8_loss: 0.0187 - val_char9_loss: 0.0707 - val_char10_loss: 0.1813 - val_char0_accuracy: 0.9976 - val_char1_accuracy: 0.9996 - val_char2_accuracy: 0.9996 - val_char3_accuracy: 0.9992 - val_char4_accuracy: 0.9996 - val_char5_accuracy: 0.9992 - val_char6_accuracy: 0.9988 - val_char7_accuracy: 0.9996 - val_char8_accuracy: 0.9960 - val_char9_accuracy: 0.9808 - val_char10_accuracy: 0.9452
    Epoch 17/150
    47500/47500 [==============================] - 7s 155us/sample - loss: 0.1608 - char0_loss: 0.0021 - char1_loss: 0.0018 - char2_loss: 0.0019 - char3_loss: 0.0023 - char4_loss: 0.0028 - char5_loss: 0.0042 - char6_loss: 0.0052 - char7_loss: 0.0055 - char8_loss: 0.0071 - char9_loss: 0.0331 - char10_loss: 0.0951 - char0_accuracy: 1.0000 - char1_accuracy: 0.9999 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 0.9999 - char7_accuracy: 1.0000 - char8_accuracy: 0.9999 - char9_accuracy: 0.9950 - char10_accuracy: 0.9684 - val_loss: 0.2616 - val_char0_loss: 0.0035 - val_char1_loss: 0.0037 - val_char2_loss: 0.0051 - val_char3_loss: 0.0038 - val_char4_loss: 0.0038 - val_char5_loss: 0.0061 - val_char6_loss: 0.0061 - val_char7_loss: 0.0076 - val_char8_loss: 0.0140 - val_char9_loss: 0.0594 - val_char10_loss: 0.1489 - val_char0_accuracy: 0.9996 - val_char1_accuracy: 0.9996 - val_char2_accuracy: 0.9992 - val_char3_accuracy: 0.9996 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 0.9988 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 0.9996 - val_char8_accuracy: 0.9984 - val_char9_accuracy: 0.9840 - val_char10_accuracy: 0.9504
    Epoch 18/150
    47500/47500 [==============================] - 8s 167us/sample - loss: 0.1314 - char0_loss: 0.0017 - char1_loss: 0.0014 - char2_loss: 0.0016 - char3_loss: 0.0018 - char4_loss: 0.0023 - char5_loss: 0.0035 - char6_loss: 0.0040 - char7_loss: 0.0043 - char8_loss: 0.0052 - char9_loss: 0.0242 - char10_loss: 0.0812 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 0.9969 - char10_accuracy: 0.9728 - val_loss: 0.2387 - val_char0_loss: 0.0033 - val_char1_loss: 0.0028 - val_char2_loss: 0.0035 - val_char3_loss: 0.0029 - val_char4_loss: 0.0041 - val_char5_loss: 0.0058 - val_char6_loss: 0.0051 - val_char7_loss: 0.0075 - val_char8_loss: 0.0137 - val_char9_loss: 0.0475 - val_char10_loss: 0.1459 - val_char0_accuracy: 0.9996 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 0.9988 - val_char3_accuracy: 0.9996 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 0.9996 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 0.9992 - val_char8_accuracy: 0.9980 - val_char9_accuracy: 0.9884 - val_char10_accuracy: 0.9512
    Epoch 19/150
    47500/47500 [==============================] - 7s 156us/sample - loss: 0.1045 - char0_loss: 0.0015 - char1_loss: 0.0016 - char2_loss: 0.0014 - char3_loss: 0.0016 - char4_loss: 0.0020 - char5_loss: 0.0027 - char6_loss: 0.0033 - char7_loss: 0.0035 - char8_loss: 0.0044 - char9_loss: 0.0167 - char10_loss: 0.0658 - char0_accuracy: 1.0000 - char1_accuracy: 0.9999 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 0.9984 - char10_accuracy: 0.9784 - val_loss: 0.2031 - val_char0_loss: 0.0029 - val_char1_loss: 0.0027 - val_char2_loss: 0.0034 - val_char3_loss: 0.0040 - val_char4_loss: 0.0052 - val_char5_loss: 0.0043 - val_char6_loss: 0.0055 - val_char7_loss: 0.0058 - val_char8_loss: 0.0113 - val_char9_loss: 0.0410 - val_char10_loss: 0.1182 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 0.9996 - val_char2_accuracy: 0.9988 - val_char3_accuracy: 0.9992 - val_char4_accuracy: 0.9996 - val_char5_accuracy: 0.9996 - val_char6_accuracy: 0.9996 - val_char7_accuracy: 0.9996 - val_char8_accuracy: 0.9980 - val_char9_accuracy: 0.9900 - val_char10_accuracy: 0.9604
    Epoch 20/150
    47500/47500 [==============================] - 7s 155us/sample - loss: 0.0969 - char0_loss: 0.0016 - char1_loss: 0.0134 - char2_loss: 0.0015 - char3_loss: 0.0016 - char4_loss: 0.0020 - char5_loss: 0.0026 - char6_loss: 0.0034 - char7_loss: 0.0033 - char8_loss: 0.0041 - char9_loss: 0.0134 - char10_loss: 0.0501 - char0_accuracy: 1.0000 - char1_accuracy: 0.9958 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 0.9990 - char10_accuracy: 0.9842 - val_loss: 0.1943 - val_char0_loss: 0.0045 - val_char1_loss: 0.0169 - val_char2_loss: 0.0045 - val_char3_loss: 0.0039 - val_char4_loss: 0.0032 - val_char5_loss: 0.0063 - val_char6_loss: 0.0066 - val_char7_loss: 0.0064 - val_char8_loss: 0.0127 - val_char9_loss: 0.0325 - val_char10_loss: 0.0993 - val_char0_accuracy: 0.9996 - val_char1_accuracy: 0.9940 - val_char2_accuracy: 0.9992 - val_char3_accuracy: 0.9996 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 0.9996 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 0.9996 - val_char8_accuracy: 0.9980 - val_char9_accuracy: 0.9940 - val_char10_accuracy: 0.9664
    Epoch 21/150
    47500/47500 [==============================] - 7s 156us/sample - loss: 0.0728 - char0_loss: 0.0014 - char1_loss: 0.0021 - char2_loss: 0.0014 - char3_loss: 0.0014 - char4_loss: 0.0018 - char5_loss: 0.0024 - char6_loss: 0.0029 - char7_loss: 0.0028 - char8_loss: 0.0039 - char9_loss: 0.0109 - char10_loss: 0.0416 - char0_accuracy: 1.0000 - char1_accuracy: 0.9996 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 0.9999 - char9_accuracy: 0.9995 - char10_accuracy: 0.9864 - val_loss: 0.1592 - val_char0_loss: 0.0033 - val_char1_loss: 0.0014 - val_char2_loss: 0.0029 - val_char3_loss: 0.0026 - val_char4_loss: 0.0029 - val_char5_loss: 0.0043 - val_char6_loss: 0.0040 - val_char7_loss: 0.0056 - val_char8_loss: 0.0109 - val_char9_loss: 0.0284 - val_char10_loss: 0.0951 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 0.9992 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 0.9996 - val_char8_accuracy: 0.9984 - val_char9_accuracy: 0.9928 - val_char10_accuracy: 0.9696
    Epoch 22/150
    47500/47500 [==============================] - 8s 168us/sample - loss: 0.0992 - char0_loss: 0.0321 - char1_loss: 0.0010 - char2_loss: 0.0032 - char3_loss: 0.0026 - char4_loss: 0.0040 - char5_loss: 0.0034 - char6_loss: 0.0034 - char7_loss: 0.0037 - char8_loss: 0.0042 - char9_loss: 0.0093 - char10_loss: 0.0321 - char0_accuracy: 0.9892 - char1_accuracy: 1.0000 - char2_accuracy: 0.9996 - char3_accuracy: 0.9997 - char4_accuracy: 0.9996 - char5_accuracy: 0.9999 - char6_accuracy: 1.0000 - char7_accuracy: 0.9999 - char8_accuracy: 0.9998 - char9_accuracy: 0.9995 - char10_accuracy: 0.9905 - val_loss: 0.2562 - val_char0_loss: 0.0132 - val_char1_loss: 0.0045 - val_char2_loss: 0.0209 - val_char3_loss: 0.0233 - val_char4_loss: 0.0099 - val_char5_loss: 0.0114 - val_char6_loss: 0.0107 - val_char7_loss: 0.0125 - val_char8_loss: 0.0238 - val_char9_loss: 0.0370 - val_char10_loss: 0.0900 - val_char0_accuracy: 0.9980 - val_char1_accuracy: 0.9992 - val_char2_accuracy: 0.9936 - val_char3_accuracy: 0.9956 - val_char4_accuracy: 0.9976 - val_char5_accuracy: 0.9996 - val_char6_accuracy: 0.9980 - val_char7_accuracy: 0.9984 - val_char8_accuracy: 0.9944 - val_char9_accuracy: 0.9884 - val_char10_accuracy: 0.9676
    Epoch 23/150
    47500/47500 [==============================] - 7s 153us/sample - loss: 0.1046 - char0_loss: 0.0021 - char1_loss: 0.0017 - char2_loss: 0.0110 - char3_loss: 0.0144 - char4_loss: 0.0085 - char5_loss: 0.0091 - char6_loss: 0.0045 - char7_loss: 0.0097 - char8_loss: 0.0058 - char9_loss: 0.0102 - char10_loss: 0.0274 - char0_accuracy: 0.9997 - char1_accuracy: 0.9999 - char2_accuracy: 0.9968 - char3_accuracy: 0.9955 - char4_accuracy: 0.9983 - char5_accuracy: 0.9985 - char6_accuracy: 0.9999 - char7_accuracy: 0.9984 - char8_accuracy: 0.9997 - char9_accuracy: 0.9994 - char10_accuracy: 0.9925 - val_loss: 0.1434 - val_char0_loss: 0.0015 - val_char1_loss: 0.0027 - val_char2_loss: 0.0033 - val_char3_loss: 0.0031 - val_char4_loss: 0.0016 - val_char5_loss: 0.0048 - val_char6_loss: 0.0065 - val_char7_loss: 0.0046 - val_char8_loss: 0.0100 - val_char9_loss: 0.0294 - val_char10_loss: 0.0778 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 0.9996 - val_char2_accuracy: 0.9992 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 0.9988 - val_char7_accuracy: 0.9992 - val_char8_accuracy: 0.9988 - val_char9_accuracy: 0.9908 - val_char10_accuracy: 0.9732
    Epoch 24/150
    47500/47500 [==============================] - 7s 154us/sample - loss: 0.0836 - char0_loss: 7.2375e-04 - char1_loss: 0.0298 - char2_loss: 0.0015 - char3_loss: 0.0013 - char4_loss: 0.0017 - char5_loss: 0.0028 - char6_loss: 0.0100 - char7_loss: 0.0031 - char8_loss: 0.0035 - char9_loss: 0.0080 - char10_loss: 0.0211 - char0_accuracy: 1.0000 - char1_accuracy: 0.9924 - char2_accuracy: 0.9999 - char3_accuracy: 1.0000 - char4_accuracy: 0.9999 - char5_accuracy: 0.9998 - char6_accuracy: 0.9979 - char7_accuracy: 0.9999 - char8_accuracy: 1.0000 - char9_accuracy: 0.9993 - char10_accuracy: 0.9948 - val_loss: 0.1563 - val_char0_loss: 0.0016 - val_char1_loss: 0.0018 - val_char2_loss: 0.0039 - val_char3_loss: 0.0028 - val_char4_loss: 0.0015 - val_char5_loss: 0.0029 - val_char6_loss: 0.0263 - val_char7_loss: 0.0046 - val_char8_loss: 0.0101 - val_char9_loss: 0.0265 - val_char10_loss: 0.0757 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 0.9996 - val_char2_accuracy: 0.9988 - val_char3_accuracy: 0.9992 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 0.9880 - val_char7_accuracy: 0.9992 - val_char8_accuracy: 0.9976 - val_char9_accuracy: 0.9912 - val_char10_accuracy: 0.9696
    Epoch 25/150
    47500/47500 [==============================] - 7s 156us/sample - loss: 0.0674 - char0_loss: 7.8518e-04 - char1_loss: 7.4969e-04 - char2_loss: 0.0011 - char3_loss: 0.0010 - char4_loss: 0.0013 - char5_loss: 0.0023 - char6_loss: 0.0187 - char7_loss: 0.0022 - char8_loss: 0.0130 - char9_loss: 0.0081 - char10_loss: 0.0182 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 0.9999 - char6_accuracy: 0.9937 - char7_accuracy: 1.0000 - char8_accuracy: 0.9966 - char9_accuracy: 0.9993 - char10_accuracy: 0.9956 - val_loss: 0.1527 - val_char0_loss: 0.0020 - val_char1_loss: 0.0013 - val_char2_loss: 0.0036 - val_char3_loss: 0.0018 - val_char4_loss: 0.0025 - val_char5_loss: 0.0042 - val_char6_loss: 0.0033 - val_char7_loss: 0.0043 - val_char8_loss: 0.0300 - val_char9_loss: 0.0320 - val_char10_loss: 0.0700 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 0.9992 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 0.9996 - val_char5_accuracy: 0.9992 - val_char6_accuracy: 0.9992 - val_char7_accuracy: 0.9992 - val_char8_accuracy: 0.9908 - val_char9_accuracy: 0.9896 - val_char10_accuracy: 0.9744
    Epoch 26/150
    47500/47500 [==============================] - 8s 177us/sample - loss: 0.0826 - char0_loss: 0.0010 - char1_loss: 9.1802e-04 - char2_loss: 0.0017 - char3_loss: 0.0036 - char4_loss: 0.0062 - char5_loss: 0.0226 - char6_loss: 0.0019 - char7_loss: 0.0111 - char8_loss: 0.0058 - char9_loss: 0.0118 - char10_loss: 0.0167 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 0.9999 - char3_accuracy: 0.9994 - char4_accuracy: 0.9985 - char5_accuracy: 0.9931 - char6_accuracy: 0.9999 - char7_accuracy: 0.9968 - char8_accuracy: 0.9987 - char9_accuracy: 0.9975 - char10_accuracy: 0.9958 - val_loss: 0.3589 - val_char0_loss: 0.0075 - val_char1_loss: 0.0042 - val_char2_loss: 0.0083 - val_char3_loss: 0.0129 - val_char4_loss: 0.0800 - val_char5_loss: 0.0112 - val_char6_loss: 0.0071 - val_char7_loss: 0.0858 - val_char8_loss: 0.0214 - val_char9_loss: 0.0339 - val_char10_loss: 0.0878 - val_char0_accuracy: 0.9984 - val_char1_accuracy: 0.9992 - val_char2_accuracy: 0.9992 - val_char3_accuracy: 0.9960 - val_char4_accuracy: 0.9732 - val_char5_accuracy: 0.9980 - val_char6_accuracy: 0.9988 - val_char7_accuracy: 0.9680 - val_char8_accuracy: 0.9932 - val_char9_accuracy: 0.9904 - val_char10_accuracy: 0.9728



{% highlight python %}
# plot every loss and accuracy

def plot_all_accuracy_and_loss(hist):
    for h in hist.history:
        if h.startswith('val_') and h.endswith('_accuracy'):
            plt.plot(hist.history[h], label=h[4:-9])
    plt.title("Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc="best")
    plt.show()

    for h in hist.history:
        if h.startswith('val_') and h.endswith('_loss'):
            plt.plot(hist.history[h], label=h[4:-5])
    plt.title("Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc="best")
    plt.show()

plot_all_accuracy_and_loss(mo_hist)

# plot diff in loss and val_loss
plot_loss_vs_validation_loss_diff(mo_hist)
plot_loss_vs_validation_loss(mo_hist)
{% endhighlight %}


![png](/assets/qr-nn/output_64_0.png)



![png](/assets/qr-nn/output_64_1.png)



![png](/assets/qr-nn/output_64_2.png)



![png](/assets/qr-nn/output_64_3.png)


Bad news: the NN never arrived at the desired accuracy of 100%. The good news: there is no overfitting, hence the NN underfits. Probably due to design. Another red flag is the performance of the first character, it also did not arriver to 100%. Since all outputs influence the same layers during training, this is a hint for improving the design.

Let's try more an NN with more hidden layers:


{% highlight python %}
#                                       |-> char 0 output
# INPUT -> FLATTEN -> HIDDEN LAYER x 3 -|-> ...
#                                       |-> char 10 output
#
#
def define_deeper_multi_output_model():
    input_layer = keras.layers.Input(shape=(21,21,), dtype='float', name='input_qr')
    flatten = keras.layers.Flatten(input_shape=(21, 21), name='flatten')(input_layer)    
    hidden_chars1 = keras.layers.Dense(21*21, activation='relu', name='hidden_chars1')(flatten)
    hidden_chars2 = keras.layers.Dense(21*21, activation='relu', name='hidden_chars2')(hidden_chars1)
    hidden_chars3 = keras.layers.Dense(21*21, activation='relu', name='hidden_chars3')(hidden_chars2)

    outputs = []
    for i in range(11):
        char_output = keras.layers.Dense(len(ALL_CHAR_CLASSES), name='char' + str(i))(hidden_chars3)
        outputs.append(char_output)

    multi_output_model = keras.Model(inputs=[input_layer], outputs=outputs)

    multi_output_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return multi_output_model

mo_model = define_deeper_multi_output_model()
mo_model.summary()
mo_hist = mo_model.fit(
        training_data, [
            np.asarray(training_labels_char0),
            np.asarray(training_labels_char1),
            np.asarray(training_labels_char2),
            np.asarray(training_labels_char3),
            np.asarray(training_labels_char4),
            np.asarray(training_labels_char5),
            np.asarray(training_labels_char6),
            np.asarray(training_labels_char7),
            np.asarray(training_labels_char8),
            np.asarray(training_labels_char9),
            np.asarray(training_labels_char10)
        ],
        epochs=150, batch_size=128,
        validation_split=0.05,
        callbacks=[EarlyStopping(monitor='val_loss' , min_delta=0.0001, patience=3)]
    )

plot_all_accuracy_and_loss(mo_hist)
plot_loss_vs_validation_loss_diff(mo_hist)
plot_loss_vs_validation_loss(mo_hist)
{% endhighlight %}

    Model: "model_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_qr (InputLayer)           [(None, 21, 21)]     0                                            
    __________________________________________________________________________________________________
    flatten (Flatten)               (None, 441)          0           input_qr[0][0]                   
    __________________________________________________________________________________________________
    hidden_chars1 (Dense)           (None, 441)          194922      flatten[0][0]                    
    __________________________________________________________________________________________________
    hidden_chars2 (Dense)           (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    hidden_chars3 (Dense)           (None, 441)          194922      hidden_chars2[0][0]              
    __________________________________________________________________________________________________
    char0 (Dense)                   (None, 28)           12376       hidden_chars3[0][0]              
    __________________________________________________________________________________________________
    char1 (Dense)                   (None, 28)           12376       hidden_chars3[0][0]              
    __________________________________________________________________________________________________
    char2 (Dense)                   (None, 28)           12376       hidden_chars3[0][0]              
    __________________________________________________________________________________________________
    char3 (Dense)                   (None, 28)           12376       hidden_chars3[0][0]              
    __________________________________________________________________________________________________
    char4 (Dense)                   (None, 28)           12376       hidden_chars3[0][0]              
    __________________________________________________________________________________________________
    char5 (Dense)                   (None, 28)           12376       hidden_chars3[0][0]              
    __________________________________________________________________________________________________
    char6 (Dense)                   (None, 28)           12376       hidden_chars3[0][0]              
    __________________________________________________________________________________________________
    char7 (Dense)                   (None, 28)           12376       hidden_chars3[0][0]              
    __________________________________________________________________________________________________
    char8 (Dense)                   (None, 28)           12376       hidden_chars3[0][0]              
    __________________________________________________________________________________________________
    char9 (Dense)                   (None, 28)           12376       hidden_chars3[0][0]              
    __________________________________________________________________________________________________
    char10 (Dense)                  (None, 28)           12376       hidden_chars3[0][0]              
    ==================================================================================================
    Total params: 720,902
    Trainable params: 720,902
    Non-trainable params: 0
    __________________________________________________________________________________________________
    Train on 47500 samples, validate on 2500 samples
    Epoch 1/150
    47500/47500 [==============================] - 12s 256us/sample - loss: 20.5515 - char0_loss: 3.0903 - char1_loss: 2.9193 - char2_loss: 2.8085 - char3_loss: 2.4853 - char4_loss: 2.2262 - char5_loss: 1.9173 - char6_loss: 1.6212 - char7_loss: 1.3288 - char8_loss: 1.0298 - char9_loss: 0.7235 - char10_loss: 0.3880 - char0_accuracy: 0.1043 - char1_accuracy: 0.1657 - char2_accuracy: 0.2093 - char3_accuracy: 0.3008 - char4_accuracy: 0.3801 - char5_accuracy: 0.4715 - char6_accuracy: 0.5549 - char7_accuracy: 0.6415 - char8_accuracy: 0.7289 - char9_accuracy: 0.8169 - char10_accuracy: 0.9070 - val_loss: 18.3684 - val_char0_loss: 2.6243 - val_char1_loss: 2.5189 - val_char2_loss: 2.6350 - val_char3_loss: 2.2872 - val_char4_loss: 2.0964 - val_char5_loss: 1.7834 - val_char6_loss: 1.4992 - val_char7_loss: 1.1615 - val_char8_loss: 0.8651 - val_char9_loss: 0.5754 - val_char10_loss: 0.2979 - val_char0_accuracy: 0.1848 - val_char1_accuracy: 0.2368 - val_char2_accuracy: 0.2412 - val_char3_accuracy: 0.3316 - val_char4_accuracy: 0.3944 - val_char5_accuracy: 0.4920 - val_char6_accuracy: 0.5692 - val_char7_accuracy: 0.6664 - val_char8_accuracy: 0.7572 - val_char9_accuracy: 0.8396 - val_char10_accuracy: 0.9216
    Epoch 2/150
    47500/47500 [==============================] - 8s 178us/sample - loss: 16.9141 - char0_loss: 2.1303 - char1_loss: 1.7195 - char2_loss: 2.4173 - char3_loss: 2.0741 - char4_loss: 2.0704 - char5_loss: 1.8242 - char6_loss: 1.5439 - char7_loss: 1.2408 - char8_loss: 0.9342 - char9_loss: 0.6331 - char10_loss: 0.3209 - char0_accuracy: 0.2642 - char1_accuracy: 0.4438 - char2_accuracy: 0.2636 - char3_accuracy: 0.3528 - char4_accuracy: 0.4013 - char5_accuracy: 0.4830 - char6_accuracy: 0.5633 - char7_accuracy: 0.6508 - char8_accuracy: 0.7370 - char9_accuracy: 0.8237 - char10_accuracy: 0.9123 - val_loss: 14.4775 - val_char0_loss: 1.5531 - val_char1_loss: 0.9097 - val_char2_loss: 2.0205 - val_char3_loss: 1.8097 - val_char4_loss: 1.9247 - val_char5_loss: 1.7686 - val_char6_loss: 1.5088 - val_char7_loss: 1.1895 - val_char8_loss: 0.8792 - val_char9_loss: 0.5846 - val_char10_loss: 0.3059 - val_char0_accuracy: 0.3748 - val_char1_accuracy: 0.6836 - val_char2_accuracy: 0.3264 - val_char3_accuracy: 0.3944 - val_char4_accuracy: 0.4260 - val_char5_accuracy: 0.4972 - val_char6_accuracy: 0.5704 - val_char7_accuracy: 0.6680 - val_char8_accuracy: 0.7536 - val_char9_accuracy: 0.8408 - val_char10_accuracy: 0.9188
    Epoch 3/150
    47500/47500 [==============================] - 8s 177us/sample - loss: 12.4124 - char0_loss: 1.0321 - char1_loss: 0.3866 - char2_loss: 1.3974 - char3_loss: 1.4410 - char4_loss: 1.7876 - char5_loss: 1.7234 - char6_loss: 1.5258 - char7_loss: 1.2327 - char8_loss: 0.9309 - char9_loss: 0.6341 - char10_loss: 0.3198 - char0_accuracy: 0.5721 - char1_accuracy: 0.9046 - char2_accuracy: 0.5141 - char3_accuracy: 0.4704 - char4_accuracy: 0.4373 - char5_accuracy: 0.4985 - char6_accuracy: 0.5689 - char7_accuracy: 0.6548 - char8_accuracy: 0.7404 - char9_accuracy: 0.8262 - char10_accuracy: 0.9139 - val_loss: 10.1614 - val_char0_loss: 0.4656 - val_char1_loss: 0.1305 - val_char2_loss: 0.6990 - val_char3_loss: 1.1557 - val_char4_loss: 1.6397 - val_char5_loss: 1.6249 - val_char6_loss: 1.4757 - val_char7_loss: 1.1762 - val_char8_loss: 0.8835 - val_char9_loss: 0.5942 - val_char10_loss: 0.2995 - val_char0_accuracy: 0.8544 - val_char1_accuracy: 0.9932 - val_char2_accuracy: 0.7580 - val_char3_accuracy: 0.5420 - val_char4_accuracy: 0.4644 - val_char5_accuracy: 0.5088 - val_char6_accuracy: 0.5820 - val_char7_accuracy: 0.6688 - val_char8_accuracy: 0.7600 - val_char9_accuracy: 0.8428 - val_char10_accuracy: 0.9188
    Epoch 4/150
    47500/47500 [==============================] - 9s 182us/sample - loss: 8.6350 - char0_loss: 0.1650 - char1_loss: 0.0867 - char2_loss: 0.3095 - char3_loss: 0.7645 - char4_loss: 1.3601 - char5_loss: 1.4372 - char6_loss: 1.4507 - char7_loss: 1.1946 - char8_loss: 0.9068 - char9_loss: 0.6355 - char10_loss: 0.3183 - char0_accuracy: 0.9784 - char1_accuracy: 0.9943 - char2_accuracy: 0.9182 - char3_accuracy: 0.6755 - char4_accuracy: 0.4985 - char5_accuracy: 0.5460 - char6_accuracy: 0.5812 - char7_accuracy: 0.6624 - char8_accuracy: 0.7465 - char9_accuracy: 0.8274 - char10_accuracy: 0.9146 - val_loss: 7.2988 - val_char0_loss: 0.0863 - val_char1_loss: 0.0708 - val_char2_loss: 0.1256 - val_char3_loss: 0.4714 - val_char4_loss: 1.1517 - val_char5_loss: 1.1962 - val_char6_loss: 1.3647 - val_char7_loss: 1.1035 - val_char8_loss: 0.8466 - val_char9_loss: 0.5864 - val_char10_loss: 0.2841 - val_char0_accuracy: 0.9944 - val_char1_accuracy: 0.9956 - val_char2_accuracy: 0.9888 - val_char3_accuracy: 0.8120 - val_char4_accuracy: 0.5496 - val_char5_accuracy: 0.5956 - val_char6_accuracy: 0.5992 - val_char7_accuracy: 0.6820 - val_char8_accuracy: 0.7640 - val_char9_accuracy: 0.8416 - val_char10_accuracy: 0.9264
    Epoch 5/150
    47500/47500 [==============================] - 8s 173us/sample - loss: 6.3297 - char0_loss: 0.0626 - char1_loss: 0.0550 - char2_loss: 0.0912 - char3_loss: 0.2177 - char4_loss: 0.8803 - char5_loss: 0.8342 - char6_loss: 1.3057 - char7_loss: 1.0804 - char8_loss: 0.8443 - char9_loss: 0.6410 - char10_loss: 0.3178 - char0_accuracy: 0.9963 - char1_accuracy: 0.9956 - char2_accuracy: 0.9922 - char3_accuracy: 0.9457 - char4_accuracy: 0.6617 - char5_accuracy: 0.7088 - char6_accuracy: 0.5996 - char7_accuracy: 0.6787 - char8_accuracy: 0.7548 - char9_accuracy: 0.8275 - char10_accuracy: 0.9152 - val_loss: 5.0750 - val_char0_loss: 0.0399 - val_char1_loss: 0.0515 - val_char2_loss: 0.0610 - val_char3_loss: 0.0954 - val_char4_loss: 0.5434 - val_char5_loss: 0.4513 - val_char6_loss: 1.2279 - val_char7_loss: 0.9569 - val_char8_loss: 0.7484 - val_char9_loss: 0.5982 - val_char10_loss: 0.2901 - val_char0_accuracy: 0.9972 - val_char1_accuracy: 0.9952 - val_char2_accuracy: 0.9936 - val_char3_accuracy: 0.9916 - val_char4_accuracy: 0.7976 - val_char5_accuracy: 0.8508 - val_char6_accuracy: 0.6208 - val_char7_accuracy: 0.7020 - val_char8_accuracy: 0.7784 - val_char9_accuracy: 0.8440 - val_char10_accuracy: 0.9236
    Epoch 6/150
    47500/47500 [==============================] - 8s 173us/sample - loss: 4.2026 - char0_loss: 0.0346 - char1_loss: 0.0456 - char2_loss: 0.0466 - char3_loss: 0.0605 - char4_loss: 0.2330 - char5_loss: 0.2024 - char6_loss: 1.0327 - char7_loss: 0.8776 - char8_loss: 0.7231 - char9_loss: 0.6297 - char10_loss: 0.3108 - char0_accuracy: 0.9980 - char1_accuracy: 0.9925 - char2_accuracy: 0.9966 - char3_accuracy: 0.9957 - char4_accuracy: 0.9410 - char5_accuracy: 0.9600 - char6_accuracy: 0.6630 - char7_accuracy: 0.7174 - char8_accuracy: 0.7793 - char9_accuracy: 0.8305 - char10_accuracy: 0.9177 - val_loss: 4.0817 - val_char0_loss: 0.0322 - val_char1_loss: 0.0750 - val_char2_loss: 0.0730 - val_char3_loss: 0.1104 - val_char4_loss: 0.3313 - val_char5_loss: 0.1883 - val_char6_loss: 0.9100 - val_char7_loss: 0.7924 - val_char8_loss: 0.6329 - val_char9_loss: 0.6196 - val_char10_loss: 0.3055 - val_char0_accuracy: 0.9972 - val_char1_accuracy: 0.9820 - val_char2_accuracy: 0.9876 - val_char3_accuracy: 0.9748 - val_char4_accuracy: 0.9144 - val_char5_accuracy: 0.9716 - val_char6_accuracy: 0.7052 - val_char7_accuracy: 0.7456 - val_char8_accuracy: 0.8008 - val_char9_accuracy: 0.8440 - val_char10_accuracy: 0.9232
    Epoch 7/150
    47500/47500 [==============================] - 9s 179us/sample - loss: 3.2960 - char0_loss: 0.0370 - char1_loss: 0.0569 - char2_loss: 0.0651 - char3_loss: 0.0812 - char4_loss: 0.1251 - char5_loss: 0.1265 - char6_loss: 0.5794 - char7_loss: 0.6646 - char8_loss: 0.6151 - char9_loss: 0.6289 - char10_loss: 0.3145 - char0_accuracy: 0.9958 - char1_accuracy: 0.9859 - char2_accuracy: 0.9890 - char3_accuracy: 0.9861 - char4_accuracy: 0.9804 - char5_accuracy: 0.9788 - char6_accuracy: 0.8119 - char7_accuracy: 0.7777 - char8_accuracy: 0.8040 - char9_accuracy: 0.8321 - char10_accuracy: 0.9177 - val_loss: 2.4376 - val_char0_loss: 0.0198 - val_char1_loss: 0.0396 - val_char2_loss: 0.0289 - val_char3_loss: 0.0244 - val_char4_loss: 0.0542 - val_char5_loss: 0.0532 - val_char6_loss: 0.3107 - val_char7_loss: 0.4906 - val_char8_loss: 0.5219 - val_char9_loss: 0.5931 - val_char10_loss: 0.2979 - val_char0_accuracy: 0.9992 - val_char1_accuracy: 0.9920 - val_char2_accuracy: 0.9960 - val_char3_accuracy: 0.9976 - val_char4_accuracy: 0.9956 - val_char5_accuracy: 0.9944 - val_char6_accuracy: 0.8956 - val_char7_accuracy: 0.8192 - val_char8_accuracy: 0.8272 - val_char9_accuracy: 0.8416 - val_char10_accuracy: 0.9236
    Epoch 8/150
    47500/47500 [==============================] - 9s 194us/sample - loss: 1.9668 - char0_loss: 0.0174 - char1_loss: 0.0284 - char2_loss: 0.0213 - char3_loss: 0.0242 - char4_loss: 0.0354 - char5_loss: 0.0368 - char6_loss: 0.1581 - char7_loss: 0.3167 - char8_loss: 0.4304 - char9_loss: 0.6021 - char10_loss: 0.2963 - char0_accuracy: 0.9986 - char1_accuracy: 0.9934 - char2_accuracy: 0.9981 - char3_accuracy: 0.9979 - char4_accuracy: 0.9976 - char5_accuracy: 0.9979 - char6_accuracy: 0.9639 - char7_accuracy: 0.8923 - char8_accuracy: 0.8520 - char9_accuracy: 0.8350 - char10_accuracy: 0.9206 - val_loss: 1.7152 - val_char0_loss: 0.0127 - val_char1_loss: 0.0240 - val_char2_loss: 0.0215 - val_char3_loss: 0.0289 - val_char4_loss: 0.0339 - val_char5_loss: 0.0325 - val_char6_loss: 0.0848 - val_char7_loss: 0.2359 - val_char8_loss: 0.3446 - val_char9_loss: 0.6034 - val_char10_loss: 0.2861 - val_char0_accuracy: 0.9980 - val_char1_accuracy: 0.9960 - val_char2_accuracy: 0.9968 - val_char3_accuracy: 0.9960 - val_char4_accuracy: 0.9972 - val_char5_accuracy: 0.9988 - val_char6_accuracy: 0.9904 - val_char7_accuracy: 0.9124 - val_char8_accuracy: 0.8792 - val_char9_accuracy: 0.8452 - val_char10_accuracy: 0.9228
    Epoch 9/150
    47500/47500 [==============================] - 9s 185us/sample - loss: 1.4013 - char0_loss: 0.0116 - char1_loss: 0.0201 - char2_loss: 0.0153 - char3_loss: 0.0175 - char4_loss: 0.0225 - char5_loss: 0.0246 - char6_loss: 0.0533 - char7_loss: 0.1156 - char8_loss: 0.2627 - char9_loss: 0.5680 - char10_loss: 0.2914 - char0_accuracy: 0.9992 - char1_accuracy: 0.9955 - char2_accuracy: 0.9989 - char3_accuracy: 0.9989 - char4_accuracy: 0.9987 - char5_accuracy: 0.9985 - char6_accuracy: 0.9956 - char7_accuracy: 0.9721 - char8_accuracy: 0.9083 - char9_accuracy: 0.8404 - char10_accuracy: 0.9208 - val_loss: 1.7453 - val_char0_loss: 0.0134 - val_char1_loss: 0.4213 - val_char2_loss: 0.0221 - val_char3_loss: 0.0251 - val_char4_loss: 0.0290 - val_char5_loss: 0.0261 - val_char6_loss: 0.0508 - val_char7_loss: 0.0953 - val_char8_loss: 0.2188 - val_char9_loss: 0.5364 - val_char10_loss: 0.3044 - val_char0_accuracy: 0.9984 - val_char1_accuracy: 0.8712 - val_char2_accuracy: 0.9972 - val_char3_accuracy: 0.9952 - val_char4_accuracy: 0.9972 - val_char5_accuracy: 0.9956 - val_char6_accuracy: 0.9928 - val_char7_accuracy: 0.9792 - val_char8_accuracy: 0.9216 - val_char9_accuracy: 0.8500 - val_char10_accuracy: 0.9236
    Epoch 10/150
    47500/47500 [==============================] - 9s 189us/sample - loss: 1.1827 - char0_loss: 0.0112 - char1_loss: 0.0817 - char2_loss: 0.0149 - char3_loss: 0.0164 - char4_loss: 0.0193 - char5_loss: 0.0199 - char6_loss: 0.0330 - char7_loss: 0.0554 - char8_loss: 0.1390 - char9_loss: 0.5208 - char10_loss: 0.2688 - char0_accuracy: 0.9991 - char1_accuracy: 0.9792 - char2_accuracy: 0.9981 - char3_accuracy: 0.9983 - char4_accuracy: 0.9987 - char5_accuracy: 0.9988 - char6_accuracy: 0.9978 - char7_accuracy: 0.9920 - char8_accuracy: 0.9552 - char9_accuracy: 0.8461 - char10_accuracy: 0.9250 - val_loss: 1.0851 - val_char0_loss: 0.0082 - val_char1_loss: 0.0052 - val_char2_loss: 0.0137 - val_char3_loss: 0.0131 - val_char4_loss: 0.0170 - val_char5_loss: 0.0158 - val_char6_loss: 0.0267 - val_char7_loss: 0.0425 - val_char8_loss: 0.1179 - val_char9_loss: 0.5343 - val_char10_loss: 0.2933 - val_char0_accuracy: 0.9988 - val_char1_accuracy: 0.9992 - val_char2_accuracy: 0.9972 - val_char3_accuracy: 0.9976 - val_char4_accuracy: 0.9988 - val_char5_accuracy: 0.9984 - val_char6_accuracy: 0.9976 - val_char7_accuracy: 0.9940 - val_char8_accuracy: 0.9620 - val_char9_accuracy: 0.8540 - val_char10_accuracy: 0.9240
    Epoch 11/150
    47500/47500 [==============================] - 9s 180us/sample - loss: 0.8848 - char0_loss: 0.0062 - char1_loss: 0.0039 - char2_loss: 0.0085 - char3_loss: 0.0099 - char4_loss: 0.0115 - char5_loss: 0.0132 - char6_loss: 0.0181 - char7_loss: 0.0272 - char8_loss: 0.0628 - char9_loss: 0.4657 - char10_loss: 0.2568 - char0_accuracy: 0.9996 - char1_accuracy: 0.9997 - char2_accuracy: 0.9993 - char3_accuracy: 0.9992 - char4_accuracy: 0.9996 - char5_accuracy: 0.9991 - char6_accuracy: 0.9995 - char7_accuracy: 0.9979 - char8_accuracy: 0.9858 - char9_accuracy: 0.8570 - char10_accuracy: 0.9266 - val_loss: 0.9428 - val_char0_loss: 0.0080 - val_char1_loss: 0.0073 - val_char2_loss: 0.0234 - val_char3_loss: 0.0121 - val_char4_loss: 0.0153 - val_char5_loss: 0.0151 - val_char6_loss: 0.0189 - val_char7_loss: 0.0315 - val_char8_loss: 0.0618 - val_char9_loss: 0.4581 - val_char10_loss: 0.2927 - val_char0_accuracy: 0.9996 - val_char1_accuracy: 0.9984 - val_char2_accuracy: 0.9960 - val_char3_accuracy: 0.9984 - val_char4_accuracy: 0.9996 - val_char5_accuracy: 0.9976 - val_char6_accuracy: 0.9992 - val_char7_accuracy: 0.9976 - val_char8_accuracy: 0.9824 - val_char9_accuracy: 0.8636 - val_char10_accuracy: 0.9236
    Epoch 12/150
    47500/47500 [==============================] - 9s 185us/sample - loss: 0.9477 - char0_loss: 0.1430 - char1_loss: 0.0110 - char2_loss: 0.0146 - char3_loss: 0.0181 - char4_loss: 0.0182 - char5_loss: 0.0196 - char6_loss: 0.0219 - char7_loss: 0.0256 - char8_loss: 0.0409 - char9_loss: 0.3920 - char10_loss: 0.2415 - char0_accuracy: 0.9712 - char1_accuracy: 0.9977 - char2_accuracy: 0.9976 - char3_accuracy: 0.9961 - char4_accuracy: 0.9974 - char5_accuracy: 0.9967 - char6_accuracy: 0.9979 - char7_accuracy: 0.9973 - char8_accuracy: 0.9933 - char9_accuracy: 0.8714 - char10_accuracy: 0.9288 - val_loss: 1.0322 - val_char0_loss: 0.0235 - val_char1_loss: 0.0154 - val_char2_loss: 0.0279 - val_char3_loss: 0.0604 - val_char4_loss: 0.0317 - val_char5_loss: 0.0355 - val_char6_loss: 0.0374 - val_char7_loss: 0.0428 - val_char8_loss: 0.0782 - val_char9_loss: 0.4077 - val_char10_loss: 0.2732 - val_char0_accuracy: 0.9952 - val_char1_accuracy: 0.9980 - val_char2_accuracy: 0.9932 - val_char3_accuracy: 0.9804 - val_char4_accuracy: 0.9932 - val_char5_accuracy: 0.9920 - val_char6_accuracy: 0.9952 - val_char7_accuracy: 0.9928 - val_char8_accuracy: 0.9744 - val_char9_accuracy: 0.8728 - val_char10_accuracy: 0.9276
    Epoch 13/150
    47500/47500 [==============================] - 10s 212us/sample - loss: 0.6518 - char0_loss: 0.0033 - char1_loss: 0.0048 - char2_loss: 0.0077 - char3_loss: 0.0076 - char4_loss: 0.0089 - char5_loss: 0.0110 - char6_loss: 0.0130 - char7_loss: 0.0156 - char8_loss: 0.0255 - char9_loss: 0.3212 - char10_loss: 0.2335 - char0_accuracy: 0.9995 - char1_accuracy: 0.9995 - char2_accuracy: 0.9988 - char3_accuracy: 0.9990 - char4_accuracy: 0.9993 - char5_accuracy: 0.9985 - char6_accuracy: 0.9994 - char7_accuracy: 0.9989 - char8_accuracy: 0.9969 - char9_accuracy: 0.8900 - char10_accuracy: 0.9317 - val_loss: 0.7379 - val_char0_loss: 0.0037 - val_char1_loss: 0.0041 - val_char2_loss: 0.0076 - val_char3_loss: 0.0098 - val_char4_loss: 0.0140 - val_char5_loss: 0.0089 - val_char6_loss: 0.0189 - val_char7_loss: 0.0222 - val_char8_loss: 0.0374 - val_char9_loss: 0.3268 - val_char10_loss: 0.2892 - val_char0_accuracy: 0.9996 - val_char1_accuracy: 0.9992 - val_char2_accuracy: 0.9984 - val_char3_accuracy: 0.9980 - val_char4_accuracy: 0.9988 - val_char5_accuracy: 0.9992 - val_char6_accuracy: 0.9972 - val_char7_accuracy: 0.9960 - val_char8_accuracy: 0.9936 - val_char9_accuracy: 0.8908 - val_char10_accuracy: 0.9276
    Epoch 14/150
    47500/47500 [==============================] - 8s 177us/sample - loss: 0.8829 - char0_loss: 0.0097 - char1_loss: 0.0243 - char2_loss: 0.0296 - char3_loss: 0.0274 - char4_loss: 0.0340 - char5_loss: 0.0341 - char6_loss: 0.0486 - char7_loss: 0.0704 - char8_loss: 0.0769 - char9_loss: 0.2914 - char10_loss: 0.2349 - char0_accuracy: 0.9979 - char1_accuracy: 0.9925 - char2_accuracy: 0.9909 - char3_accuracy: 0.9930 - char4_accuracy: 0.9913 - char5_accuracy: 0.9907 - char6_accuracy: 0.9897 - char7_accuracy: 0.9859 - char8_accuracy: 0.9850 - char9_accuracy: 0.9059 - char10_accuracy: 0.9321 - val_loss: 0.6366 - val_char0_loss: 0.0089 - val_char1_loss: 0.0043 - val_char2_loss: 0.0078 - val_char3_loss: 0.0070 - val_char4_loss: 0.0065 - val_char5_loss: 0.0084 - val_char6_loss: 0.0112 - val_char7_loss: 0.0129 - val_char8_loss: 0.0253 - val_char9_loss: 0.2662 - val_char10_loss: 0.2800 - val_char0_accuracy: 0.9980 - val_char1_accuracy: 0.9992 - val_char2_accuracy: 0.9980 - val_char3_accuracy: 0.9988 - val_char4_accuracy: 0.9996 - val_char5_accuracy: 0.9992 - val_char6_accuracy: 0.9988 - val_char7_accuracy: 0.9988 - val_char8_accuracy: 0.9948 - val_char9_accuracy: 0.9024 - val_char10_accuracy: 0.9264
    Epoch 15/150
    47500/47500 [==============================] - 9s 191us/sample - loss: 0.4272 - char0_loss: 0.0032 - char1_loss: 0.0035 - char2_loss: 0.0045 - char3_loss: 0.0050 - char4_loss: 0.0052 - char5_loss: 0.0045 - char6_loss: 0.0062 - char7_loss: 0.0075 - char8_loss: 0.0123 - char9_loss: 0.1765 - char10_loss: 0.1986 - char0_accuracy: 0.9996 - char1_accuracy: 0.9996 - char2_accuracy: 0.9993 - char3_accuracy: 0.9993 - char4_accuracy: 0.9996 - char5_accuracy: 0.9998 - char6_accuracy: 0.9998 - char7_accuracy: 0.9996 - char8_accuracy: 0.9991 - char9_accuracy: 0.9362 - char10_accuracy: 0.9386 - val_loss: 0.5662 - val_char0_loss: 0.0041 - val_char1_loss: 0.0057 - val_char2_loss: 0.0174 - val_char3_loss: 0.0134 - val_char4_loss: 0.0108 - val_char5_loss: 0.0102 - val_char6_loss: 0.0098 - val_char7_loss: 0.0165 - val_char8_loss: 0.0230 - val_char9_loss: 0.1776 - val_char10_loss: 0.2780 - val_char0_accuracy: 0.9996 - val_char1_accuracy: 0.9988 - val_char2_accuracy: 0.9940 - val_char3_accuracy: 0.9960 - val_char4_accuracy: 0.9972 - val_char5_accuracy: 0.9984 - val_char6_accuracy: 0.9988 - val_char7_accuracy: 0.9968 - val_char8_accuracy: 0.9952 - val_char9_accuracy: 0.9376 - val_char10_accuracy: 0.9268
    Epoch 16/150
    47500/47500 [==============================] - 9s 194us/sample - loss: 0.3576 - char0_loss: 0.0040 - char1_loss: 0.0075 - char2_loss: 0.0098 - char3_loss: 0.0037 - char4_loss: 0.0093 - char5_loss: 0.0046 - char6_loss: 0.0069 - char7_loss: 0.0069 - char8_loss: 0.0111 - char9_loss: 0.1029 - char10_loss: 0.1928 - char0_accuracy: 0.9992 - char1_accuracy: 0.9980 - char2_accuracy: 0.9974 - char3_accuracy: 0.9997 - char4_accuracy: 0.9982 - char5_accuracy: 0.9997 - char6_accuracy: 0.9996 - char7_accuracy: 0.9996 - char8_accuracy: 0.9990 - char9_accuracy: 0.9657 - char10_accuracy: 0.9413 - val_loss: 0.6363 - val_char0_loss: 0.0091 - val_char1_loss: 0.0069 - val_char2_loss: 0.0527 - val_char3_loss: 0.0185 - val_char4_loss: 0.0997 - val_char5_loss: 0.0161 - val_char6_loss: 0.0241 - val_char7_loss: 0.0108 - val_char8_loss: 0.0231 - val_char9_loss: 0.1323 - val_char10_loss: 0.2451 - val_char0_accuracy: 0.9976 - val_char1_accuracy: 0.9972 - val_char2_accuracy: 0.9812 - val_char3_accuracy: 0.9940 - val_char4_accuracy: 0.9620 - val_char5_accuracy: 0.9964 - val_char6_accuracy: 0.9932 - val_char7_accuracy: 0.9992 - val_char8_accuracy: 0.9960 - val_char9_accuracy: 0.9500 - val_char10_accuracy: 0.9340
    Epoch 17/150
    47500/47500 [==============================] - 9s 195us/sample - loss: 0.4439 - char0_loss: 0.0263 - char1_loss: 0.0732 - char2_loss: 0.0114 - char3_loss: 0.0214 - char4_loss: 0.0148 - char5_loss: 0.0124 - char6_loss: 0.0123 - char7_loss: 0.0123 - char8_loss: 0.0169 - char9_loss: 0.0700 - char10_loss: 0.1741 - char0_accuracy: 0.9960 - char1_accuracy: 0.9840 - char2_accuracy: 0.9968 - char3_accuracy: 0.9933 - char4_accuracy: 0.9961 - char5_accuracy: 0.9980 - char6_accuracy: 0.9981 - char7_accuracy: 0.9984 - char8_accuracy: 0.9974 - char9_accuracy: 0.9798 - char10_accuracy: 0.9453 - val_loss: 6.7127 - val_char0_loss: 6.1261 - val_char1_loss: 0.0338 - val_char2_loss: 0.0102 - val_char3_loss: 0.0323 - val_char4_loss: 0.0207 - val_char5_loss: 0.0208 - val_char6_loss: 0.0326 - val_char7_loss: 0.0229 - val_char8_loss: 0.0530 - val_char9_loss: 0.1028 - val_char10_loss: 0.2793 - val_char0_accuracy: 0.6056 - val_char1_accuracy: 0.9876 - val_char2_accuracy: 0.9980 - val_char3_accuracy: 0.9908 - val_char4_accuracy: 0.9928 - val_char5_accuracy: 0.9952 - val_char6_accuracy: 0.9904 - val_char7_accuracy: 0.9968 - val_char8_accuracy: 0.9832 - val_char9_accuracy: 0.9652 - val_char10_accuracy: 0.9244
    Epoch 18/150
    47500/47500 [==============================] - 9s 197us/sample - loss: 0.4194 - char0_loss: 0.1017 - char1_loss: 0.0046 - char2_loss: 0.0077 - char3_loss: 0.0149 - char4_loss: 0.0102 - char5_loss: 0.0166 - char6_loss: 0.0123 - char7_loss: 0.0140 - char8_loss: 0.0261 - char9_loss: 0.0487 - char10_loss: 0.1623 - char0_accuracy: 0.9825 - char1_accuracy: 0.9989 - char2_accuracy: 0.9983 - char3_accuracy: 0.9955 - char4_accuracy: 0.9974 - char5_accuracy: 0.9952 - char6_accuracy: 0.9980 - char7_accuracy: 0.9973 - char8_accuracy: 0.9931 - char9_accuracy: 0.9879 - char10_accuracy: 0.9486 - val_loss: 0.3584 - val_char0_loss: 0.0024 - val_char1_loss: 0.0044 - val_char2_loss: 0.0052 - val_char3_loss: 0.0047 - val_char4_loss: 0.0059 - val_char5_loss: 0.0047 - val_char6_loss: 0.0119 - val_char7_loss: 0.0064 - val_char8_loss: 0.0154 - val_char9_loss: 0.0623 - val_char10_loss: 0.2375 - val_char0_accuracy: 0.9996 - val_char1_accuracy: 0.9980 - val_char2_accuracy: 0.9984 - val_char3_accuracy: 0.9984 - val_char4_accuracy: 0.9992 - val_char5_accuracy: 0.9988 - val_char6_accuracy: 0.9972 - val_char7_accuracy: 0.9992 - val_char8_accuracy: 0.9972 - val_char9_accuracy: 0.9792 - val_char10_accuracy: 0.9356
    Epoch 19/150
    47500/47500 [==============================] - 8s 174us/sample - loss: 0.2011 - char0_loss: 0.0013 - char1_loss: 0.0015 - char2_loss: 0.0057 - char3_loss: 0.0013 - char4_loss: 0.0028 - char5_loss: 0.0029 - char6_loss: 0.0042 - char7_loss: 0.0058 - char8_loss: 0.0077 - char9_loss: 0.0229 - char10_loss: 0.1444 - char0_accuracy: 0.9999 - char1_accuracy: 0.9998 - char2_accuracy: 0.9983 - char3_accuracy: 1.0000 - char4_accuracy: 0.9997 - char5_accuracy: 0.9997 - char6_accuracy: 0.9995 - char7_accuracy: 0.9992 - char8_accuracy: 0.9987 - char9_accuracy: 0.9961 - char10_accuracy: 0.9523 - val_loss: 0.3436 - val_char0_loss: 0.0022 - val_char1_loss: 0.0027 - val_char2_loss: 0.0115 - val_char3_loss: 0.0068 - val_char4_loss: 0.0078 - val_char5_loss: 0.0355 - val_char6_loss: 0.0074 - val_char7_loss: 0.0126 - val_char8_loss: 0.0146 - val_char9_loss: 0.0410 - val_char10_loss: 0.2054 - val_char0_accuracy: 0.9996 - val_char1_accuracy: 0.9992 - val_char2_accuracy: 0.9972 - val_char3_accuracy: 0.9988 - val_char4_accuracy: 0.9988 - val_char5_accuracy: 0.9868 - val_char6_accuracy: 0.9976 - val_char7_accuracy: 0.9964 - val_char8_accuracy: 0.9956 - val_char9_accuracy: 0.9896 - val_char10_accuracy: 0.9436
    Epoch 20/150
    47500/47500 [==============================] - 8s 178us/sample - loss: 0.3785 - char0_loss: 0.0052 - char1_loss: 0.0078 - char2_loss: 0.0899 - char3_loss: 0.0162 - char4_loss: 0.0219 - char5_loss: 0.0183 - char6_loss: 0.0196 - char7_loss: 0.0215 - char8_loss: 0.0174 - char9_loss: 0.0275 - char10_loss: 0.1330 - char0_accuracy: 0.9991 - char1_accuracy: 0.9980 - char2_accuracy: 0.9829 - char3_accuracy: 0.9949 - char4_accuracy: 0.9937 - char5_accuracy: 0.9942 - char6_accuracy: 0.9945 - char7_accuracy: 0.9938 - char8_accuracy: 0.9954 - char9_accuracy: 0.9947 - char10_accuracy: 0.9562 - val_loss: 0.3301 - val_char0_loss: 0.0030 - val_char1_loss: 0.0148 - val_char2_loss: 0.0041 - val_char3_loss: 0.0088 - val_char4_loss: 0.0040 - val_char5_loss: 0.0147 - val_char6_loss: 0.0127 - val_char7_loss: 0.0072 - val_char8_loss: 0.0177 - val_char9_loss: 0.0372 - val_char10_loss: 0.2082 - val_char0_accuracy: 0.9992 - val_char1_accuracy: 0.9948 - val_char2_accuracy: 0.9984 - val_char3_accuracy: 0.9972 - val_char4_accuracy: 0.9992 - val_char5_accuracy: 0.9948 - val_char6_accuracy: 0.9968 - val_char7_accuracy: 0.9980 - val_char8_accuracy: 0.9960 - val_char9_accuracy: 0.9896 - val_char10_accuracy: 0.9412
    Epoch 21/150
    47500/47500 [==============================] - 9s 187us/sample - loss: 0.2028 - char0_loss: 0.0032 - char1_loss: 0.0039 - char2_loss: 0.0019 - char3_loss: 0.0266 - char4_loss: 0.0044 - char5_loss: 0.0071 - char6_loss: 0.0081 - char7_loss: 0.0064 - char8_loss: 0.0089 - char9_loss: 0.0188 - char10_loss: 0.1132 - char0_accuracy: 0.9995 - char1_accuracy: 0.9992 - char2_accuracy: 0.9997 - char3_accuracy: 0.9927 - char4_accuracy: 0.9992 - char5_accuracy: 0.9981 - char6_accuracy: 0.9980 - char7_accuracy: 0.9988 - char8_accuracy: 0.9983 - char9_accuracy: 0.9966 - char10_accuracy: 0.9626 - val_loss: 0.3425 - val_char0_loss: 0.0041 - val_char1_loss: 0.0097 - val_char2_loss: 0.0031 - val_char3_loss: 0.0133 - val_char4_loss: 0.0181 - val_char5_loss: 0.0328 - val_char6_loss: 0.0049 - val_char7_loss: 0.0102 - val_char8_loss: 0.0280 - val_char9_loss: 0.0452 - val_char10_loss: 0.1752 - val_char0_accuracy: 0.9988 - val_char1_accuracy: 0.9976 - val_char2_accuracy: 0.9996 - val_char3_accuracy: 0.9948 - val_char4_accuracy: 0.9936 - val_char5_accuracy: 0.9880 - val_char6_accuracy: 0.9996 - val_char7_accuracy: 0.9988 - val_char8_accuracy: 0.9920 - val_char9_accuracy: 0.9832 - val_char10_accuracy: 0.9456
    Epoch 22/150
    47500/47500 [==============================] - 9s 190us/sample - loss: 0.2498 - char0_loss: 0.0055 - char1_loss: 0.0300 - char2_loss: 0.0045 - char3_loss: 0.0050 - char4_loss: 0.0144 - char5_loss: 0.0102 - char6_loss: 0.0082 - char7_loss: 0.0183 - char8_loss: 0.0207 - char9_loss: 0.0261 - char10_loss: 0.1073 - char0_accuracy: 0.9988 - char1_accuracy: 0.9899 - char2_accuracy: 0.9987 - char3_accuracy: 0.9987 - char4_accuracy: 0.9956 - char5_accuracy: 0.9972 - char6_accuracy: 0.9981 - char7_accuracy: 0.9943 - char8_accuracy: 0.9945 - char9_accuracy: 0.9943 - char10_accuracy: 0.9650 - val_loss: 0.9724 - val_char0_loss: 0.0273 - val_char1_loss: 0.0358 - val_char2_loss: 0.0246 - val_char3_loss: 0.0165 - val_char4_loss: 0.0363 - val_char5_loss: 0.0845 - val_char6_loss: 0.0673 - val_char7_loss: 0.0859 - val_char8_loss: 0.2014 - val_char9_loss: 0.0966 - val_char10_loss: 0.2978 - val_char0_accuracy: 0.9916 - val_char1_accuracy: 0.9868 - val_char2_accuracy: 0.9920 - val_char3_accuracy: 0.9956 - val_char4_accuracy: 0.9880 - val_char5_accuracy: 0.9712 - val_char6_accuracy: 0.9776 - val_char7_accuracy: 0.9668 - val_char8_accuracy: 0.9452 - val_char9_accuracy: 0.9688 - val_char10_accuracy: 0.9260
    Epoch 23/150
    47500/47500 [==============================] - 10s 200us/sample - loss: 0.2800 - char0_loss: 0.0108 - char1_loss: 0.0062 - char2_loss: 0.0271 - char3_loss: 0.0105 - char4_loss: 0.0196 - char5_loss: 0.0254 - char6_loss: 0.0193 - char7_loss: 0.0188 - char8_loss: 0.0258 - char9_loss: 0.0210 - char10_loss: 0.0960 - char0_accuracy: 0.9971 - char1_accuracy: 0.9983 - char2_accuracy: 0.9913 - char3_accuracy: 0.9967 - char4_accuracy: 0.9937 - char5_accuracy: 0.9919 - char6_accuracy: 0.9935 - char7_accuracy: 0.9938 - char8_accuracy: 0.9916 - char9_accuracy: 0.9950 - char10_accuracy: 0.9673 - val_loss: 0.3866 - val_char0_loss: 0.0344 - val_char1_loss: 0.0032 - val_char2_loss: 0.0137 - val_char3_loss: 0.0074 - val_char4_loss: 0.0178 - val_char5_loss: 0.0511 - val_char6_loss: 0.0325 - val_char7_loss: 0.0099 - val_char8_loss: 0.0114 - val_char9_loss: 0.0385 - val_char10_loss: 0.1692 - val_char0_accuracy: 0.9864 - val_char1_accuracy: 0.9996 - val_char2_accuracy: 0.9940 - val_char3_accuracy: 0.9968 - val_char4_accuracy: 0.9944 - val_char5_accuracy: 0.9800 - val_char6_accuracy: 0.9880 - val_char7_accuracy: 0.9972 - val_char8_accuracy: 0.9964 - val_char9_accuracy: 0.9860 - val_char10_accuracy: 0.9480



![png](/assets/qr-nn/output_66_1.png)



![png](/assets/qr-nn/output_66_2.png)



![png](/assets/qr-nn/output_66_3.png)



![png](/assets/qr-nn/output_66_4.png)


Since more layers did make things worse for other positions, try to split layers per position.


{% highlight python %}
#                                   |-> HIDDEN LAYER -> char 0 output
# INPUT -> FLATTEN -> HIDDEN LAYER -|-> ...
#                                   |-> HIDDEN LAYER -> char 10 output
#
#
def define_split_multi_output_model():
    input_layer = keras.layers.Input(shape=(21,21,), dtype='float', name='input_qr')
    flatten = keras.layers.Flatten(input_shape=(21, 21), name='flatten')(input_layer)    
    hidden_chars1 = keras.layers.Dense(21*21, activation='relu', name='hidden_chars1')(flatten)

    outputs = []
    for i in range(11):
        hidden_chars2 = keras.layers.Dense(21*21, activation='relu')(hidden_chars1)
        char_output = keras.layers.Dense(len(ALL_CHAR_CLASSES), name='char' + str(i))(hidden_chars2)
        outputs.append(char_output)

    multi_output_model = keras.Model(inputs=[input_layer], outputs=outputs)

    multi_output_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return multi_output_model

mo_model = define_split_multi_output_model()
mo_model.summary()
mo_hist = mo_model.fit(
        training_data, [
            np.asarray(training_labels_char0),
            np.asarray(training_labels_char1),
            np.asarray(training_labels_char2),
            np.asarray(training_labels_char3),
            np.asarray(training_labels_char4),
            np.asarray(training_labels_char5),
            np.asarray(training_labels_char6),
            np.asarray(training_labels_char7),
            np.asarray(training_labels_char8),
            np.asarray(training_labels_char9),
            np.asarray(training_labels_char10)
        ],
        epochs=150, batch_size=128,
        validation_split=0.05,
        callbacks=[EarlyStopping(monitor='val_loss' , min_delta=0.0001, patience=3)]
    )

plot_all_accuracy_and_loss(mo_hist)
plot_loss_vs_validation_loss_diff(mo_hist)
plot_loss_vs_validation_loss(mo_hist)
{% endhighlight %}

    Model: "model_2"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_qr (InputLayer)           [(None, 21, 21)]     0                                            
    __________________________________________________________________________________________________
    flatten (Flatten)               (None, 441)          0           input_qr[0][0]                   
    __________________________________________________________________________________________________
    hidden_chars1 (Dense)           (None, 441)          194922      flatten[0][0]                    
    __________________________________________________________________________________________________
    dense_46 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_47 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_48 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_49 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_50 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_51 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_52 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_53 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_54 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_55 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_56 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    char0 (Dense)                   (None, 28)           12376       dense_46[0][0]                   
    __________________________________________________________________________________________________
    char1 (Dense)                   (None, 28)           12376       dense_47[0][0]                   
    __________________________________________________________________________________________________
    char2 (Dense)                   (None, 28)           12376       dense_48[0][0]                   
    __________________________________________________________________________________________________
    char3 (Dense)                   (None, 28)           12376       dense_49[0][0]                   
    __________________________________________________________________________________________________
    char4 (Dense)                   (None, 28)           12376       dense_50[0][0]                   
    __________________________________________________________________________________________________
    char5 (Dense)                   (None, 28)           12376       dense_51[0][0]                   
    __________________________________________________________________________________________________
    char6 (Dense)                   (None, 28)           12376       dense_52[0][0]                   
    __________________________________________________________________________________________________
    char7 (Dense)                   (None, 28)           12376       dense_53[0][0]                   
    __________________________________________________________________________________________________
    char8 (Dense)                   (None, 28)           12376       dense_54[0][0]                   
    __________________________________________________________________________________________________
    char9 (Dense)                   (None, 28)           12376       dense_55[0][0]                   
    __________________________________________________________________________________________________
    char10 (Dense)                  (None, 28)           12376       dense_56[0][0]                   
    ==================================================================================================
    Total params: 2,475,200
    Trainable params: 2,475,200
    Non-trainable params: 0
    __________________________________________________________________________________________________
    Train on 47500 samples, validate on 2500 samples
    Epoch 1/150
    47500/47500 [==============================] - 21s 446us/sample - loss: 18.7172 - char0_loss: 2.5787 - char1_loss: 2.4106 - char2_loss: 2.5598 - char3_loss: 2.2529 - char4_loss: 2.1216 - char5_loss: 1.8427 - char6_loss: 1.5831 - char7_loss: 1.2891 - char8_loss: 1.0000 - char9_loss: 0.6974 - char10_loss: 0.3705 - char0_accuracy: 0.2017 - char1_accuracy: 0.2985 - char2_accuracy: 0.2532 - char3_accuracy: 0.3384 - char4_accuracy: 0.4008 - char5_accuracy: 0.4844 - char6_accuracy: 0.5609 - char7_accuracy: 0.6475 - char8_accuracy: 0.7342 - char9_accuracy: 0.8197 - char10_accuracy: 0.9090 - val_loss: 13.6801 - val_char0_loss: 1.5834 - val_char1_loss: 1.2245 - val_char2_loss: 1.7631 - val_char3_loss: 1.5974 - val_char4_loss: 1.7671 - val_char5_loss: 1.5444 - val_char6_loss: 1.3968 - val_char7_loss: 1.1112 - val_char8_loss: 0.8217 - val_char9_loss: 0.5638 - val_char10_loss: 0.2836 - val_char0_accuracy: 0.4088 - val_char1_accuracy: 0.6176 - val_char2_accuracy: 0.4412 - val_char3_accuracy: 0.4344 - val_char4_accuracy: 0.4480 - val_char5_accuracy: 0.5336 - val_char6_accuracy: 0.5932 - val_char7_accuracy: 0.6796 - val_char8_accuracy: 0.7632 - val_char9_accuracy: 0.8412 - val_char10_accuracy: 0.9196
    Epoch 2/150
    47500/47500 [==============================] - 16s 338us/sample - loss: 7.6295 - char0_loss: 0.5371 - char1_loss: 0.3112 - char2_loss: 0.5593 - char3_loss: 0.7068 - char4_loss: 0.9731 - char5_loss: 0.8918 - char6_loss: 1.0516 - char7_loss: 0.9608 - char8_loss: 0.7353 - char9_loss: 0.5942 - char10_loss: 0.3029 - char0_accuracy: 0.8431 - char1_accuracy: 0.9271 - char2_accuracy: 0.8525 - char3_accuracy: 0.7477 - char4_accuracy: 0.7017 - char5_accuracy: 0.7124 - char6_accuracy: 0.6897 - char7_accuracy: 0.6989 - char8_accuracy: 0.7812 - char9_accuracy: 0.8296 - char10_accuracy: 0.9142 - val_loss: 3.2147 - val_char0_loss: 0.0428 - val_char1_loss: 0.0301 - val_char2_loss: 0.0480 - val_char3_loss: 0.2409 - val_char4_loss: 0.1991 - val_char5_loss: 0.2513 - val_char6_loss: 0.4251 - val_char7_loss: 0.6930 - val_char8_loss: 0.5092 - val_char9_loss: 0.5065 - val_char10_loss: 0.2634 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 0.9156 - val_char4_accuracy: 0.9780 - val_char5_accuracy: 0.9360 - val_char6_accuracy: 0.9180 - val_char7_accuracy: 0.7660 - val_char8_accuracy: 0.8276 - val_char9_accuracy: 0.8456 - val_char10_accuracy: 0.9252
    Epoch 3/150
    47500/47500 [==============================] - 16s 344us/sample - loss: 1.9212 - char0_loss: 0.0186 - char1_loss: 0.0138 - char2_loss: 0.0200 - char3_loss: 0.0803 - char4_loss: 0.0534 - char5_loss: 0.0862 - char6_loss: 0.1329 - char7_loss: 0.3603 - char8_loss: 0.3725 - char9_loss: 0.5042 - char10_loss: 0.2778 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 0.9882 - char4_accuracy: 0.9986 - char5_accuracy: 0.9904 - char6_accuracy: 0.9916 - char7_accuracy: 0.9058 - char8_accuracy: 0.8744 - char9_accuracy: 0.8400 - char10_accuracy: 0.9173 - val_loss: 1.1202 - val_char0_loss: 0.0093 - val_char1_loss: 0.0075 - val_char2_loss: 0.0094 - val_char3_loss: 0.0204 - val_char4_loss: 0.0166 - val_char5_loss: 0.0257 - val_char6_loss: 0.0308 - val_char7_loss: 0.0906 - val_char8_loss: 0.2234 - val_char9_loss: 0.4407 - val_char10_loss: 0.2449 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 0.9952 - val_char8_accuracy: 0.9200 - val_char9_accuracy: 0.8548 - val_char10_accuracy: 0.9256
    Epoch 4/150
    47500/47500 [==============================] - 16s 338us/sample - loss: 0.8811 - char0_loss: 0.0059 - char1_loss: 0.0047 - char2_loss: 0.0061 - char3_loss: 0.0104 - char4_loss: 0.0114 - char5_loss: 0.0134 - char6_loss: 0.0162 - char7_loss: 0.0400 - char8_loss: 0.1288 - char9_loss: 0.3929 - char10_loss: 0.2498 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 0.9997 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 0.9992 - char8_accuracy: 0.9708 - char9_accuracy: 0.8686 - char10_accuracy: 0.9205 - val_loss: 0.6418 - val_char0_loss: 0.0040 - val_char1_loss: 0.0036 - val_char2_loss: 0.0044 - val_char3_loss: 0.0065 - val_char4_loss: 0.0062 - val_char5_loss: 0.0085 - val_char6_loss: 0.0093 - val_char7_loss: 0.0169 - val_char8_loss: 0.0564 - val_char9_loss: 0.3079 - val_char10_loss: 0.2209 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 0.9948 - val_char9_accuracy: 0.8968 - val_char10_accuracy: 0.9292
    Epoch 5/150
    47500/47500 [==============================] - 17s 354us/sample - loss: 0.7962 - char0_loss: 0.0081 - char1_loss: 0.0071 - char2_loss: 0.0315 - char3_loss: 0.0668 - char4_loss: 0.0555 - char5_loss: 0.0384 - char6_loss: 0.0360 - char7_loss: 0.0322 - char8_loss: 0.0549 - char9_loss: 0.2386 - char10_loss: 0.2265 - char0_accuracy: 0.9997 - char1_accuracy: 0.9997 - char2_accuracy: 0.9947 - char3_accuracy: 0.9907 - char4_accuracy: 0.9916 - char5_accuracy: 0.9948 - char6_accuracy: 0.9954 - char7_accuracy: 0.9963 - char8_accuracy: 0.9954 - char9_accuracy: 0.9352 - char10_accuracy: 0.9253 - val_loss: 0.3923 - val_char0_loss: 0.0022 - val_char1_loss: 0.0018 - val_char2_loss: 0.0026 - val_char3_loss: 0.0049 - val_char4_loss: 0.0041 - val_char5_loss: 0.0049 - val_char6_loss: 0.0055 - val_char7_loss: 0.0076 - val_char8_loss: 0.0199 - val_char9_loss: 0.1354 - val_char10_loss: 0.2043 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 0.9996 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 0.9996 - val_char9_accuracy: 0.9636 - val_char10_accuracy: 0.9280
    Epoch 6/150
    47500/47500 [==============================] - 16s 341us/sample - loss: 0.3033 - char0_loss: 0.0015 - char1_loss: 0.0013 - char2_loss: 0.0018 - char3_loss: 0.0032 - char4_loss: 0.0027 - char5_loss: 0.0033 - char6_loss: 0.0037 - char7_loss: 0.0049 - char8_loss: 0.0112 - char9_loss: 0.0738 - char10_loss: 0.1957 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 0.9999 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 0.9912 - char10_accuracy: 0.9304 - val_loss: 0.2419 - val_char0_loss: 0.0013 - val_char1_loss: 0.0011 - val_char2_loss: 0.0014 - val_char3_loss: 0.0027 - val_char4_loss: 0.0023 - val_char5_loss: 0.0027 - val_char6_loss: 0.0030 - val_char7_loss: 0.0039 - val_char8_loss: 0.0093 - val_char9_loss: 0.0390 - val_char10_loss: 0.1760 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 0.9996 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 0.9956 - val_char10_accuracy: 0.9332
    Epoch 7/150
    47500/47500 [==============================] - 16s 342us/sample - loss: 0.2141 - char0_loss: 0.0010 - char1_loss: 8.4777e-04 - char2_loss: 0.0012 - char3_loss: 0.0019 - char4_loss: 0.0017 - char5_loss: 0.0020 - char6_loss: 0.0023 - char7_loss: 0.0029 - char8_loss: 0.0056 - char9_loss: 0.0222 - char10_loss: 0.1727 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 0.9994 - char10_accuracy: 0.9349 - val_loss: 0.1883 - val_char0_loss: 9.3200e-04 - val_char1_loss: 7.9361e-04 - val_char2_loss: 9.9153e-04 - val_char3_loss: 0.0016 - val_char4_loss: 0.0016 - val_char5_loss: 0.0018 - val_char6_loss: 0.0020 - val_char7_loss: 0.0025 - val_char8_loss: 0.0050 - val_char9_loss: 0.0145 - val_char10_loss: 0.1575 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9384
    Epoch 8/150
    47500/47500 [==============================] - 16s 347us/sample - loss: 0.1769 - char0_loss: 7.3491e-04 - char1_loss: 6.2026e-04 - char2_loss: 8.3853e-04 - char3_loss: 0.0013 - char4_loss: 0.0012 - char5_loss: 0.0014 - char6_loss: 0.0016 - char7_loss: 0.0019 - char8_loss: 0.0033 - char9_loss: 0.0092 - char10_loss: 0.1557 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 0.9999 - char10_accuracy: 0.9398 - val_loss: 0.1742 - val_char0_loss: 6.8422e-04 - val_char1_loss: 6.2762e-04 - val_char2_loss: 7.5111e-04 - val_char3_loss: 0.0013 - val_char4_loss: 0.0012 - val_char5_loss: 0.0013 - val_char6_loss: 0.0014 - val_char7_loss: 0.0018 - val_char8_loss: 0.0031 - val_char9_loss: 0.0072 - val_char10_loss: 0.1556 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9392
    Epoch 9/150
    47500/47500 [==============================] - 20s 426us/sample - loss: 0.1527 - char0_loss: 5.6344e-04 - char1_loss: 4.7806e-04 - char2_loss: 6.4966e-04 - char3_loss: 9.3767e-04 - char4_loss: 9.1309e-04 - char5_loss: 0.0010 - char6_loss: 0.0012 - char7_loss: 0.0014 - char8_loss: 0.0022 - char9_loss: 0.0046 - char10_loss: 0.1386 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9467 - val_loss: 0.1424 - val_char0_loss: 5.3700e-04 - val_char1_loss: 4.6939e-04 - val_char2_loss: 5.9320e-04 - val_char3_loss: 9.8667e-04 - val_char4_loss: 9.5481e-04 - val_char5_loss: 0.0010 - val_char6_loss: 0.0011 - val_char7_loss: 0.0013 - val_char8_loss: 0.0022 - val_char9_loss: 0.0042 - val_char10_loss: 0.1296 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9488
    Epoch 10/150
    47500/47500 [==============================] - 16s 336us/sample - loss: 0.1274 - char0_loss: 4.5356e-04 - char1_loss: 3.8702e-04 - char2_loss: 5.3518e-04 - char3_loss: 7.4128e-04 - char4_loss: 7.3653e-04 - char5_loss: 7.9749e-04 - char6_loss: 8.9560e-04 - char7_loss: 0.0010 - char8_loss: 0.0016 - char9_loss: 0.0060 - char10_loss: 0.1150 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 0.9994 - char10_accuracy: 0.9581 - val_loss: 0.1168 - val_char0_loss: 4.6606e-04 - val_char1_loss: 4.2814e-04 - val_char2_loss: 5.4990e-04 - val_char3_loss: 8.2829e-04 - val_char4_loss: 8.5065e-04 - val_char5_loss: 8.3363e-04 - val_char6_loss: 9.7064e-04 - val_char7_loss: 0.0011 - val_char8_loss: 0.0018 - val_char9_loss: 0.0038 - val_char10_loss: 0.1055 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9576
    Epoch 11/150
    47500/47500 [==============================] - 17s 356us/sample - loss: 0.1035 - char0_loss: 4.7136e-04 - char1_loss: 3.9967e-04 - char2_loss: 5.4778e-04 - char3_loss: 6.9927e-04 - char4_loss: 7.4303e-04 - char5_loss: 7.5481e-04 - char6_loss: 8.7267e-04 - char7_loss: 9.9065e-04 - char8_loss: 0.0019 - char9_loss: 0.0051 - char10_loss: 0.0908 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 0.9999 - char9_accuracy: 0.9994 - char10_accuracy: 0.9716 - val_loss: 0.0729 - val_char0_loss: 3.5414e-04 - val_char1_loss: 3.1033e-04 - val_char2_loss: 3.8876e-04 - val_char3_loss: 5.4687e-04 - val_char4_loss: 7.5760e-04 - val_char5_loss: 5.9125e-04 - val_char6_loss: 6.7417e-04 - val_char7_loss: 7.7530e-04 - val_char8_loss: 0.0011 - val_char9_loss: 0.0016 - val_char10_loss: 0.0661 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9768
    Epoch 12/150
    47500/47500 [==============================] - 18s 378us/sample - loss: 0.0477 - char0_loss: 2.6621e-04 - char1_loss: 2.3091e-04 - char2_loss: 3.2073e-04 - char3_loss: 4.2709e-04 - char4_loss: 4.3851e-04 - char5_loss: 4.4585e-04 - char6_loss: 5.1446e-04 - char7_loss: 5.6909e-04 - char8_loss: 7.6681e-04 - char9_loss: 0.0011 - char10_loss: 0.0426 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9887 - val_loss: 0.0408 - val_char0_loss: 2.4585e-04 - val_char1_loss: 2.1882e-04 - val_char2_loss: 2.8306e-04 - val_char3_loss: 4.0665e-04 - val_char4_loss: 4.2644e-04 - val_char5_loss: 4.1529e-04 - val_char6_loss: 4.8411e-04 - val_char7_loss: 5.5130e-04 - val_char8_loss: 8.3506e-04 - val_char9_loss: 0.0013 - val_char10_loss: 0.0361 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9932
    Epoch 13/150
    47500/47500 [==============================] - 16s 344us/sample - loss: 0.0236 - char0_loss: 1.9541e-04 - char1_loss: 1.7082e-04 - char2_loss: 2.4141e-04 - char3_loss: 3.1694e-04 - char4_loss: 3.2470e-04 - char5_loss: 3.2870e-04 - char6_loss: 3.6935e-04 - char7_loss: 4.1345e-04 - char8_loss: 5.4341e-04 - char9_loss: 7.8156e-04 - char10_loss: 0.0199 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9972 - val_loss: 0.0196 - val_char0_loss: 1.8799e-04 - val_char1_loss: 1.6622e-04 - val_char2_loss: 2.2133e-04 - val_char3_loss: 3.3089e-04 - val_char4_loss: 3.2686e-04 - val_char5_loss: 3.0892e-04 - val_char6_loss: 3.5098e-04 - val_char7_loss: 3.9577e-04 - val_char8_loss: 5.7255e-04 - val_char9_loss: 8.4229e-04 - val_char10_loss: 0.0162 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9984
    Epoch 14/150
    47500/47500 [==============================] - 17s 348us/sample - loss: 0.0131 - char0_loss: 1.5071e-04 - char1_loss: 1.3083e-04 - char2_loss: 1.8572e-04 - char3_loss: 2.4627e-04 - char4_loss: 2.5503e-04 - char5_loss: 2.5052e-04 - char6_loss: 2.7948e-04 - char7_loss: 3.1391e-04 - char8_loss: 4.0961e-04 - char9_loss: 5.8185e-04 - char10_loss: 0.0103 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9991 - val_loss: 0.0112 - val_char0_loss: 1.4276e-04 - val_char1_loss: 1.2635e-04 - val_char2_loss: 1.6882e-04 - val_char3_loss: 2.9736e-04 - val_char4_loss: 2.8737e-04 - val_char5_loss: 2.4213e-04 - val_char6_loss: 2.7526e-04 - val_char7_loss: 3.0665e-04 - val_char8_loss: 4.4047e-04 - val_char9_loss: 6.6537e-04 - val_char10_loss: 0.0082 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9996
    Epoch 15/150
    47500/47500 [==============================] - 17s 354us/sample - loss: 0.0074 - char0_loss: 1.1541e-04 - char1_loss: 9.9703e-05 - char2_loss: 1.4329e-04 - char3_loss: 1.8845e-04 - char4_loss: 1.9752e-04 - char5_loss: 1.8981e-04 - char6_loss: 2.1353e-04 - char7_loss: 2.3932e-04 - char8_loss: 3.0677e-04 - char9_loss: 4.2919e-04 - char10_loss: 0.0053 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9997 - val_loss: 0.0069 - val_char0_loss: 1.1123e-04 - val_char1_loss: 1.0186e-04 - val_char2_loss: 1.2983e-04 - val_char3_loss: 2.4650e-04 - val_char4_loss: 2.0981e-04 - val_char5_loss: 1.8130e-04 - val_char6_loss: 2.1789e-04 - val_char7_loss: 2.3680e-04 - val_char8_loss: 3.2777e-04 - val_char9_loss: 4.8480e-04 - val_char10_loss: 0.0046 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 16/150
    47500/47500 [==============================] - 17s 354us/sample - loss: 0.0047 - char0_loss: 8.8984e-05 - char1_loss: 7.7619e-05 - char2_loss: 1.1134e-04 - char3_loss: 1.4834e-04 - char4_loss: 1.5847e-04 - char5_loss: 1.4661e-04 - char6_loss: 1.6394e-04 - char7_loss: 1.8421e-04 - char8_loss: 2.4041e-04 - char9_loss: 3.2488e-04 - char10_loss: 0.0031 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9999 - val_loss: 0.0049 - val_char0_loss: 8.9834e-05 - val_char1_loss: 7.8555e-05 - val_char2_loss: 1.0211e-04 - val_char3_loss: 1.5857e-04 - val_char4_loss: 2.1232e-04 - val_char5_loss: 1.3877e-04 - val_char6_loss: 1.6666e-04 - val_char7_loss: 1.8370e-04 - val_char8_loss: 2.5697e-04 - val_char9_loss: 3.8577e-04 - val_char10_loss: 0.0031 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 17/150
    47500/47500 [==============================] - 17s 348us/sample - loss: 0.0032 - char0_loss: 7.0198e-05 - char1_loss: 6.1029e-05 - char2_loss: 8.7959e-05 - char3_loss: 1.1827e-04 - char4_loss: 1.2401e-04 - char5_loss: 1.1454e-04 - char6_loss: 1.2900e-04 - char7_loss: 1.4381e-04 - char8_loss: 1.8175e-04 - char9_loss: 2.5495e-04 - char10_loss: 0.0020 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 0.0035 - val_char0_loss: 7.0332e-05 - val_char1_loss: 6.1706e-05 - val_char2_loss: 8.1612e-05 - val_char3_loss: 1.4671e-04 - val_char4_loss: 1.2898e-04 - val_char5_loss: 1.1582e-04 - val_char6_loss: 1.2395e-04 - val_char7_loss: 1.4767e-04 - val_char8_loss: 2.0580e-04 - val_char9_loss: 2.9426e-04 - val_char10_loss: 0.0021 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 18/150
    47500/47500 [==============================] - 17s 363us/sample - loss: 0.0023 - char0_loss: 5.6012e-05 - char1_loss: 4.8510e-05 - char2_loss: 6.9911e-05 - char3_loss: 9.5074e-05 - char4_loss: 1.0122e-04 - char5_loss: 9.1355e-05 - char6_loss: 1.0068e-04 - char7_loss: 1.1332e-04 - char8_loss: 1.4348e-04 - char9_loss: 1.9878e-04 - char10_loss: 0.0013 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 0.0027 - val_char0_loss: 5.5932e-05 - val_char1_loss: 5.0512e-05 - val_char2_loss: 6.4706e-05 - val_char3_loss: 1.0756e-04 - val_char4_loss: 1.1151e-04 - val_char5_loss: 8.8016e-05 - val_char6_loss: 9.7176e-05 - val_char7_loss: 1.2004e-04 - val_char8_loss: 1.6219e-04 - val_char9_loss: 2.3077e-04 - val_char10_loss: 0.0016 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 19/150
    47500/47500 [==============================] - 18s 371us/sample - loss: 0.0016 - char0_loss: 4.4613e-05 - char1_loss: 3.8676e-05 - char2_loss: 5.5813e-05 - char3_loss: 7.6161e-05 - char4_loss: 8.0315e-05 - char5_loss: 7.2186e-05 - char6_loss: 7.9344e-05 - char7_loss: 8.9421e-05 - char8_loss: 1.1144e-04 - char9_loss: 1.5590e-04 - char10_loss: 8.2670e-04 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 0.0019 - val_char0_loss: 4.3854e-05 - val_char1_loss: 4.0020e-05 - val_char2_loss: 5.2230e-05 - val_char3_loss: 1.1438e-04 - val_char4_loss: 8.3604e-05 - val_char5_loss: 6.9683e-05 - val_char6_loss: 8.1313e-05 - val_char7_loss: 9.3317e-05 - val_char8_loss: 1.2436e-04 - val_char9_loss: 1.9478e-04 - val_char10_loss: 0.0010 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 20/150
    47500/47500 [==============================] - 17s 360us/sample - loss: 0.0012 - char0_loss: 3.5901e-05 - char1_loss: 3.0924e-05 - char2_loss: 4.4846e-05 - char3_loss: 6.1302e-05 - char4_loss: 6.3739e-05 - char5_loss: 5.7080e-05 - char6_loss: 6.3319e-05 - char7_loss: 7.1508e-05 - char8_loss: 8.8397e-05 - char9_loss: 1.2355e-04 - char10_loss: 5.9545e-04 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 0.0015 - val_char0_loss: 3.5288e-05 - val_char1_loss: 3.1570e-05 - val_char2_loss: 4.2710e-05 - val_char3_loss: 7.1566e-05 - val_char4_loss: 6.8222e-05 - val_char5_loss: 5.5490e-05 - val_char6_loss: 6.6843e-05 - val_char7_loss: 7.4805e-05 - val_char8_loss: 9.9527e-05 - val_char9_loss: 1.5450e-04 - val_char10_loss: 8.5448e-04 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 21/150
    47500/47500 [==============================] - 17s 352us/sample - loss: 9.6148e-04 - char0_loss: 2.8834e-05 - char1_loss: 2.4960e-05 - char2_loss: 3.6210e-05 - char3_loss: 4.8851e-05 - char4_loss: 5.1803e-05 - char5_loss: 4.5805e-05 - char6_loss: 5.0425e-05 - char7_loss: 5.7197e-05 - char8_loss: 6.9728e-05 - char9_loss: 9.8410e-05 - char10_loss: 4.4908e-04 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 0.0012 - val_char0_loss: 2.9303e-05 - val_char1_loss: 2.6215e-05 - val_char2_loss: 3.3958e-05 - val_char3_loss: 5.5398e-05 - val_char4_loss: 5.5084e-05 - val_char5_loss: 4.4239e-05 - val_char6_loss: 5.2354e-05 - val_char7_loss: 5.9469e-05 - val_char8_loss: 8.0037e-05 - val_char9_loss: 1.2288e-04 - val_char10_loss: 6.2169e-04 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 22/150
    47500/47500 [==============================] - 17s 366us/sample - loss: 7.4219e-04 - char0_loss: 2.3201e-05 - char1_loss: 2.0086e-05 - char2_loss: 2.9165e-05 - char3_loss: 4.0049e-05 - char4_loss: 4.1730e-05 - char5_loss: 3.6778e-05 - char6_loss: 4.0427e-05 - char7_loss: 4.5761e-05 - char8_loss: 5.6042e-05 - char9_loss: 7.8524e-05 - char10_loss: 3.2945e-04 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 9.8906e-04 - val_char0_loss: 2.3558e-05 - val_char1_loss: 2.1095e-05 - val_char2_loss: 2.7597e-05 - val_char3_loss: 5.0635e-05 - val_char4_loss: 4.7326e-05 - val_char5_loss: 3.5696e-05 - val_char6_loss: 4.3730e-05 - val_char7_loss: 4.9218e-05 - val_char8_loss: 6.4364e-05 - val_char9_loss: 9.6316e-05 - val_char10_loss: 5.3693e-04 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 23/150
    47500/47500 [==============================] - 17s 362us/sample - loss: 5.8458e-04 - char0_loss: 1.8773e-05 - char1_loss: 1.6314e-05 - char2_loss: 2.3652e-05 - char3_loss: 3.1921e-05 - char4_loss: 3.4237e-05 - char5_loss: 2.9828e-05 - char6_loss: 3.3071e-05 - char7_loss: 3.6789e-05 - char8_loss: 4.4615e-05 - char9_loss: 6.2640e-05 - char10_loss: 2.5293e-04 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 7.4674e-04 - val_char0_loss: 1.8710e-05 - val_char1_loss: 1.6834e-05 - val_char2_loss: 2.2902e-05 - val_char3_loss: 4.5240e-05 - val_char4_loss: 3.7902e-05 - val_char5_loss: 2.8915e-05 - val_char6_loss: 3.4477e-05 - val_char7_loss: 3.7962e-05 - val_char8_loss: 5.2709e-05 - val_char9_loss: 8.1640e-05 - val_char10_loss: 3.7594e-04 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 24/150
    47500/47500 [==============================] - 17s 356us/sample - loss: 4.5874e-04 - char0_loss: 1.5241e-05 - char1_loss: 1.3173e-05 - char2_loss: 1.9181e-05 - char3_loss: 2.5557e-05 - char4_loss: 2.8403e-05 - char5_loss: 2.4046e-05 - char6_loss: 2.6338e-05 - char7_loss: 2.9519e-05 - char8_loss: 3.5811e-05 - char9_loss: 5.0097e-05 - char10_loss: 1.9082e-04 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 6.2583e-04 - val_char0_loss: 1.5462e-05 - val_char1_loss: 1.3705e-05 - val_char2_loss: 1.8182e-05 - val_char3_loss: 3.4454e-05 - val_char4_loss: 3.2969e-05 - val_char5_loss: 2.3445e-05 - val_char6_loss: 3.0128e-05 - val_char7_loss: 3.0846e-05 - val_char8_loss: 4.3349e-05 - val_char9_loss: 6.2985e-05 - val_char10_loss: 3.2355e-04 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 25/150
    47500/47500 [==============================] - 17s 352us/sample - loss: 3.6459e-04 - char0_loss: 1.2564e-05 - char1_loss: 1.0705e-05 - char2_loss: 1.5486e-05 - char3_loss: 2.0728e-05 - char4_loss: 2.2965e-05 - char5_loss: 1.9451e-05 - char6_loss: 2.1219e-05 - char7_loss: 2.3759e-05 - char8_loss: 2.8618e-05 - char9_loss: 4.0074e-05 - char10_loss: 1.4863e-04 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 4.7420e-04 - val_char0_loss: 1.2549e-05 - val_char1_loss: 1.1144e-05 - val_char2_loss: 1.4639e-05 - val_char3_loss: 2.2032e-05 - val_char4_loss: 2.5994e-05 - val_char5_loss: 1.9020e-05 - val_char6_loss: 2.4395e-05 - val_char7_loss: 2.4966e-05 - val_char8_loss: 3.3299e-05 - val_char9_loss: 5.1081e-05 - val_char10_loss: 2.3855e-04 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 26/150
    47500/47500 [==============================] - 18s 375us/sample - loss: 1.2952 - char0_loss: 0.0219 - char1_loss: 0.1961 - char2_loss: 0.0529 - char3_loss: 0.0701 - char4_loss: 0.2452 - char5_loss: 0.1049 - char6_loss: 0.1272 - char7_loss: 0.1688 - char8_loss: 0.1339 - char9_loss: 0.1196 - char10_loss: 0.0515 - char0_accuracy: 0.9937 - char1_accuracy: 0.9762 - char2_accuracy: 0.9886 - char3_accuracy: 0.9877 - char4_accuracy: 0.9698 - char5_accuracy: 0.9839 - char6_accuracy: 0.9845 - char7_accuracy: 0.9836 - char8_accuracy: 0.9859 - char9_accuracy: 0.9839 - char10_accuracy: 0.9906 - val_loss: 0.0255 - val_char0_loss: 3.8201e-04 - val_char1_loss: 0.0011 - val_char2_loss: 7.1723e-04 - val_char3_loss: 0.0011 - val_char4_loss: 0.0019 - val_char5_loss: 0.0012 - val_char6_loss: 0.0011 - val_char7_loss: 0.0017 - val_char8_loss: 0.0024 - val_char9_loss: 0.0053 - val_char10_loss: 0.0086 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9996
    Epoch 27/150
    47500/47500 [==============================] - 17s 356us/sample - loss: 0.0135 - char0_loss: 2.0202e-04 - char1_loss: 7.4443e-04 - char2_loss: 4.1601e-04 - char3_loss: 5.6271e-04 - char4_loss: 0.0011 - char5_loss: 6.6307e-04 - char6_loss: 7.0854e-04 - char7_loss: 0.0011 - char8_loss: 0.0012 - char9_loss: 0.0029 - char10_loss: 0.0039 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9999 - val_loss: 0.0117 - val_char0_loss: 2.5436e-04 - val_char1_loss: 5.4975e-04 - val_char2_loss: 3.0850e-04 - val_char3_loss: 4.1471e-04 - val_char4_loss: 9.0348e-04 - val_char5_loss: 5.0946e-04 - val_char6_loss: 5.5433e-04 - val_char7_loss: 8.7515e-04 - val_char8_loss: 9.2940e-04 - val_char9_loss: 0.0025 - val_char10_loss: 0.0038 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 28/150
    47500/47500 [==============================] - 17s 362us/sample - loss: 0.0073 - char0_loss: 1.1106e-04 - char1_loss: 4.1880e-04 - char2_loss: 2.3103e-04 - char3_loss: 2.9714e-04 - char4_loss: 6.5491e-04 - char5_loss: 3.8042e-04 - char6_loss: 4.1250e-04 - char7_loss: 6.3477e-04 - char8_loss: 7.5632e-04 - char9_loss: 0.0015 - char10_loss: 0.0019 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 0.0077 - val_char0_loss: 1.6536e-04 - val_char1_loss: 3.8004e-04 - val_char2_loss: 2.2433e-04 - val_char3_loss: 2.9798e-04 - val_char4_loss: 6.0860e-04 - val_char5_loss: 3.6353e-04 - val_char6_loss: 3.9013e-04 - val_char7_loss: 5.3166e-04 - val_char8_loss: 7.7784e-04 - val_char9_loss: 0.0015 - val_char10_loss: 0.0024 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000



![png](/assets/qr-nn/output_68_1.png)



![png](/assets/qr-nn/output_68_2.png)



![png](/assets/qr-nn/output_68_3.png)



![png](/assets/qr-nn/output_68_4.png)



{% highlight python %}
# let's zoom in into the tail of accuracy
for h in mo_hist.history:
    if h.startswith('val_') and h.endswith('_accuracy'):
        plt.plot(mo_hist.history[h][-20:], label=h[4:-9])
plt.title("Accuracy, last 20 epochs")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="best")
plt.show()

{% endhighlight %}


![png](/assets/qr-nn/output_69_0.png)



{% highlight python %}
# check how the best epoch looks like
best_epoch = np.argmin(mo_hist.history['val_loss'], axis=0)
print("Best epoch:", best_epoch, "out of", len(mo_hist.history['val_loss']))
for h in mo_hist.history:
    if h.startswith('val_'):
        print(h, mo_hist.history[h][best_epoch])
{% endhighlight %}

    Best epoch: 24 out of 28
    val_loss 0.0004742016372270882
    val_char0_loss 1.254878e-05
    val_char1_loss 1.1144185e-05
    val_char2_loss 1.4638953e-05
    val_char3_loss 2.2031929e-05
    val_char4_loss 2.5994039e-05
    val_char5_loss 1.9019555e-05
    val_char6_loss 2.4395402e-05
    val_char7_loss 2.4965897e-05
    val_char8_loss 3.329939e-05
    val_char9_loss 5.108057e-05
    val_char10_loss 0.00023854862
    val_char0_accuracy 1.0
    val_char1_accuracy 1.0
    val_char2_accuracy 1.0
    val_char3_accuracy 1.0
    val_char4_accuracy 1.0
    val_char5_accuracy 1.0
    val_char6_accuracy 1.0
    val_char7_accuracy 1.0
    val_char8_accuracy 1.0
    val_char9_accuracy 1.0
    val_char10_accuracy 1.0


The performance is pretty good.

We may give it a try and use the fact that size determination was working great.


{% highlight python %}
#
# INPUT -> FLATTEN -> HIDDEN_CHARS ->| for every char take both HIDDEN_CHARS and SIZE
#                  -> HIDDEN_SIZE -> 11 CLASSES(one per size)
#


def define_split_with_size_multi_output_model():
    input_layer = keras.layers.Input(shape=(21,21,), dtype='float', name='input_qr')
    flatten = keras.layers.Flatten(input_shape=(21, 21), name='flatten')(input_layer)    
    hidden_chars1 = keras.layers.Dense(21*21, activation='relu', name='hidden_chars1')(flatten)
    hidden_size = keras.layers.Dense(21*21, activation='relu', name='hidden_size')(flatten)
    size_output = keras.layers.Dense(MAX_SIZE, name='size_output')(hidden_size)

    # stop back propagation since size is independent from actual characters
    size_without_more_optimizations = tf.stop_gradient(size_output, name='size_wo_gradient')

    outputs = [size_output]
    for i in range(11):
        hidden_chars2 = keras.layers.Dense(21*21, activation='relu')(hidden_chars1)
        combined_char_inputs = keras.layers.concatenate([hidden_chars2, size_without_more_optimizations])
        char_output = keras.layers.Dense(len(ALL_CHAR_CLASSES), name='char' + str(i))(combined_char_inputs)
        outputs.append(char_output)

    multi_output_model = keras.Model(inputs=[input_layer], outputs=outputs)

    multi_output_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return multi_output_model

mo_model = define_split_with_size_multi_output_model()
mo_model.summary()
mo_hist = mo_model.fit(
        training_data, [
            np.asarray(training_label_sizes),
            np.asarray(training_labels_char0),
            np.asarray(training_labels_char1),
            np.asarray(training_labels_char2),
            np.asarray(training_labels_char3),
            np.asarray(training_labels_char4),
            np.asarray(training_labels_char5),
            np.asarray(training_labels_char6),
            np.asarray(training_labels_char7),
            np.asarray(training_labels_char8),
            np.asarray(training_labels_char9),
            np.asarray(training_labels_char10)
        ],
        epochs=150, batch_size=128,
        validation_split=0.05,
        callbacks=[EarlyStopping(monitor='val_loss' , min_delta=0.0001, patience=3)]
    )

plot_all_accuracy_and_loss(mo_hist)
plot_loss_vs_validation_loss_diff(mo_hist)
plot_loss_vs_validation_loss(mo_hist)
{% endhighlight %}

    Model: "model_4"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_qr (InputLayer)           [(None, 21, 21)]     0                                            
    __________________________________________________________________________________________________
    flatten (Flatten)               (None, 441)          0           input_qr[0][0]                   
    __________________________________________________________________________________________________
    hidden_size (Dense)             (None, 441)          194922      flatten[0][0]                    
    __________________________________________________________________________________________________
    size_output (Dense)             (None, 11)           4862        hidden_size[0][0]                
    __________________________________________________________________________________________________
    hidden_chars1 (Dense)           (None, 441)          194922      flatten[0][0]                    
    __________________________________________________________________________________________________
    dense_68 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    tf_op_layer_size_wo_gradient_1  [(None, 11)]         0           size_output[0][0]                
    __________________________________________________________________________________________________
    dense_69 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_70 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_71 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_72 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_73 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_74 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_75 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_76 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_77 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_78 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    concatenate_11 (Concatenate)    (None, 452)          0           dense_68[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_1[0]
    __________________________________________________________________________________________________
    concatenate_12 (Concatenate)    (None, 452)          0           dense_69[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_1[0]
    __________________________________________________________________________________________________
    concatenate_13 (Concatenate)    (None, 452)          0           dense_70[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_1[0]
    __________________________________________________________________________________________________
    concatenate_14 (Concatenate)    (None, 452)          0           dense_71[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_1[0]
    __________________________________________________________________________________________________
    concatenate_15 (Concatenate)    (None, 452)          0           dense_72[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_1[0]
    __________________________________________________________________________________________________
    concatenate_16 (Concatenate)    (None, 452)          0           dense_73[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_1[0]
    __________________________________________________________________________________________________
    concatenate_17 (Concatenate)    (None, 452)          0           dense_74[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_1[0]
    __________________________________________________________________________________________________
    concatenate_18 (Concatenate)    (None, 452)          0           dense_75[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_1[0]
    __________________________________________________________________________________________________
    concatenate_19 (Concatenate)    (None, 452)          0           dense_76[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_1[0]
    __________________________________________________________________________________________________
    concatenate_20 (Concatenate)    (None, 452)          0           dense_77[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_1[0]
    __________________________________________________________________________________________________
    concatenate_21 (Concatenate)    (None, 452)          0           dense_78[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_1[0]
    __________________________________________________________________________________________________
    char0 (Dense)                   (None, 28)           12684       concatenate_11[0][0]             
    __________________________________________________________________________________________________
    char1 (Dense)                   (None, 28)           12684       concatenate_12[0][0]             
    __________________________________________________________________________________________________
    char2 (Dense)                   (None, 28)           12684       concatenate_13[0][0]             
    __________________________________________________________________________________________________
    char3 (Dense)                   (None, 28)           12684       concatenate_14[0][0]             
    __________________________________________________________________________________________________
    char4 (Dense)                   (None, 28)           12684       concatenate_15[0][0]             
    __________________________________________________________________________________________________
    char5 (Dense)                   (None, 28)           12684       concatenate_16[0][0]             
    __________________________________________________________________________________________________
    char6 (Dense)                   (None, 28)           12684       concatenate_17[0][0]             
    __________________________________________________________________________________________________
    char7 (Dense)                   (None, 28)           12684       concatenate_18[0][0]             
    __________________________________________________________________________________________________
    char8 (Dense)                   (None, 28)           12684       concatenate_19[0][0]             
    __________________________________________________________________________________________________
    char9 (Dense)                   (None, 28)           12684       concatenate_20[0][0]             
    __________________________________________________________________________________________________
    char10 (Dense)                  (None, 28)           12684       concatenate_21[0][0]             
    ==================================================================================================
    Total params: 2,678,372
    Trainable params: 2,678,372
    Non-trainable params: 0
    __________________________________________________________________________________________________
    Train on 47500 samples, validate on 2500 samples
    Epoch 1/150
    47500/47500 [==============================] - 24s 501us/sample - loss: 18.9960 - size_output_loss: 0.4042 - char0_loss: 2.5609 - char1_loss: 2.3572 - char2_loss: 2.5476 - char3_loss: 2.2178 - char4_loss: 2.1085 - char5_loss: 1.8481 - char6_loss: 1.5832 - char7_loss: 1.2850 - char8_loss: 0.9977 - char9_loss: 0.7024 - char10_loss: 0.3687 - size_output_accuracy: 0.9047 - char0_accuracy: 0.2051 - char1_accuracy: 0.3218 - char2_accuracy: 0.2522 - char3_accuracy: 0.3399 - char4_accuracy: 0.4028 - char5_accuracy: 0.4854 - char6_accuracy: 0.5611 - char7_accuracy: 0.6497 - char8_accuracy: 0.7350 - char9_accuracy: 0.8202 - char10_accuracy: 0.9090 - val_loss: 13.4235 - val_size_output_loss: 0.0673 - val_char0_loss: 1.5513 - val_char1_loss: 1.0914 - val_char2_loss: 1.7299 - val_char3_loss: 1.5527 - val_char4_loss: 1.7107 - val_char5_loss: 1.5531 - val_char6_loss: 1.4050 - val_char7_loss: 1.0813 - val_char8_loss: 0.8142 - val_char9_loss: 0.5713 - val_char10_loss: 0.2780 - val_size_output_accuracy: 0.9940 - val_char0_accuracy: 0.3784 - val_char1_accuracy: 0.6996 - val_char2_accuracy: 0.4276 - val_char3_accuracy: 0.4272 - val_char4_accuracy: 0.4536 - val_char5_accuracy: 0.5236 - val_char6_accuracy: 0.5920 - val_char7_accuracy: 0.6884 - val_char8_accuracy: 0.7644 - val_char9_accuracy: 0.8400 - val_char10_accuracy: 0.9204
    Epoch 2/150
    47500/47500 [==============================] - 18s 386us/sample - loss: 7.3058 - size_output_loss: 0.0302 - char0_loss: 0.5094 - char1_loss: 0.2675 - char2_loss: 0.5726 - char3_loss: 0.7283 - char4_loss: 0.8118 - char5_loss: 0.8794 - char6_loss: 1.0389 - char7_loss: 0.8666 - char8_loss: 0.6944 - char9_loss: 0.5933 - char10_loss: 0.3005 - size_output_accuracy: 0.9980 - char0_accuracy: 0.8367 - char1_accuracy: 0.9450 - char2_accuracy: 0.8238 - char3_accuracy: 0.7569 - char4_accuracy: 0.7556 - char5_accuracy: 0.7130 - char6_accuracy: 0.6855 - char7_accuracy: 0.7259 - char8_accuracy: 0.7924 - char9_accuracy: 0.8306 - char10_accuracy: 0.9157 - val_loss: 2.7176 - val_size_output_loss: 0.0134 - val_char0_loss: 0.0461 - val_char1_loss: 0.0287 - val_char2_loss: 0.0461 - val_char3_loss: 0.0994 - val_char4_loss: 0.0917 - val_char5_loss: 0.2644 - val_char6_loss: 0.3959 - val_char7_loss: 0.4841 - val_char8_loss: 0.4488 - val_char9_loss: 0.5213 - val_char10_loss: 0.2731 - val_size_output_accuracy: 0.9996 - val_char0_accuracy: 0.9992 - val_char1_accuracy: 0.9992 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 0.9392 - val_char6_accuracy: 0.9276 - val_char7_accuracy: 0.8444 - val_char8_accuracy: 0.8472 - val_char9_accuracy: 0.8448 - val_char10_accuracy: 0.9220
    Epoch 3/150
    47500/47500 [==============================] - 20s 431us/sample - loss: 1.6010 - size_output_loss: 0.0078 - char0_loss: 0.0190 - char1_loss: 0.0128 - char2_loss: 0.0184 - char3_loss: 0.0336 - char4_loss: 0.0328 - char5_loss: 0.0809 - char6_loss: 0.1153 - char7_loss: 0.1841 - char8_loss: 0.3197 - char9_loss: 0.4983 - char10_loss: 0.2767 - size_output_accuracy: 0.9999 - char0_accuracy: 0.9999 - char1_accuracy: 0.9998 - char2_accuracy: 1.0000 - char3_accuracy: 0.9999 - char4_accuracy: 0.9999 - char5_accuracy: 0.9915 - char6_accuracy: 0.9913 - char7_accuracy: 0.9685 - char8_accuracy: 0.8866 - char9_accuracy: 0.8425 - char10_accuracy: 0.9179 - val_loss: 1.0212 - val_size_output_loss: 0.0048 - val_char0_loss: 0.0091 - val_char1_loss: 0.0069 - val_char2_loss: 0.0085 - val_char3_loss: 0.0128 - val_char4_loss: 0.0135 - val_char5_loss: 0.0231 - val_char6_loss: 0.0246 - val_char7_loss: 0.0458 - val_char8_loss: 0.1988 - val_char9_loss: 0.4292 - val_char10_loss: 0.2432 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 0.9992 - val_char8_accuracy: 0.9276 - val_char9_accuracy: 0.8536 - val_char10_accuracy: 0.9260
    Epoch 4/150
    47500/47500 [==============================] - 22s 471us/sample - loss: 0.8206 - size_output_loss: 0.0032 - char0_loss: 0.0057 - char1_loss: 0.0043 - char2_loss: 0.0053 - char3_loss: 0.0076 - char4_loss: 0.0076 - char5_loss: 0.0114 - char6_loss: 0.0133 - char7_loss: 0.0203 - char8_loss: 0.1264 - char9_loss: 0.3693 - char10_loss: 0.2464 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 0.9655 - char9_accuracy: 0.8684 - char10_accuracy: 0.9218 - val_loss: 0.6138 - val_size_output_loss: 0.0027 - val_char0_loss: 0.0042 - val_char1_loss: 0.0034 - val_char2_loss: 0.0038 - val_char3_loss: 0.0051 - val_char4_loss: 0.0051 - val_char5_loss: 0.0074 - val_char6_loss: 0.0078 - val_char7_loss: 0.0110 - val_char8_loss: 0.0558 - val_char9_loss: 0.2901 - val_char10_loss: 0.2176 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 0.9924 - val_char9_accuracy: 0.8884 - val_char10_accuracy: 0.9300
    Epoch 5/150
    47500/47500 [==============================] - 28s 596us/sample - loss: 0.5177 - size_output_loss: 0.0017 - char0_loss: 0.0029 - char1_loss: 0.0022 - char2_loss: 0.0027 - char3_loss: 0.0036 - char4_loss: 0.0035 - char5_loss: 0.0047 - char6_loss: 0.0053 - char7_loss: 0.0068 - char8_loss: 0.0287 - char9_loss: 0.2402 - char10_loss: 0.2144 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 0.9992 - char9_accuracy: 0.9150 - char10_accuracy: 0.9275 - val_loss: 0.3781 - val_size_output_loss: 0.0016 - val_char0_loss: 0.0024 - val_char1_loss: 0.0019 - val_char2_loss: 0.0021 - val_char3_loss: 0.0029 - val_char4_loss: 0.0029 - val_char5_loss: 0.0038 - val_char6_loss: 0.0040 - val_char7_loss: 0.0049 - val_char8_loss: 0.0152 - val_char9_loss: 0.1481 - val_char10_loss: 0.1882 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 0.9996 - val_char9_accuracy: 0.9604 - val_char10_accuracy: 0.9304
    Epoch 6/150
    47500/47500 [==============================] - 19s 410us/sample - loss: 0.4726 - size_output_loss: 0.0011 - char0_loss: 0.0054 - char1_loss: 0.0068 - char2_loss: 0.0594 - char3_loss: 0.0398 - char4_loss: 0.0423 - char5_loss: 0.0266 - char6_loss: 0.0107 - char7_loss: 0.0078 - char8_loss: 0.0150 - char9_loss: 0.0801 - char10_loss: 0.1766 - size_output_accuracy: 1.0000 - char0_accuracy: 0.9998 - char1_accuracy: 0.9992 - char2_accuracy: 0.9874 - char3_accuracy: 0.9907 - char4_accuracy: 0.9919 - char5_accuracy: 0.9959 - char6_accuracy: 0.9989 - char7_accuracy: 0.9998 - char8_accuracy: 0.9995 - char9_accuracy: 0.9907 - char10_accuracy: 0.9364 - val_loss: 0.2258 - val_size_output_loss: 0.0011 - val_char0_loss: 0.0016 - val_char1_loss: 0.0014 - val_char2_loss: 0.0031 - val_char3_loss: 0.0030 - val_char4_loss: 0.0035 - val_char5_loss: 0.0039 - val_char6_loss: 0.0027 - val_char7_loss: 0.0033 - val_char8_loss: 0.0071 - val_char9_loss: 0.0365 - val_char10_loss: 0.1596 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 0.9960 - val_char10_accuracy: 0.9392
    Epoch 7/150
    47500/47500 [==============================] - 18s 380us/sample - loss: 0.1801 - size_output_loss: 7.0053e-04 - char0_loss: 0.0011 - char1_loss: 7.8851e-04 - char2_loss: 0.0019 - char3_loss: 0.0018 - char4_loss: 0.0024 - char5_loss: 0.0022 - char6_loss: 0.0018 - char7_loss: 0.0021 - char8_loss: 0.0044 - char9_loss: 0.0200 - char10_loss: 0.1406 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 0.9999 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 0.9996 - char10_accuracy: 0.9463 - val_loss: 0.1583 - val_size_output_loss: 7.0115e-04 - val_char0_loss: 8.9383e-04 - val_char1_loss: 6.5773e-04 - val_char2_loss: 0.0014 - val_char3_loss: 0.0014 - val_char4_loss: 0.0014 - val_char5_loss: 0.0019 - val_char6_loss: 0.0014 - val_char7_loss: 0.0017 - val_char8_loss: 0.0037 - val_char9_loss: 0.0139 - val_char10_loss: 0.1302 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 0.9996 - val_char10_accuracy: 0.9500
    Epoch 8/150
    47500/47500 [==============================] - 18s 374us/sample - loss: 0.1259 - size_output_loss: 4.8805e-04 - char0_loss: 6.9390e-04 - char1_loss: 4.9064e-04 - char2_loss: 0.0011 - char3_loss: 0.0011 - char4_loss: 0.0011 - char5_loss: 0.0013 - char6_loss: 0.0011 - char7_loss: 0.0013 - char8_loss: 0.0025 - char9_loss: 0.0081 - char10_loss: 0.1065 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9592 - val_loss: 0.1189 - val_size_output_loss: 5.1153e-04 - val_char0_loss: 6.6655e-04 - val_char1_loss: 4.8759e-04 - val_char2_loss: 9.2972e-04 - val_char3_loss: 0.0010 - val_char4_loss: 0.0010 - val_char5_loss: 0.0012 - val_char6_loss: 0.0010 - val_char7_loss: 0.0012 - val_char8_loss: 0.0023 - val_char9_loss: 0.0071 - val_char10_loss: 0.1032 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 0.9996 - val_char10_accuracy: 0.9572
    Epoch 9/150
    47500/47500 [==============================] - 20s 411us/sample - loss: 0.0910 - size_output_loss: 3.4824e-04 - char0_loss: 5.1653e-04 - char1_loss: 3.6368e-04 - char2_loss: 7.4462e-04 - char3_loss: 7.8575e-04 - char4_loss: 7.9390e-04 - char5_loss: 8.9367e-04 - char6_loss: 8.0486e-04 - char7_loss: 8.8263e-04 - char8_loss: 0.0016 - char9_loss: 0.0043 - char10_loss: 0.0788 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9684 - val_loss: 0.0918 - val_size_output_loss: 3.7844e-04 - val_char0_loss: 4.8163e-04 - val_char1_loss: 3.6973e-04 - val_char2_loss: 6.8291e-04 - val_char3_loss: 7.3452e-04 - val_char4_loss: 8.3240e-04 - val_char5_loss: 9.2194e-04 - val_char6_loss: 7.6042e-04 - val_char7_loss: 8.5188e-04 - val_char8_loss: 0.0016 - val_char9_loss: 0.0040 - val_char10_loss: 0.0814 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9628
    Epoch 10/150
    47500/47500 [==============================] - 18s 387us/sample - loss: 0.0682 - size_output_loss: 2.5652e-04 - char0_loss: 3.9509e-04 - char1_loss: 2.8533e-04 - char2_loss: 5.6065e-04 - char3_loss: 6.0264e-04 - char4_loss: 6.0456e-04 - char5_loss: 6.7553e-04 - char6_loss: 6.1035e-04 - char7_loss: 6.5227e-04 - char8_loss: 0.0011 - char9_loss: 0.0026 - char10_loss: 0.0599 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9755 - val_loss: 0.0685 - val_size_output_loss: 2.7697e-04 - val_char0_loss: 3.8373e-04 - val_char1_loss: 2.9319e-04 - val_char2_loss: 5.2759e-04 - val_char3_loss: 6.1833e-04 - val_char4_loss: 6.3682e-04 - val_char5_loss: 7.1171e-04 - val_char6_loss: 5.8763e-04 - val_char7_loss: 6.7575e-04 - val_char8_loss: 0.0013 - val_char9_loss: 0.0026 - val_char10_loss: 0.0614 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9720
    Epoch 11/150
    47500/47500 [==============================] - 19s 402us/sample - loss: 0.0441 - size_output_loss: 1.9250e-04 - char0_loss: 3.1589e-04 - char1_loss: 2.3197e-04 - char2_loss: 4.4025e-04 - char3_loss: 4.7382e-04 - char4_loss: 4.8968e-04 - char5_loss: 5.2810e-04 - char6_loss: 4.7244e-04 - char7_loss: 4.9678e-04 - char8_loss: 8.3676e-04 - char9_loss: 0.0017 - char10_loss: 0.0379 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9879 - val_loss: 0.0400 - val_size_output_loss: 2.1984e-04 - val_char0_loss: 3.1023e-04 - val_char1_loss: 2.3848e-04 - val_char2_loss: 4.0783e-04 - val_char3_loss: 4.6645e-04 - val_char4_loss: 5.0626e-04 - val_char5_loss: 5.6510e-04 - val_char6_loss: 4.4491e-04 - val_char7_loss: 5.0870e-04 - val_char8_loss: 9.1498e-04 - val_char9_loss: 0.0019 - val_char10_loss: 0.0339 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9896
    Epoch 12/150
    47500/47500 [==============================] - 21s 437us/sample - loss: 0.0225 - size_output_loss: 1.4735e-04 - char0_loss: 2.3860e-04 - char1_loss: 1.7761e-04 - char2_loss: 3.3508e-04 - char3_loss: 3.6764e-04 - char4_loss: 3.7098e-04 - char5_loss: 4.0250e-04 - char6_loss: 3.5427e-04 - char7_loss: 3.6811e-04 - char8_loss: 6.0472e-04 - char9_loss: 0.0011 - char10_loss: 0.0180 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9969 - val_loss: 0.0203 - val_size_output_loss: 1.7088e-04 - val_char0_loss: 2.2562e-04 - val_char1_loss: 1.8088e-04 - val_char2_loss: 3.0709e-04 - val_char3_loss: 3.5274e-04 - val_char4_loss: 3.7050e-04 - val_char5_loss: 4.1212e-04 - val_char6_loss: 3.3892e-04 - val_char7_loss: 3.6599e-04 - val_char8_loss: 6.2879e-04 - val_char9_loss: 0.0014 - val_char10_loss: 0.0160 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9956
    Epoch 13/150
    47500/47500 [==============================] - 18s 370us/sample - loss: 0.0112 - size_output_loss: 1.1340e-04 - char0_loss: 1.7954e-04 - char1_loss: 1.3649e-04 - char2_loss: 2.5585e-04 - char3_loss: 2.7900e-04 - char4_loss: 2.8483e-04 - char5_loss: 3.0589e-04 - char6_loss: 2.6667e-04 - char7_loss: 2.7170e-04 - char8_loss: 4.3664e-04 - char9_loss: 7.9480e-04 - char10_loss: 0.0078 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9994 - val_loss: 0.0108 - val_size_output_loss: 1.3365e-04 - val_char0_loss: 1.6907e-04 - val_char1_loss: 1.4117e-04 - val_char2_loss: 2.4439e-04 - val_char3_loss: 2.6621e-04 - val_char4_loss: 2.9474e-04 - val_char5_loss: 3.5744e-04 - val_char6_loss: 2.5903e-04 - val_char7_loss: 2.7383e-04 - val_char8_loss: 4.7587e-04 - val_char9_loss: 0.0012 - val_char10_loss: 0.0072 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 0.9996 - val_char10_accuracy: 1.0000
    Epoch 14/150
    47500/47500 [==============================] - 18s 378us/sample - loss: 0.0063 - size_output_loss: 8.7497e-05 - char0_loss: 1.3810e-04 - char1_loss: 1.0632e-04 - char2_loss: 2.0017e-04 - char3_loss: 2.1750e-04 - char4_loss: 2.2389e-04 - char5_loss: 2.3827e-04 - char6_loss: 2.0433e-04 - char7_loss: 2.0418e-04 - char8_loss: 3.2623e-04 - char9_loss: 5.5949e-04 - char10_loss: 0.0038 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9999 - val_loss: 0.0065 - val_size_output_loss: 1.1283e-04 - val_char0_loss: 1.3735e-04 - val_char1_loss: 1.1150e-04 - val_char2_loss: 1.8952e-04 - val_char3_loss: 2.1471e-04 - val_char4_loss: 2.3001e-04 - val_char5_loss: 2.4995e-04 - val_char6_loss: 1.9782e-04 - val_char7_loss: 2.1541e-04 - val_char8_loss: 3.5948e-04 - val_char9_loss: 7.2153e-04 - val_char10_loss: 0.0038 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 15/150
    47500/47500 [==============================] - 19s 404us/sample - loss: 0.0042 - size_output_loss: 6.8527e-05 - char0_loss: 1.0929e-04 - char1_loss: 8.4140e-05 - char2_loss: 1.5832e-04 - char3_loss: 1.7162e-04 - char4_loss: 1.7934e-04 - char5_loss: 1.8565e-04 - char6_loss: 1.5876e-04 - char7_loss: 1.5686e-04 - char8_loss: 2.4645e-04 - char9_loss: 4.0683e-04 - char10_loss: 0.0022 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 0.0047 - val_size_output_loss: 8.5045e-05 - val_char0_loss: 1.0534e-04 - val_char1_loss: 8.8971e-05 - val_char2_loss: 1.5081e-04 - val_char3_loss: 1.7135e-04 - val_char4_loss: 2.0713e-04 - val_char5_loss: 1.9500e-04 - val_char6_loss: 1.5606e-04 - val_char7_loss: 1.6289e-04 - val_char8_loss: 2.8636e-04 - val_char9_loss: 5.3834e-04 - val_char10_loss: 0.0026 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 16/150
    47500/47500 [==============================] - 18s 388us/sample - loss: 0.0030 - size_output_loss: 5.3991e-05 - char0_loss: 8.6060e-05 - char1_loss: 6.7463e-05 - char2_loss: 1.2680e-04 - char3_loss: 1.3627e-04 - char4_loss: 1.4300e-04 - char5_loss: 1.4614e-04 - char6_loss: 1.2543e-04 - char7_loss: 1.2221e-04 - char8_loss: 1.8992e-04 - char9_loss: 3.0391e-04 - char10_loss: 0.0015 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 0.0035 - val_size_output_loss: 6.7495e-05 - val_char0_loss: 8.8391e-05 - val_char1_loss: 7.1388e-05 - val_char2_loss: 1.2127e-04 - val_char3_loss: 1.4180e-04 - val_char4_loss: 1.6610e-04 - val_char5_loss: 1.5247e-04 - val_char6_loss: 1.2320e-04 - val_char7_loss: 1.2861e-04 - val_char8_loss: 2.1303e-04 - val_char9_loss: 4.2062e-04 - val_char10_loss: 0.0019 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 17/150
    47500/47500 [==============================] - 18s 377us/sample - loss: 0.0021 - size_output_loss: 4.2725e-05 - char0_loss: 6.8551e-05 - char1_loss: 5.3974e-05 - char2_loss: 1.0220e-04 - char3_loss: 1.0872e-04 - char4_loss: 1.1495e-04 - char5_loss: 1.1635e-04 - char6_loss: 9.9073e-05 - char7_loss: 9.5596e-05 - char8_loss: 1.4744e-04 - char9_loss: 2.2719e-04 - char10_loss: 9.1579e-04 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 0.0027 - val_size_output_loss: 5.5528e-05 - val_char0_loss: 6.7232e-05 - val_char1_loss: 5.8102e-05 - val_char2_loss: 9.8377e-05 - val_char3_loss: 1.1191e-04 - val_char4_loss: 1.4186e-04 - val_char5_loss: 1.3361e-04 - val_char6_loss: 9.8835e-05 - val_char7_loss: 1.0295e-04 - val_char8_loss: 1.6435e-04 - val_char9_loss: 3.4459e-04 - val_char10_loss: 0.0013 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 18/150
    47500/47500 [==============================] - 21s 445us/sample - loss: 0.0017 - size_output_loss: 3.3895e-05 - char0_loss: 5.6638e-05 - char1_loss: 4.3725e-05 - char2_loss: 8.2841e-05 - char3_loss: 8.7855e-05 - char4_loss: 9.4119e-05 - char5_loss: 9.3799e-05 - char6_loss: 7.9154e-05 - char7_loss: 7.5899e-05 - char8_loss: 1.1653e-04 - char9_loss: 1.7462e-04 - char10_loss: 7.6168e-04 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 0.0019 - val_size_output_loss: 4.4678e-05 - val_char0_loss: 5.4375e-05 - val_char1_loss: 4.7667e-05 - val_char2_loss: 7.9730e-05 - val_char3_loss: 8.6788e-05 - val_char4_loss: 1.0074e-04 - val_char5_loss: 1.0173e-04 - val_char6_loss: 7.9290e-05 - val_char7_loss: 8.1004e-05 - val_char8_loss: 1.3326e-04 - val_char9_loss: 2.7137e-04 - val_char10_loss: 8.8405e-04 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 19/150
    47500/47500 [==============================] - 19s 409us/sample - loss: 0.0012 - size_output_loss: 2.7041e-05 - char0_loss: 4.3965e-05 - char1_loss: 3.5388e-05 - char2_loss: 6.7181e-05 - char3_loss: 7.0922e-05 - char4_loss: 7.5867e-05 - char5_loss: 7.5277e-05 - char6_loss: 6.3249e-05 - char7_loss: 6.0091e-05 - char8_loss: 9.0333e-05 - char9_loss: 1.3377e-04 - char10_loss: 4.7975e-04 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 0.0015 - val_size_output_loss: 3.4825e-05 - val_char0_loss: 4.2844e-05 - val_char1_loss: 3.8444e-05 - val_char2_loss: 6.5300e-05 - val_char3_loss: 7.0345e-05 - val_char4_loss: 9.4447e-05 - val_char5_loss: 8.0086e-05 - val_char6_loss: 6.3934e-05 - val_char7_loss: 6.6175e-05 - val_char8_loss: 1.0426e-04 - val_char9_loss: 1.9093e-04 - val_char10_loss: 7.0616e-04 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 20/150
    47500/47500 [==============================] - 19s 393us/sample - loss: 9.4015e-04 - size_output_loss: 2.1622e-05 - char0_loss: 3.5311e-05 - char1_loss: 2.8577e-05 - char2_loss: 5.4623e-05 - char3_loss: 5.7400e-05 - char4_loss: 6.1571e-05 - char5_loss: 6.0533e-05 - char6_loss: 5.0605e-05 - char7_loss: 4.7826e-05 - char8_loss: 7.0873e-05 - char9_loss: 1.0295e-04 - char10_loss: 3.4695e-04 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 0.0013 - val_size_output_loss: 2.8500e-05 - val_char0_loss: 3.5148e-05 - val_char1_loss: 3.1155e-05 - val_char2_loss: 5.2460e-05 - val_char3_loss: 5.9400e-05 - val_char4_loss: 9.0367e-05 - val_char5_loss: 6.6542e-05 - val_char6_loss: 5.1119e-05 - val_char7_loss: 5.2647e-05 - val_char8_loss: 8.4618e-05 - val_char9_loss: 1.4603e-04 - val_char10_loss: 5.6755e-04 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 21/150
    47500/47500 [==============================] - 19s 399us/sample - loss: 7.4018e-04 - size_output_loss: 1.7278e-05 - char0_loss: 2.8334e-05 - char1_loss: 2.3217e-05 - char2_loss: 4.4191e-05 - char3_loss: 4.6546e-05 - char4_loss: 5.0015e-05 - char5_loss: 4.9100e-05 - char6_loss: 4.0832e-05 - char7_loss: 3.8288e-05 - char8_loss: 5.6502e-05 - char9_loss: 8.0290e-05 - char10_loss: 2.6475e-04 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 0.0010 - val_size_output_loss: 2.4362e-05 - val_char0_loss: 3.1667e-05 - val_char1_loss: 2.5349e-05 - val_char2_loss: 4.2374e-05 - val_char3_loss: 5.2510e-05 - val_char4_loss: 6.6185e-05 - val_char5_loss: 5.3509e-05 - val_char6_loss: 4.1419e-05 - val_char7_loss: 4.1652e-05 - val_char8_loss: 6.7186e-05 - val_char9_loss: 1.2865e-04 - val_char10_loss: 4.6972e-04 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 22/150
    47500/47500 [==============================] - 19s 406us/sample - loss: 0.5982 - size_output_loss: 1.3883e-05 - char0_loss: 0.4024 - char1_loss: 0.0515 - char2_loss: 0.0252 - char3_loss: 0.0068 - char4_loss: 0.0149 - char5_loss: 0.0187 - char6_loss: 0.0117 - char7_loss: 0.0169 - char8_loss: 0.0193 - char9_loss: 0.0206 - char10_loss: 0.0087 - size_output_accuracy: 1.0000 - char0_accuracy: 0.9572 - char1_accuracy: 0.9864 - char2_accuracy: 0.9920 - char3_accuracy: 0.9981 - char4_accuracy: 0.9956 - char5_accuracy: 0.9941 - char6_accuracy: 0.9963 - char7_accuracy: 0.9945 - char8_accuracy: 0.9947 - char9_accuracy: 0.9939 - char10_accuracy: 0.9973 - val_loss: 0.0129 - val_size_output_loss: 1.8566e-05 - val_char0_loss: 0.0072 - val_char1_loss: 5.3554e-04 - val_char2_loss: 3.9978e-04 - val_char3_loss: 2.3171e-04 - val_char4_loss: 5.8392e-04 - val_char5_loss: 5.1264e-04 - val_char6_loss: 2.8942e-04 - val_char7_loss: 4.2136e-04 - val_char8_loss: 6.7547e-04 - val_char9_loss: 8.3108e-04 - val_char10_loss: 0.0012 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 23/150
    47500/47500 [==============================] - 18s 387us/sample - loss: 0.0072 - size_output_loss: 1.1145e-05 - char0_loss: 0.0040 - char1_loss: 3.5238e-04 - char2_loss: 2.3902e-04 - char3_loss: 1.2201e-04 - char4_loss: 2.7721e-04 - char5_loss: 3.3965e-04 - char6_loss: 1.8259e-04 - char7_loss: 2.6997e-04 - char8_loss: 4.0561e-04 - char9_loss: 4.8175e-04 - char10_loss: 5.1069e-04 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 0.0054 - val_size_output_loss: 1.4899e-05 - val_char0_loss: 0.0025 - val_char1_loss: 2.4707e-04 - val_char2_loss: 1.9222e-04 - val_char3_loss: 1.1141e-04 - val_char4_loss: 2.7321e-04 - val_char5_loss: 2.5353e-04 - val_char6_loss: 1.3555e-04 - val_char7_loss: 1.9920e-04 - val_char8_loss: 3.3179e-04 - val_char9_loss: 4.6305e-04 - val_char10_loss: 6.5441e-04 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 24/150
    47500/47500 [==============================] - 18s 381us/sample - loss: 0.0036 - size_output_loss: 8.9727e-06 - char0_loss: 0.0018 - char1_loss: 1.9250e-04 - char2_loss: 1.3311e-04 - char3_loss: 7.0725e-05 - char4_loss: 1.5141e-04 - char5_loss: 1.9256e-04 - char6_loss: 1.0311e-04 - char7_loss: 1.5120e-04 - char8_loss: 2.2634e-04 - char9_loss: 2.7212e-04 - char10_loss: 2.9921e-04 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 0.0033 - val_size_output_loss: 1.2399e-05 - val_char0_loss: 0.0014 - val_char1_loss: 1.6224e-04 - val_char2_loss: 1.2709e-04 - val_char3_loss: 7.6828e-05 - val_char4_loss: 1.9743e-04 - val_char5_loss: 1.6742e-04 - val_char6_loss: 9.1447e-05 - val_char7_loss: 1.3252e-04 - val_char8_loss: 2.2349e-04 - val_char9_loss: 3.0969e-04 - val_char10_loss: 4.4204e-04 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000



![png](/assets/qr-nn/output_72_1.png)



![png](/assets/qr-nn/output_72_2.png)



![png](/assets/qr-nn/output_72_3.png)



![png](/assets/qr-nn/output_72_4.png)



{% highlight python %}
# let's zoom in into the tail of accuracy
for h in mo_hist.history:
    if h.startswith('val_') and h.endswith('_accuracy'):
        plt.plot(mo_hist.history[h][-20:], label=h[4:-9])
plt.title("Accuracy, last 20 epochs")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="best")
plt.show()


# let's zoom in into the tail of loss
plt.plot(mo_hist.history['loss'][-20:], label='loss')
plt.plot(mo_hist.history['val_loss'][-20:], label='val_loss')
plt.title("Loss, last 20 epochs")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="best")
plt.show()

{% endhighlight %}


![png](/assets/qr-nn/output_73_0.png)



![png](/assets/qr-nn/output_73_1.png)


## FINAL model
Since the last model workes great, let's retrain it with checkpoint savings


{% highlight python %}
mo_model = define_split_with_size_multi_output_model()
mo_model.summary()
mo_hist = mo_model.fit(
        training_data, [
            np.asarray(training_label_sizes),
            np.asarray(training_labels_char0),
            np.asarray(training_labels_char1),
            np.asarray(training_labels_char2),
            np.asarray(training_labels_char3),
            np.asarray(training_labels_char4),
            np.asarray(training_labels_char5),
            np.asarray(training_labels_char6),
            np.asarray(training_labels_char7),
            np.asarray(training_labels_char8),
            np.asarray(training_labels_char9),
            np.asarray(training_labels_char10)
        ],
        epochs=150, batch_size=128,
        validation_split=0.05,
        callbacks=[
            EarlyStopping(monitor='val_loss' , min_delta=0.0001, patience=3),
            ModelCheckpoint(filepath='qr-ffnn.h5', monitor='val_loss', save_best_only=True, verbose=1)
        ]
    )
{% endhighlight %}

    Model: "model_5"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_qr (InputLayer)           [(None, 21, 21)]     0                                            
    __________________________________________________________________________________________________
    flatten (Flatten)               (None, 441)          0           input_qr[0][0]                   
    __________________________________________________________________________________________________
    hidden_size (Dense)             (None, 441)          194922      flatten[0][0]                    
    __________________________________________________________________________________________________
    size_output (Dense)             (None, 11)           4862        hidden_size[0][0]                
    __________________________________________________________________________________________________
    hidden_chars1 (Dense)           (None, 441)          194922      flatten[0][0]                    
    __________________________________________________________________________________________________
    dense_79 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    tf_op_layer_size_wo_gradient_2  [(None, 11)]         0           size_output[0][0]                
    __________________________________________________________________________________________________
    dense_80 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_81 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_82 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_83 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_84 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_85 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_86 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_87 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_88 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    dense_89 (Dense)                (None, 441)          194922      hidden_chars1[0][0]              
    __________________________________________________________________________________________________
    concatenate_22 (Concatenate)    (None, 452)          0           dense_79[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_2[0]
    __________________________________________________________________________________________________
    concatenate_23 (Concatenate)    (None, 452)          0           dense_80[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_2[0]
    __________________________________________________________________________________________________
    concatenate_24 (Concatenate)    (None, 452)          0           dense_81[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_2[0]
    __________________________________________________________________________________________________
    concatenate_25 (Concatenate)    (None, 452)          0           dense_82[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_2[0]
    __________________________________________________________________________________________________
    concatenate_26 (Concatenate)    (None, 452)          0           dense_83[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_2[0]
    __________________________________________________________________________________________________
    concatenate_27 (Concatenate)    (None, 452)          0           dense_84[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_2[0]
    __________________________________________________________________________________________________
    concatenate_28 (Concatenate)    (None, 452)          0           dense_85[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_2[0]
    __________________________________________________________________________________________________
    concatenate_29 (Concatenate)    (None, 452)          0           dense_86[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_2[0]
    __________________________________________________________________________________________________
    concatenate_30 (Concatenate)    (None, 452)          0           dense_87[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_2[0]
    __________________________________________________________________________________________________
    concatenate_31 (Concatenate)    (None, 452)          0           dense_88[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_2[0]
    __________________________________________________________________________________________________
    concatenate_32 (Concatenate)    (None, 452)          0           dense_89[0][0]                   
                                                                     tf_op_layer_size_wo_gradient_2[0]
    __________________________________________________________________________________________________
    char0 (Dense)                   (None, 28)           12684       concatenate_22[0][0]             
    __________________________________________________________________________________________________
    char1 (Dense)                   (None, 28)           12684       concatenate_23[0][0]             
    __________________________________________________________________________________________________
    char2 (Dense)                   (None, 28)           12684       concatenate_24[0][0]             
    __________________________________________________________________________________________________
    char3 (Dense)                   (None, 28)           12684       concatenate_25[0][0]             
    __________________________________________________________________________________________________
    char4 (Dense)                   (None, 28)           12684       concatenate_26[0][0]             
    __________________________________________________________________________________________________
    char5 (Dense)                   (None, 28)           12684       concatenate_27[0][0]             
    __________________________________________________________________________________________________
    char6 (Dense)                   (None, 28)           12684       concatenate_28[0][0]             
    __________________________________________________________________________________________________
    char7 (Dense)                   (None, 28)           12684       concatenate_29[0][0]             
    __________________________________________________________________________________________________
    char8 (Dense)                   (None, 28)           12684       concatenate_30[0][0]             
    __________________________________________________________________________________________________
    char9 (Dense)                   (None, 28)           12684       concatenate_31[0][0]             
    __________________________________________________________________________________________________
    char10 (Dense)                  (None, 28)           12684       concatenate_32[0][0]             
    ==================================================================================================
    Total params: 2,678,372
    Trainable params: 2,678,372
    Non-trainable params: 0
    __________________________________________________________________________________________________
    Train on 47500 samples, validate on 2500 samples
    Epoch 1/150
    47488/47500 [============================>.] - ETA: 0s - loss: 19.1971 - size_output_loss: 0.4180 - char0_loss: 2.6202 - char1_loss: 2.4073 - char2_loss: 2.5896 - char3_loss: 2.2260 - char4_loss: 2.1343 - char5_loss: 1.8537 - char6_loss: 1.5963 - char7_loss: 1.2869 - char8_loss: 1.0011 - char9_loss: 0.6995 - char10_loss: 0.3642 - size_output_accuracy: 0.8998 - char0_accuracy: 0.1919 - char1_accuracy: 0.3048 - char2_accuracy: 0.2443 - char3_accuracy: 0.3413 - char4_accuracy: 0.3994 - char5_accuracy: 0.4859 - char6_accuracy: 0.5622 - char7_accuracy: 0.6487 - char8_accuracy: 0.7343 - char9_accuracy: 0.8201 - char10_accuracy: 0.9092
    Epoch 00001: val_loss improved from inf to 14.00581, saving model to qr-ffnn.h5
    47500/47500 [==============================] - 26s 548us/sample - loss: 19.1973 - size_output_loss: 0.4170 - char0_loss: 2.6177 - char1_loss: 2.4046 - char2_loss: 2.5892 - char3_loss: 2.2257 - char4_loss: 2.1362 - char5_loss: 1.8563 - char6_loss: 1.5980 - char7_loss: 1.2881 - char8_loss: 1.0010 - char9_loss: 0.7001 - char10_loss: 0.3651 - size_output_accuracy: 0.8998 - char0_accuracy: 0.1920 - char1_accuracy: 0.3049 - char2_accuracy: 0.2443 - char3_accuracy: 0.3413 - char4_accuracy: 0.3994 - char5_accuracy: 0.4858 - char6_accuracy: 0.5621 - char7_accuracy: 0.6486 - char8_accuracy: 0.7344 - char9_accuracy: 0.8201 - char10_accuracy: 0.9092 - val_loss: 14.0058 - val_size_output_loss: 0.0634 - val_char0_loss: 1.7132 - val_char1_loss: 1.2125 - val_char2_loss: 1.8853 - val_char3_loss: 1.6198 - val_char4_loss: 1.7714 - val_char5_loss: 1.5487 - val_char6_loss: 1.4140 - val_char7_loss: 1.0889 - val_char8_loss: 0.8272 - val_char9_loss: 0.5618 - val_char10_loss: 0.2782 - val_size_output_accuracy: 0.9928 - val_char0_accuracy: 0.3304 - val_char1_accuracy: 0.6664 - val_char2_accuracy: 0.3860 - val_char3_accuracy: 0.4308 - val_char4_accuracy: 0.4372 - val_char5_accuracy: 0.5328 - val_char6_accuracy: 0.5852 - val_char7_accuracy: 0.6844 - val_char8_accuracy: 0.7640 - val_char9_accuracy: 0.8416 - val_char10_accuracy: 0.9212
    Epoch 2/150
    47488/47500 [============================>.] - ETA: 0s - loss: 7.8029 - size_output_loss: 0.0315 - char0_loss: 0.6359 - char1_loss: 0.3084 - char2_loss: 0.6456 - char3_loss: 0.7828 - char4_loss: 0.9234 - char5_loss: 0.8953 - char6_loss: 1.0818 - char7_loss: 0.8631 - char8_loss: 0.7519 - char9_loss: 0.5827 - char10_loss: 0.3005 - size_output_accuracy: 0.9978 - char0_accuracy: 0.8029 - char1_accuracy: 0.9348 - char2_accuracy: 0.8162 - char3_accuracy: 0.7161 - char4_accuracy: 0.7072 - char5_accuracy: 0.7135 - char6_accuracy: 0.6665 - char7_accuracy: 0.7213 - char8_accuracy: 0.7778 - char9_accuracy: 0.8312 - char10_accuracy: 0.9152
    Epoch 00002: val_loss improved from 14.00581 to 3.06730, saving model to qr-ffnn.h5
    47500/47500 [==============================] - 18s 372us/sample - loss: 7.8012 - size_output_loss: 0.0314 - char0_loss: 0.6344 - char1_loss: 0.3076 - char2_loss: 0.6440 - char3_loss: 0.7811 - char4_loss: 0.9211 - char5_loss: 0.8932 - char6_loss: 1.0795 - char7_loss: 0.8609 - char8_loss: 0.7506 - char9_loss: 0.5821 - char10_loss: 0.2997 - size_output_accuracy: 0.9978 - char0_accuracy: 0.8030 - char1_accuracy: 0.9348 - char2_accuracy: 0.8163 - char3_accuracy: 0.7162 - char4_accuracy: 0.7072 - char5_accuracy: 0.7136 - char6_accuracy: 0.6666 - char7_accuracy: 0.7213 - char8_accuracy: 0.7778 - char9_accuracy: 0.8313 - char10_accuracy: 0.9153 - val_loss: 3.0673 - val_size_output_loss: 0.0139 - val_char0_loss: 0.0554 - val_char1_loss: 0.0303 - val_char2_loss: 0.0481 - val_char3_loss: 0.1804 - val_char4_loss: 0.1695 - val_char5_loss: 0.2560 - val_char6_loss: 0.4972 - val_char7_loss: 0.5444 - val_char8_loss: 0.5055 - val_char9_loss: 0.4911 - val_char10_loss: 0.2715 - val_size_output_accuracy: 0.9996 - val_char0_accuracy: 0.9992 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 0.9600 - val_char4_accuracy: 0.9884 - val_char5_accuracy: 0.9424 - val_char6_accuracy: 0.8720 - val_char7_accuracy: 0.7940 - val_char8_accuracy: 0.8344 - val_char9_accuracy: 0.8468 - val_char10_accuracy: 0.9204
    Epoch 3/150
    47488/47500 [============================>.] - ETA: 0s - loss: 1.8218 - size_output_loss: 0.0089 - char0_loss: 0.0220 - char1_loss: 0.0135 - char2_loss: 0.0191 - char3_loss: 0.0552 - char4_loss: 0.0497 - char5_loss: 0.0657 - char6_loss: 0.1532 - char7_loss: 0.3108 - char8_loss: 0.3678 - char9_loss: 0.4834 - char10_loss: 0.2725 - size_output_accuracy: 0.9998 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 0.9968 - char4_accuracy: 0.9991 - char5_accuracy: 0.9986 - char6_accuracy: 0.9811 - char7_accuracy: 0.8981 - char8_accuracy: 0.8768 - char9_accuracy: 0.8431 - char10_accuracy: 0.9197
    Epoch 00003: val_loss improved from 3.06730 to 1.09164, saving model to qr-ffnn.h5
    47500/47500 [==============================] - 18s 372us/sample - loss: 1.8216 - size_output_loss: 0.0089 - char0_loss: 0.0220 - char1_loss: 0.0135 - char2_loss: 0.0191 - char3_loss: 0.0551 - char4_loss: 0.0497 - char5_loss: 0.0656 - char6_loss: 0.1528 - char7_loss: 0.3101 - char8_loss: 0.3670 - char9_loss: 0.4833 - char10_loss: 0.2730 - size_output_accuracy: 0.9998 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 0.9968 - char4_accuracy: 0.9991 - char5_accuracy: 0.9986 - char6_accuracy: 0.9811 - char7_accuracy: 0.8981 - char8_accuracy: 0.8768 - char9_accuracy: 0.8431 - char10_accuracy: 0.9197 - val_loss: 1.0916 - val_size_output_loss: 0.0057 - val_char0_loss: 0.0097 - val_char1_loss: 0.0073 - val_char2_loss: 0.0089 - val_char3_loss: 0.0166 - val_char4_loss: 0.0157 - val_char5_loss: 0.0189 - val_char6_loss: 0.0286 - val_char7_loss: 0.0989 - val_char8_loss: 0.2258 - val_char9_loss: 0.4105 - val_char10_loss: 0.2438 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 0.9896 - val_char8_accuracy: 0.9232 - val_char9_accuracy: 0.8600 - val_char10_accuracy: 0.9232
    Epoch 4/150
    47360/47500 [============================>.] - ETA: 0s - loss: 0.9257 - size_output_loss: 0.0037 - char0_loss: 0.0075 - char1_loss: 0.0059 - char2_loss: 0.0136 - char3_loss: 0.0270 - char4_loss: 0.0209 - char5_loss: 0.0333 - char6_loss: 0.0298 - char7_loss: 0.0435 - char8_loss: 0.1219 - char9_loss: 0.3862 - char10_loss: 0.2324 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 0.9986 - char3_accuracy: 0.9965 - char4_accuracy: 0.9978 - char5_accuracy: 0.9952 - char6_accuracy: 0.9974 - char7_accuracy: 0.9985 - char8_accuracy: 0.9775 - char9_accuracy: 0.8678 - char10_accuracy: 0.9249
    Epoch 00004: val_loss improved from 1.09164 to 0.60665, saving model to qr-ffnn.h5
    47500/47500 [==============================] - 19s 407us/sample - loss: 0.9253 - size_output_loss: 0.0037 - char0_loss: 0.0074 - char1_loss: 0.0059 - char2_loss: 0.0135 - char3_loss: 0.0268 - char4_loss: 0.0208 - char5_loss: 0.0332 - char6_loss: 0.0297 - char7_loss: 0.0434 - char8_loss: 0.1215 - char9_loss: 0.3859 - char10_loss: 0.2323 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 0.9986 - char3_accuracy: 0.9965 - char4_accuracy: 0.9978 - char5_accuracy: 0.9952 - char6_accuracy: 0.9974 - char7_accuracy: 0.9985 - char8_accuracy: 0.9775 - char9_accuracy: 0.8678 - char10_accuracy: 0.9249 - val_loss: 0.6067 - val_size_output_loss: 0.0028 - val_char0_loss: 0.0048 - val_char1_loss: 0.0036 - val_char2_loss: 0.0045 - val_char3_loss: 0.0069 - val_char4_loss: 0.0062 - val_char5_loss: 0.0084 - val_char6_loss: 0.0107 - val_char7_loss: 0.0165 - val_char8_loss: 0.0469 - val_char9_loss: 0.3022 - val_char10_loss: 0.1939 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 0.9988 - val_char9_accuracy: 0.9016 - val_char10_accuracy: 0.9324
    Epoch 5/150
    47360/47500 [============================>.] - ETA: 0s - loss: 0.4789 - size_output_loss: 0.0019 - char0_loss: 0.0031 - char1_loss: 0.0023 - char2_loss: 0.0027 - char3_loss: 0.0041 - char4_loss: 0.0039 - char5_loss: 0.0048 - char6_loss: 0.0062 - char7_loss: 0.0093 - char8_loss: 0.0243 - char9_loss: 0.2282 - char10_loss: 0.1882 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 0.9999 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 0.9998 - char9_accuracy: 0.9308 - char10_accuracy: 0.9329
    Epoch 00005: val_loss improved from 0.60665 to 0.34777, saving model to qr-ffnn.h5
    47500/47500 [==============================] - 18s 389us/sample - loss: 0.4787 - size_output_loss: 0.0019 - char0_loss: 0.0031 - char1_loss: 0.0023 - char2_loss: 0.0027 - char3_loss: 0.0041 - char4_loss: 0.0039 - char5_loss: 0.0048 - char6_loss: 0.0062 - char7_loss: 0.0093 - char8_loss: 0.0242 - char9_loss: 0.2277 - char10_loss: 0.1894 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 0.9999 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 0.9998 - char9_accuracy: 0.9310 - char10_accuracy: 0.9328 - val_loss: 0.3478 - val_size_output_loss: 0.0016 - val_char0_loss: 0.0024 - val_char1_loss: 0.0018 - val_char2_loss: 0.0020 - val_char3_loss: 0.0031 - val_char4_loss: 0.0027 - val_char5_loss: 0.0033 - val_char6_loss: 0.0042 - val_char7_loss: 0.0067 - val_char8_loss: 0.0127 - val_char9_loss: 0.1374 - val_char10_loss: 0.1719 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 0.9656 - val_char10_accuracy: 0.9324
    Epoch 6/150
    47360/47500 [============================>.] - ETA: 0s - loss: 0.2586 - size_output_loss: 0.0011 - char0_loss: 0.0019 - char1_loss: 0.0013 - char2_loss: 0.0015 - char3_loss: 0.0023 - char4_loss: 0.0019 - char5_loss: 0.0025 - char6_loss: 0.0031 - char7_loss: 0.0043 - char8_loss: 0.0084 - char9_loss: 0.0642 - char10_loss: 0.1661 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 0.9916 - char10_accuracy: 0.9358
    Epoch 00006: val_loss improved from 0.34777 to 0.21970, saving model to qr-ffnn.h5
    47500/47500 [==============================] - 18s 376us/sample - loss: 0.2586 - size_output_loss: 0.0011 - char0_loss: 0.0019 - char1_loss: 0.0013 - char2_loss: 0.0015 - char3_loss: 0.0023 - char4_loss: 0.0019 - char5_loss: 0.0025 - char6_loss: 0.0031 - char7_loss: 0.0043 - char8_loss: 0.0084 - char9_loss: 0.0640 - char10_loss: 0.1669 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 0.9916 - char10_accuracy: 0.9358 - val_loss: 0.2197 - val_size_output_loss: 0.0010 - val_char0_loss: 0.0014 - val_char1_loss: 0.0012 - val_char2_loss: 0.0013 - val_char3_loss: 0.0019 - val_char4_loss: 0.0017 - val_char5_loss: 0.0021 - val_char6_loss: 0.0025 - val_char7_loss: 0.0036 - val_char8_loss: 0.0063 - val_char9_loss: 0.0358 - val_char10_loss: 0.1614 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 0.9940 - val_char10_accuracy: 0.9376
    Epoch 7/150
    47360/47500 [============================>.] - ETA: 0s - loss: 0.1838 - size_output_loss: 7.2968e-04 - char0_loss: 0.0011 - char1_loss: 8.6899e-04 - char2_loss: 9.9524e-04 - char3_loss: 0.0015 - char4_loss: 0.0013 - char5_loss: 0.0016 - char6_loss: 0.0020 - char7_loss: 0.0025 - char8_loss: 0.0052 - char9_loss: 0.0163 - char10_loss: 0.1498 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 0.9998 - char9_accuracy: 0.9998 - char10_accuracy: 0.9416
    Epoch 00007: val_loss improved from 0.21970 to 0.18217, saving model to qr-ffnn.h5
    47500/47500 [==============================] - 20s 425us/sample - loss: 0.1838 - size_output_loss: 7.2925e-04 - char0_loss: 0.0011 - char1_loss: 8.6857e-04 - char2_loss: 9.9322e-04 - char3_loss: 0.0015 - char4_loss: 0.0013 - char5_loss: 0.0016 - char6_loss: 0.0020 - char7_loss: 0.0025 - char8_loss: 0.0051 - char9_loss: 0.0163 - char10_loss: 0.1498 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 0.9998 - char9_accuracy: 0.9998 - char10_accuracy: 0.9416 - val_loss: 0.1822 - val_size_output_loss: 6.9147e-04 - val_char0_loss: 9.6547e-04 - val_char1_loss: 8.0711e-04 - val_char2_loss: 8.6606e-04 - val_char3_loss: 0.0014 - val_char4_loss: 0.0012 - val_char5_loss: 0.0015 - val_char6_loss: 0.0017 - val_char7_loss: 0.0023 - val_char8_loss: 0.0036 - val_char9_loss: 0.0110 - val_char10_loss: 0.1568 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9364
    Epoch 8/150
    47360/47500 [============================>.] - ETA: 0s - loss: 0.1475 - size_output_loss: 5.0269e-04 - char0_loss: 7.9194e-04 - char1_loss: 6.1929e-04 - char2_loss: 7.1800e-04 - char3_loss: 0.0011 - char4_loss: 9.1215e-04 - char5_loss: 0.0011 - char6_loss: 0.0014 - char7_loss: 0.0017 - char8_loss: 0.0025 - char9_loss: 0.0065 - char10_loss: 0.1297 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9496
    Epoch 00008: val_loss improved from 0.18217 to 0.12832, saving model to qr-ffnn.h5
    47500/47500 [==============================] - 19s 392us/sample - loss: 0.1477 - size_output_loss: 5.0316e-04 - char0_loss: 7.9083e-04 - char1_loss: 6.1919e-04 - char2_loss: 7.1713e-04 - char3_loss: 0.0011 - char4_loss: 9.1236e-04 - char5_loss: 0.0012 - char6_loss: 0.0014 - char7_loss: 0.0017 - char8_loss: 0.0025 - char9_loss: 0.0065 - char10_loss: 0.1309 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9495 - val_loss: 0.1283 - val_size_output_loss: 5.1543e-04 - val_char0_loss: 6.8775e-04 - val_char1_loss: 5.8515e-04 - val_char2_loss: 6.3153e-04 - val_char3_loss: 0.0010 - val_char4_loss: 8.6108e-04 - val_char5_loss: 0.0010 - val_char6_loss: 0.0012 - val_char7_loss: 0.0016 - val_char8_loss: 0.0023 - val_char9_loss: 0.0057 - val_char10_loss: 0.1122 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 0.9996 - val_char10_accuracy: 0.9580
    Epoch 9/150
    47488/47500 [============================>.] - ETA: 0s - loss: 0.2613 - size_output_loss: 3.5372e-04 - char0_loss: 0.0017 - char1_loss: 0.0749 - char2_loss: 0.0021 - char3_loss: 0.0033 - char4_loss: 0.0477 - char5_loss: 0.0021 - char6_loss: 0.0127 - char7_loss: 0.0035 - char8_loss: 0.0039 - char9_loss: 0.0074 - char10_loss: 0.1017 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 0.9811 - char2_accuracy: 1.0000 - char3_accuracy: 0.9998 - char4_accuracy: 0.9881 - char5_accuracy: 1.0000 - char6_accuracy: 0.9973 - char7_accuracy: 0.9999 - char8_accuracy: 0.9999 - char9_accuracy: 0.9998 - char10_accuracy: 0.9629
    Epoch 00009: val_loss did not improve from 0.12832
    47500/47500 [==============================] - 18s 378us/sample - loss: 0.2613 - size_output_loss: 3.5432e-04 - char0_loss: 0.0017 - char1_loss: 0.0747 - char2_loss: 0.0021 - char3_loss: 0.0033 - char4_loss: 0.0476 - char5_loss: 0.0021 - char6_loss: 0.0127 - char7_loss: 0.0035 - char8_loss: 0.0039 - char9_loss: 0.0074 - char10_loss: 0.1020 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 0.9811 - char2_accuracy: 1.0000 - char3_accuracy: 0.9998 - char4_accuracy: 0.9881 - char5_accuracy: 1.0000 - char6_accuracy: 0.9973 - char7_accuracy: 0.9999 - char8_accuracy: 0.9999 - char9_accuracy: 0.9998 - char10_accuracy: 0.9629 - val_loss: 0.1594 - val_size_output_loss: 3.8849e-04 - val_char0_loss: 0.0012 - val_char1_loss: 0.0294 - val_char2_loss: 0.0012 - val_char3_loss: 0.0016 - val_char4_loss: 0.0034 - val_char5_loss: 0.0017 - val_char6_loss: 0.0026 - val_char7_loss: 0.0024 - val_char8_loss: 0.0035 - val_char9_loss: 0.0066 - val_char10_loss: 0.1064 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 0.9944 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9544
    Epoch 10/150
    47488/47500 [============================>.] - ETA: 0s - loss: 0.0849 - size_output_loss: 2.5821e-04 - char0_loss: 5.5144e-04 - char1_loss: 0.0050 - char2_loss: 5.1768e-04 - char3_loss: 7.6070e-04 - char4_loss: 0.0015 - char5_loss: 8.1777e-04 - char6_loss: 0.0011 - char7_loss: 0.0011 - char8_loss: 0.0015 - char9_loss: 0.0027 - char10_loss: 0.0689 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 0.9997 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9757
    Epoch 00010: val_loss improved from 0.12832 to 0.07525, saving model to qr-ffnn.h5
    47500/47500 [==============================] - 18s 385us/sample - loss: 0.0848 - size_output_loss: 2.5828e-04 - char0_loss: 5.5095e-04 - char1_loss: 0.0050 - char2_loss: 5.1739e-04 - char3_loss: 7.5990e-04 - char4_loss: 0.0015 - char5_loss: 8.1665e-04 - char6_loss: 0.0011 - char7_loss: 0.0011 - char8_loss: 0.0015 - char9_loss: 0.0027 - char10_loss: 0.0688 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 0.9997 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9757 - val_loss: 0.0753 - val_size_output_loss: 2.8417e-04 - val_char0_loss: 3.9920e-04 - val_char1_loss: 0.0016 - val_char2_loss: 3.5656e-04 - val_char3_loss: 5.6323e-04 - val_char4_loss: 9.5683e-04 - val_char5_loss: 6.2435e-04 - val_char6_loss: 7.2700e-04 - val_char7_loss: 8.1555e-04 - val_char8_loss: 0.0012 - val_char9_loss: 0.0023 - val_char10_loss: 0.0654 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9752
    Epoch 11/150
    47360/47500 [============================>.] - ETA: 0s - loss: 0.0532 - size_output_loss: 1.9347e-04 - char0_loss: 3.1114e-04 - char1_loss: 0.0010 - char2_loss: 2.7743e-04 - char3_loss: 4.2833e-04 - char4_loss: 6.9072e-04 - char5_loss: 4.7119e-04 - char6_loss: 5.6256e-04 - char7_loss: 5.8764e-04 - char8_loss: 8.2005e-04 - char9_loss: 0.0014 - char10_loss: 0.0464 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9852
    Epoch 00011: val_loss improved from 0.07525 to 0.05058, saving model to qr-ffnn.h5
    47500/47500 [==============================] - 20s 418us/sample - loss: 0.0531 - size_output_loss: 1.9362e-04 - char0_loss: 3.1099e-04 - char1_loss: 0.0010 - char2_loss: 2.7716e-04 - char3_loss: 4.2829e-04 - char4_loss: 6.8995e-04 - char5_loss: 4.7053e-04 - char6_loss: 5.6206e-04 - char7_loss: 5.8715e-04 - char8_loss: 8.2029e-04 - char9_loss: 0.0014 - char10_loss: 0.0463 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9852 - val_loss: 0.0506 - val_size_output_loss: 2.2308e-04 - val_char0_loss: 2.8065e-04 - val_char1_loss: 8.1193e-04 - val_char2_loss: 2.5886e-04 - val_char3_loss: 4.1125e-04 - val_char4_loss: 5.9437e-04 - val_char5_loss: 4.5681e-04 - val_char6_loss: 5.0909e-04 - val_char7_loss: 5.7975e-04 - val_char8_loss: 7.8452e-04 - val_char9_loss: 0.0015 - val_char10_loss: 0.0453 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9840
    Epoch 12/150
    47488/47500 [============================>.] - ETA: 0s - loss: 0.0288 - size_output_loss: 1.4592e-04 - char0_loss: 2.2540e-04 - char1_loss: 5.7302e-04 - char2_loss: 2.0683e-04 - char3_loss: 3.1565e-04 - char4_loss: 4.6149e-04 - char5_loss: 3.5011e-04 - char6_loss: 3.9836e-04 - char7_loss: 4.2194e-04 - char8_loss: 5.7277e-04 - char9_loss: 9.6940e-04 - char10_loss: 0.0242 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9948
    Epoch 00012: val_loss improved from 0.05058 to 0.02853, saving model to qr-ffnn.h5
    47500/47500 [==============================] - 19s 394us/sample - loss: 0.0288 - size_output_loss: 1.4574e-04 - char0_loss: 2.2572e-04 - char1_loss: 5.7268e-04 - char2_loss: 2.0697e-04 - char3_loss: 3.1578e-04 - char4_loss: 4.6064e-04 - char5_loss: 3.4989e-04 - char6_loss: 3.9839e-04 - char7_loss: 4.2193e-04 - char8_loss: 5.7318e-04 - char9_loss: 9.6965e-04 - char10_loss: 0.0242 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9948 - val_loss: 0.0285 - val_size_output_loss: 1.8114e-04 - val_char0_loss: 2.1768e-04 - val_char1_loss: 5.1469e-04 - val_char2_loss: 1.9223e-04 - val_char3_loss: 3.1858e-04 - val_char4_loss: 4.3662e-04 - val_char5_loss: 3.4089e-04 - val_char6_loss: 3.6400e-04 - val_char7_loss: 4.3446e-04 - val_char8_loss: 5.8917e-04 - val_char9_loss: 0.0011 - val_char10_loss: 0.0243 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9956
    Epoch 13/150
    47360/47500 [============================>.] - ETA: 0s - loss: 0.0143 - size_output_loss: 1.1237e-04 - char0_loss: 1.7184e-04 - char1_loss: 3.7387e-04 - char2_loss: 1.5816e-04 - char3_loss: 2.4104e-04 - char4_loss: 3.3615e-04 - char5_loss: 2.6873e-04 - char6_loss: 2.9636e-04 - char7_loss: 3.1630e-04 - char8_loss: 4.2027e-04 - char9_loss: 6.9542e-04 - char10_loss: 0.0109 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9992
    Epoch 00013: val_loss improved from 0.02853 to 0.01458, saving model to qr-ffnn.h5
    47500/47500 [==============================] - 19s 394us/sample - loss: 0.0143 - size_output_loss: 1.1240e-04 - char0_loss: 1.7176e-04 - char1_loss: 3.7336e-04 - char2_loss: 1.5797e-04 - char3_loss: 2.4059e-04 - char4_loss: 3.3639e-04 - char5_loss: 2.6828e-04 - char6_loss: 2.9635e-04 - char7_loss: 3.1594e-04 - char8_loss: 4.1959e-04 - char9_loss: 6.9401e-04 - char10_loss: 0.0109 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9992 - val_loss: 0.0146 - val_size_output_loss: 1.4119e-04 - val_char0_loss: 1.5920e-04 - val_char1_loss: 3.4460e-04 - val_char2_loss: 1.5175e-04 - val_char3_loss: 2.4433e-04 - val_char4_loss: 3.1848e-04 - val_char5_loss: 2.6679e-04 - val_char6_loss: 2.7103e-04 - val_char7_loss: 3.1627e-04 - val_char8_loss: 4.3955e-04 - val_char9_loss: 7.5110e-04 - val_char10_loss: 0.0112 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9984
    Epoch 14/150
    47360/47500 [============================>.] - ETA: 0s - loss: 0.0084 - size_output_loss: 8.7072e-05 - char0_loss: 1.3266e-04 - char1_loss: 2.6485e-04 - char2_loss: 1.2470e-04 - char3_loss: 1.8794e-04 - char4_loss: 2.5394e-04 - char5_loss: 2.0860e-04 - char6_loss: 2.2881e-04 - char7_loss: 2.4148e-04 - char8_loss: 3.1327e-04 - char9_loss: 5.0952e-04 - char10_loss: 0.0058 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9996
    Epoch 00014: val_loss improved from 0.01458 to 0.00813, saving model to qr-ffnn.h5
    47500/47500 [==============================] - 20s 430us/sample - loss: 0.0084 - size_output_loss: 8.6886e-05 - char0_loss: 1.3258e-04 - char1_loss: 2.6490e-04 - char2_loss: 1.2469e-04 - char3_loss: 1.8796e-04 - char4_loss: 2.5428e-04 - char5_loss: 2.0862e-04 - char6_loss: 2.2928e-04 - char7_loss: 2.4151e-04 - char8_loss: 3.1347e-04 - char9_loss: 5.1030e-04 - char10_loss: 0.0058 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 0.9996 - val_loss: 0.0081 - val_size_output_loss: 1.1082e-04 - val_char0_loss: 1.2932e-04 - val_char1_loss: 2.5388e-04 - val_char2_loss: 1.1886e-04 - val_char3_loss: 1.8786e-04 - val_char4_loss: 2.4588e-04 - val_char5_loss: 2.0810e-04 - val_char6_loss: 2.1225e-04 - val_char7_loss: 2.4255e-04 - val_char8_loss: 3.3675e-04 - val_char9_loss: 5.7442e-04 - val_char10_loss: 0.0055 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9996
    Epoch 15/150
    47488/47500 [============================>.] - ETA: 0s - loss: 0.0049 - size_output_loss: 6.8185e-05 - char0_loss: 1.0443e-04 - char1_loss: 1.9687e-04 - char2_loss: 9.8727e-05 - char3_loss: 1.4853e-04 - char4_loss: 1.9962e-04 - char5_loss: 1.6431e-04 - char6_loss: 1.7864e-04 - char7_loss: 1.8697e-04 - char8_loss: 2.4028e-04 - char9_loss: 3.7878e-04 - char10_loss: 0.0029 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000
    Epoch 00015: val_loss improved from 0.00813 to 0.00621, saving model to qr-ffnn.h5
    47500/47500 [==============================] - 18s 383us/sample - loss: 0.0049 - size_output_loss: 6.8140e-05 - char0_loss: 1.0446e-04 - char1_loss: 1.9671e-04 - char2_loss: 9.8801e-05 - char3_loss: 1.4837e-04 - char4_loss: 1.9947e-04 - char5_loss: 1.6425e-04 - char6_loss: 1.7844e-04 - char7_loss: 1.8664e-04 - char8_loss: 2.3968e-04 - char9_loss: 3.7848e-04 - char10_loss: 0.0029 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 0.0062 - val_size_output_loss: 8.9245e-05 - val_char0_loss: 1.0282e-04 - val_char1_loss: 1.9213e-04 - val_char2_loss: 9.6472e-05 - val_char3_loss: 1.5064e-04 - val_char4_loss: 1.9825e-04 - val_char5_loss: 1.6346e-04 - val_char6_loss: 1.6704e-04 - val_char7_loss: 1.9202e-04 - val_char8_loss: 2.7322e-04 - val_char9_loss: 4.4535e-04 - val_char10_loss: 0.0042 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 0.9996
    Epoch 16/150
    47488/47500 [============================>.] - ETA: 0s - loss: 0.0034 - size_output_loss: 5.3775e-05 - char0_loss: 8.2949e-05 - char1_loss: 1.5128e-04 - char2_loss: 7.9768e-05 - char3_loss: 1.1876e-04 - char4_loss: 1.6111e-04 - char5_loss: 1.3021e-04 - char6_loss: 1.4135e-04 - char7_loss: 1.4750e-04 - char8_loss: 1.8591e-04 - char9_loss: 2.8600e-04 - char10_loss: 0.0019 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000
    Epoch 00016: val_loss improved from 0.00621 to 0.00460, saving model to qr-ffnn.h5
    47500/47500 [==============================] - 19s 399us/sample - loss: 0.0034 - size_output_loss: 5.3910e-05 - char0_loss: 8.2890e-05 - char1_loss: 1.5116e-04 - char2_loss: 7.9708e-05 - char3_loss: 1.1872e-04 - char4_loss: 1.6086e-04 - char5_loss: 1.3016e-04 - char6_loss: 1.4127e-04 - char7_loss: 1.4767e-04 - char8_loss: 1.8563e-04 - char9_loss: 2.8669e-04 - char10_loss: 0.0019 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 0.0046 - val_size_output_loss: 7.5378e-05 - val_char0_loss: 7.9404e-05 - val_char1_loss: 1.4881e-04 - val_char2_loss: 7.6781e-05 - val_char3_loss: 1.2176e-04 - val_char4_loss: 1.6056e-04 - val_char5_loss: 1.3289e-04 - val_char6_loss: 1.3759e-04 - val_char7_loss: 1.5048e-04 - val_char8_loss: 2.1139e-04 - val_char9_loss: 3.4907e-04 - val_char10_loss: 0.0029 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 1.0000 - val_char10_accuracy: 1.0000
    Epoch 17/150
    47488/47500 [============================>.] - ETA: 0s - loss: 0.2155 - size_output_loss: 4.2650e-05 - char0_loss: 0.0173 - char1_loss: 0.0047 - char2_loss: 0.0098 - char3_loss: 0.0225 - char4_loss: 0.0025 - char5_loss: 0.0064 - char6_loss: 0.0064 - char7_loss: 0.0640 - char8_loss: 0.0378 - char9_loss: 0.0324 - char10_loss: 0.0117 - size_output_accuracy: 1.0000 - char0_accuracy: 0.9943 - char1_accuracy: 0.9985 - char2_accuracy: 0.9963 - char3_accuracy: 0.9938 - char4_accuracy: 0.9996 - char5_accuracy: 0.9981 - char6_accuracy: 0.9983 - char7_accuracy: 0.9896 - char8_accuracy: 0.9926 - char9_accuracy: 0.9936 - char10_accuracy: 0.9965                   
    Epoch 00017: val_loss did not improve from 0.00460
    47500/47500 [==============================] - 21s 436us/sample - loss: 0.2156 - size_output_loss: 4.2661e-05 - char0_loss: 0.0181 - char1_loss: 0.0048 - char2_loss: 0.0099 - char3_loss: 0.0226 - char4_loss: 0.0025 - char5_loss: 0.0064 - char6_loss: 0.0064 - char7_loss: 0.0639 - char8_loss: 0.0379 - char9_loss: 0.0328 - char10_loss: 0.0116 - size_output_accuracy: 1.0000 - char0_accuracy: 0.9943 - char1_accuracy: 0.9985 - char2_accuracy: 0.9963 - char3_accuracy: 0.9938 - char4_accuracy: 0.9996 - char5_accuracy: 0.9981 - char6_accuracy: 0.9983 - char7_accuracy: 0.9896 - char8_accuracy: 0.9926 - char9_accuracy: 0.9936 - char10_accuracy: 0.9965 - val_loss: 0.8814 - val_size_output_loss: 5.7654e-05 - val_char0_loss: 0.3101 - val_char1_loss: 0.0257 - val_char2_loss: 0.0670 - val_char3_loss: 0.1780 - val_char4_loss: 0.0123 - val_char5_loss: 0.0678 - val_char6_loss: 0.0203 - val_char7_loss: 0.0537 - val_char8_loss: 0.0475 - val_char9_loss: 0.0651 - val_char10_loss: 0.0344 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 0.8884 - val_char1_accuracy: 0.9956 - val_char2_accuracy: 0.9792 - val_char3_accuracy: 0.9352 - val_char4_accuracy: 0.9980 - val_char5_accuracy: 0.9776 - val_char6_accuracy: 0.9956 - val_char7_accuracy: 0.9848 - val_char8_accuracy: 0.9872 - val_char9_accuracy: 0.9804 - val_char10_accuracy: 0.9924
    Epoch 18/150
    47360/47500 [============================>.] - ETA: 0s - loss: 0.0366 - size_output_loss: 3.3803e-05 - char0_loss: 0.0053 - char1_loss: 0.0013 - char2_loss: 0.0030 - char3_loss: 0.0052 - char4_loss: 9.7082e-04 - char5_loss: 0.0018 - char6_loss: 0.0011 - char7_loss: 0.0036 - char8_loss: 0.0037 - char9_loss: 0.0049 - char10_loss: 0.0057 - size_output_accuracy: 1.0000 - char0_accuracy: 0.9986 - char1_accuracy: 0.9999 - char2_accuracy: 0.9995 - char3_accuracy: 0.9986 - char4_accuracy: 1.0000 - char5_accuracy: 0.9998 - char6_accuracy: 1.0000 - char7_accuracy: 0.9997 - char8_accuracy: 0.9996 - char9_accuracy: 0.9994 - char10_accuracy: 0.9995
    Epoch 00018: val_loss did not improve from 0.00460
    47500/47500 [==============================] - 19s 409us/sample - loss: 0.0365 - size_output_loss: 3.3719e-05 - char0_loss: 0.0053 - char1_loss: 0.0013 - char2_loss: 0.0029 - char3_loss: 0.0052 - char4_loss: 9.6650e-04 - char5_loss: 0.0018 - char6_loss: 0.0011 - char7_loss: 0.0036 - char8_loss: 0.0037 - char9_loss: 0.0049 - char10_loss: 0.0056 - size_output_accuracy: 1.0000 - char0_accuracy: 0.9986 - char1_accuracy: 0.9999 - char2_accuracy: 0.9995 - char3_accuracy: 0.9986 - char4_accuracy: 1.0000 - char5_accuracy: 0.9998 - char6_accuracy: 1.0000 - char7_accuracy: 0.9997 - char8_accuracy: 0.9996 - char9_accuracy: 0.9994 - char10_accuracy: 0.9995 - val_loss: 0.0087 - val_size_output_loss: 5.0393e-05 - val_char0_loss: 3.1363e-04 - val_char1_loss: 2.0042e-04 - val_char2_loss: 2.0277e-04 - val_char3_loss: 3.2096e-04 - val_char4_loss: 2.1315e-04 - val_char5_loss: 2.3959e-04 - val_char6_loss: 2.4534e-04 - val_char7_loss: 0.0011 - val_char8_loss: 9.3877e-04 - val_char9_loss: 0.0017 - val_char10_loss: 0.0032 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 0.9996 - val_char10_accuracy: 1.0000
    Epoch 19/150
    47488/47500 [============================>.] - ETA: 0s - loss: 0.0049 - size_output_loss: 2.6901e-05 - char0_loss: 2.1235e-04 - char1_loss: 1.3347e-04 - char2_loss: 1.4847e-04 - char3_loss: 2.2903e-04 - char4_loss: 1.5532e-04 - char5_loss: 1.5430e-04 - char6_loss: 1.7252e-04 - char7_loss: 7.3921e-04 - char8_loss: 6.4731e-04 - char9_loss: 8.1930e-04 - char10_loss: 0.0014 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000
    Epoch 00019: val_loss did not improve from 0.00460
    47500/47500 [==============================] - 19s 392us/sample - loss: 0.0049 - size_output_loss: 2.6949e-05 - char0_loss: 2.1213e-04 - char1_loss: 1.3332e-04 - char2_loss: 1.4840e-04 - char3_loss: 2.2962e-04 - char4_loss: 1.5513e-04 - char5_loss: 1.5414e-04 - char6_loss: 1.7281e-04 - char7_loss: 7.3804e-04 - char8_loss: 6.4920e-04 - char9_loss: 8.1917e-04 - char10_loss: 0.0014 - size_output_accuracy: 1.0000 - char0_accuracy: 1.0000 - char1_accuracy: 1.0000 - char2_accuracy: 1.0000 - char3_accuracy: 1.0000 - char4_accuracy: 1.0000 - char5_accuracy: 1.0000 - char6_accuracy: 1.0000 - char7_accuracy: 1.0000 - char8_accuracy: 1.0000 - char9_accuracy: 1.0000 - char10_accuracy: 1.0000 - val_loss: 0.0054 - val_size_output_loss: 4.0515e-05 - val_char0_loss: 1.8744e-04 - val_char1_loss: 1.1478e-04 - val_char2_loss: 1.2041e-04 - val_char3_loss: 1.9583e-04 - val_char4_loss: 1.3104e-04 - val_char5_loss: 1.4401e-04 - val_char6_loss: 1.4391e-04 - val_char7_loss: 6.3192e-04 - val_char8_loss: 5.2981e-04 - val_char9_loss: 0.0012 - val_char10_loss: 0.0019 - val_size_output_accuracy: 1.0000 - val_char0_accuracy: 1.0000 - val_char1_accuracy: 1.0000 - val_char2_accuracy: 1.0000 - val_char3_accuracy: 1.0000 - val_char4_accuracy: 1.0000 - val_char5_accuracy: 1.0000 - val_char6_accuracy: 1.0000 - val_char7_accuracy: 1.0000 - val_char8_accuracy: 1.0000 - val_char9_accuracy: 0.9996 - val_char10_accuracy: 1.0000



{% highlight python %}
# load best model
best_model = keras.models.load_model('qr-ffnn.h5')
{% endhighlight %}


{% highlight python %}
# try on most popular words
predictions = best_model.predict(words11_data)

def softmax(values):
    exp = np.exp(values - np.max(values))
    return exp / exp.sum()

errors = 0
conf = 1
for i in range(len(words11)):
    expected_word = words11[i].ljust(11, EOI)
    predicted_chars = []
    for k in range(1, 12):
        largest_index = np.argmax(predictions[k][i], axis=0)
        c = ALL_CHAR_CLASSES[largest_index]
        predicted_chars.append(c)
    predicted_word = ''.join(predicted_chars)
    if expected_word != predicted_word:
        errors += 1
        print(i)
        print(expected_word)
        print(predicted_word)
print("Total errors:", errors, "out of", len(words11))
{% endhighlight %}

    48
    information
    informatiof
    5836
    information
    informatiof
    9112
    workstation
    workstatiof
    Total errors: 3 out of 9894


3 errors out of almost 10k - this is a pretty good solution.

# Try recurrent neural network (RNN)

QR code may be seen as a stream of characters. Every next one may depend on previous output. It makes sense to try RNN and see how it will behave.


{% highlight python %}
# simple RNN with 3 LSTM layers. Output is 11 classes.
def make_rnn():

    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(21,21)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.RepeatVector(11))
    model.add(keras.layers.LSTM(21*21, return_sequences=True))
    model.add(keras.layers.LSTM(21*21, return_sequences=True))
    model.add(keras.layers.LSTM(21, return_sequences=True))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(len(ALL_CHAR_CLASSES))))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

rnn = make_rnn()
rnn.summary()
{% endhighlight %}

    Model: "sequential_72"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_17 (Flatten)         (None, 441)               0         
    _________________________________________________________________
    repeat_vector (RepeatVector) (None, 11, 441)           0         
    _________________________________________________________________
    lstm (LSTM)                  (None, 11, 441)           1557612   
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 11, 441)           1557612   
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 11, 21)            38892     
    _________________________________________________________________
    time_distributed (TimeDistri (None, 11, 28)            616       
    =================================================================
    Total params: 3,154,732
    Trainable params: 3,154,732
    Non-trainable params: 0
    _________________________________________________________________



{% highlight python %}
# create labels for RNN
# labels must have 11 classes, one per each position
rnn_labels = []
for label in training_labels:
    label_matrix = []
    for i in range(11):
        c = label[i] if len(label) > i else EOI
        label_matrix.append([ALL_CHAR_CLASSES.index(c)])
    rnn_labels.append(label_matrix)
rnn_labels = np.asarray(rnn_labels)
{% endhighlight %}


{% highlight python %}
#train_data, train_labels
rnn_labels.shape
{% endhighlight %}




    (50000, 11, 1)




{% highlight python %}
rnn_hist = rnn.fit(
        training_data,
        rnn_labels,
        epochs=150,
        batch_size=128,
        validation_split=0.05,
        callbacks=[EarlyStopping(monitor='val_loss' , min_delta=0.0001, patience=3)]
    )
{% endhighlight %}

    Train on 47500 samples, validate on 2500 samples
    Epoch 1/150
    47500/47500 [==============================] - 150s 3ms/sample - loss: 1.4395 - accuracy: 0.5210 - val_loss: 1.3092 - val_accuracy: 0.5425
    Epoch 2/150
    47500/47500 [==============================] - 165s 3ms/sample - loss: 1.3053 - accuracy: 0.5395 - val_loss: 1.1978 - val_accuracy: 0.5624
    Epoch 3/150
    47500/47500 [==============================] - 141s 3ms/sample - loss: 1.1711 - accuracy: 0.5690 - val_loss: 1.0711 - val_accuracy: 0.6024
    Epoch 4/150
    47500/47500 [==============================] - 139s 3ms/sample - loss: 1.0013 - accuracy: 0.6194 - val_loss: 0.8846 - val_accuracy: 0.6553
    Epoch 5/150
    47500/47500 [==============================] - 142s 3ms/sample - loss: 0.8064 - accuracy: 0.6732 - val_loss: 0.7353 - val_accuracy: 0.6890
    Epoch 6/150
    47500/47500 [==============================] - 142s 3ms/sample - loss: 0.6457 - accuracy: 0.7094 - val_loss: 0.5541 - val_accuracy: 0.7342
    Epoch 7/150
    47500/47500 [==============================] - 151s 3ms/sample - loss: 0.5126 - accuracy: 0.7513 - val_loss: 0.4660 - val_accuracy: 0.7577
    Epoch 8/150
    47500/47500 [==============================] - 138s 3ms/sample - loss: 0.4563 - accuracy: 0.7707 - val_loss: 0.4098 - val_accuracy: 0.7850
    Epoch 9/150
    47500/47500 [==============================] - 137s 3ms/sample - loss: 0.3878 - accuracy: 0.8008 - val_loss: 0.3573 - val_accuracy: 0.8161
    Epoch 10/150
    47500/47500 [==============================] - 143s 3ms/sample - loss: 0.3444 - accuracy: 0.8325 - val_loss: 0.3185 - val_accuracy: 0.8474
    Epoch 11/150
    47500/47500 [==============================] - 142s 3ms/sample - loss: 0.2688 - accuracy: 0.8833 - val_loss: 0.2188 - val_accuracy: 0.9126
    Epoch 12/150
    47500/47500 [==============================] - 143s 3ms/sample - loss: 0.1644 - accuracy: 0.9452 - val_loss: 0.1264 - val_accuracy: 0.9643
    Epoch 13/150
    47500/47500 [==============================] - 147s 3ms/sample - loss: 0.1334 - accuracy: 0.9634 - val_loss: 0.0785 - val_accuracy: 0.9795
    Epoch 14/150
    47500/47500 [==============================] - 141s 3ms/sample - loss: 0.0653 - accuracy: 0.9832 - val_loss: 0.0574 - val_accuracy: 0.9848
    Epoch 15/150
    47500/47500 [==============================] - 141s 3ms/sample - loss: 0.0461 - accuracy: 0.9889 - val_loss: 0.0565 - val_accuracy: 0.9857
    Epoch 16/150
    47500/47500 [==============================] - 142s 3ms/sample - loss: 0.0475 - accuracy: 0.9888 - val_loss: 0.0301 - val_accuracy: 0.9932
    Epoch 17/150
    47500/47500 [==============================] - 143s 3ms/sample - loss: 0.0230 - accuracy: 0.9951 - val_loss: 0.0207 - val_accuracy: 0.9953
    Epoch 18/150
    47500/47500 [==============================] - 140s 3ms/sample - loss: 0.0299 - accuracy: 0.9927 - val_loss: 0.0526 - val_accuracy: 0.9858
    Epoch 19/150
    47500/47500 [==============================] - 137s 3ms/sample - loss: 0.0244 - accuracy: 0.9942 - val_loss: 0.0163 - val_accuracy: 0.9960
    Epoch 20/150
    47500/47500 [==============================] - 135s 3ms/sample - loss: 0.0121 - accuracy: 0.9972 - val_loss: 0.0165 - val_accuracy: 0.9955
    Epoch 21/150
    47500/47500 [==============================] - 141s 3ms/sample - loss: 0.0103 - accuracy: 0.9974 - val_loss: 0.0092 - val_accuracy: 0.9975
    Epoch 22/150
    47500/47500 [==============================] - 138s 3ms/sample - loss: 0.0414 - accuracy: 0.9893 - val_loss: 0.0169 - val_accuracy: 0.9955
    Epoch 23/150
    47500/47500 [==============================] - 138s 3ms/sample - loss: 0.0110 - accuracy: 0.9976 - val_loss: 0.0082 - val_accuracy: 0.9978
    Epoch 24/150
    47500/47500 [==============================] - 136s 3ms/sample - loss: 0.0075 - accuracy: 0.9985 - val_loss: 0.0096 - val_accuracy: 0.9975
    Epoch 25/150
    47500/47500 [==============================] - 140s 3ms/sample - loss: 0.0149 - accuracy: 0.9965 - val_loss: 0.0207 - val_accuracy: 0.9953
    Epoch 26/150
    47500/47500 [==============================] - 139s 3ms/sample - loss: 0.0120 - accuracy: 0.9975 - val_loss: 0.0095 - val_accuracy: 0.9981


Final accuracy is .9981, not as good as last FFNN, but not terribly bad either. Let's see performance on the dataset of most popular words:


{% highlight python %}
output11 = rnn.predict(words11_data, batch_size=1)
{% endhighlight %}


{% highlight python %}
# check every predicted label and count errors
def get_word_from_rnn_output(output, index):
    w = []
    for i in range(11):
        largest_index = np.argmax(output[index][i], axis=0)
        c = ALL_CHAR_CLASSES[largest_index]
        if c == EOI:
            break
        w.append(c)
    return ''.join(w)

errors = 0
error_samples = []
for i in range(len(words11)):
    actual = words11[i]
    predicted = get_word_from_rnn_output(output11, i)
    if actual != predicted:
        error_samples.append((actual, predicted))
        errors += 1
print("Errors:" , errors, errors/len(words11))
# print few samples
print(error_samples[:10])
{% endhighlight %}

    Errors: 297 0.03001819284414797
    [('these', 'thesg'), ('service', 'servkce'), ('state', 'statg'), ('system', 'systgm'), ('special', 'speckal'), ('download', 'downnoad'), ('library', 'librcry'), ('party', 'parts'), ('quote', 'quotg'), ('possible', 'posskble')]


The eventual error rate is 3% comparing to 0.03% of the best feed-forward neural network. It is much worse, but the network design is much simpler. One day I may spend more time applying an RNN to this problem.


{% highlight python %}

{% endhighlight %}
