import matplotlib.pyplot as plt
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from catdog import *
from catdog_train import cnn_model
test_data=np.load("test_data.npy")
fig = plt.figure()
model=cnn_model(IMG_SIZE)
model.load(MODEL_NAME)

for num,data in enumerate(test_data[:12]):
    img_num=data[1]
    img_data=data[0]
    y=fig.add_subplot(3,4,num+1)
    orig=img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out=model.predict([data])[0]
    if np.argmax(model_out) == 1: str_label='Dog'
    else: str_label = 'cat'
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
'''
with open('submission-file.csv','w') as f:
    f.write('id,label\n')
    for data in tqdm(test_data):
        img_num=data[1]
        img_data=data[0]
        orig=img_data
        data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
        model_out=model.predict([data])[0]
        f.write('{},{}\n'.format(img_num,model_out[1]))
'''
