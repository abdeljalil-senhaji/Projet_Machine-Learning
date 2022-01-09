import matplotlib.pylab as plt 
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import plot_confusion_matrix 
import numpy as np

from keras.utils import to_categorical
import keras
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler



def ANN(X, Y):
    '''
	@param X: expression genes
	@param Y: type of cancer
	'''


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=40)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)



    Y_encoded = list()
    for i in Y :
        if i =='BRCA': Y_encoded.append(0)
        if i =='PRAD': Y_encoded.append(1)
        if i =='LUAD': Y_encoded.append(2)
        if i =='KIRC': Y_encoded.append(3)
        if i =='COAD': Y_encoded.append(4)

    
    Y_bis = to_categorical(Y_encoded)


    init ='random_uniform'
    input_layer = Input(shape= (20531,))
    mid_layer = Dense(100, activation ='relu', kernel_initializer = init)(input_layer) 
    mid_layer_2 = Dense(50, activation ='relu', kernel_initializer = init)(mid_layer)
    output_layer = Dense(5, activation ='softmax', kernel_initializer = init)(mid_layer_2) 


    dropout = keras.layers.Dropout(0.2)(input_layer) 


    model = Model(inputs=input_layer, outputs=output_layer)
    X= scaler.fit_transform(X)


    model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

    History = model.fit(X, Y_bis, batch_size=64, epochs = 100, verbose = 0, validation_split=0.33, shuffle=True)
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./output/ANN/function_loss.png")
    #plt.show()
