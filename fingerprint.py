import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from sklearn import svm
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

def main():
    #fp_pickle()
    fp_svm()
    #fp_keras()

def fp_pickle():
    print("\nProcessamento com SVM\n")
    # dirpath = 'dataset_2009' # Excelente. Acurácia alta. Não podemos usar.
    dirpath = 'digitais/index' # Pequeno. Acurácia baixa. Podemos usar.

    filenames = os.listdir(dirpath)
    num_images = len(filenames)

    X = []
    y = []
    for fname in filenames:
        srcpath = os.path.join(dirpath, fname)
        label = os.path.basename(srcpath).split('_')[0]
        y.append(label)
        img = np.array(Image.open(srcpath).resize((312,372))) 
        img = rgb2gray(img)
        X.append(img)


    X = np.asarray([np.reshape(x, (116064, 1)) for x in X]) # 116064 = 372*312
    y = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    fingerprint_data = []
    fingerprint_data.append(X_train)
    fingerprint_data.append(X_test)
    fingerprint_data.append(y_train)
    fingerprint_data.append(y_test)

    pkl_file = open('fingerprint_data.pkl', 'wb')

    pickle.dump(fingerprint_data, pkl_file)
    pkl_file.close()

def fp_svm():
    pkl_file = open('fingerprint_data.pkl', 'rb')
    x_train, x_test, y_train, y_test = pickle.load(pkl_file)

    x_train = x_train.reshape(x_train.shape[0],372*312)
    x_test = x_test.reshape(x_test.shape[0],372*312)

    for str in ['linear', 'poly', 'rbf', 'sigmoid']:
        clf = svm.SVC(kernel=str, gamma='auto') # or gamma='scale'
        clf.fit(x_train, y_train)
        acc = clf.score(x_test, y_test)
        print('Acurácia', str, '= ', acc*100, '%')

    # Modelo com maior acurácia

    clf = svm.SVC(kernel='linear', gamma='auto') # or gamma='scale'
    clf.fit(x_train, y_train)

    # Validação cruzada kFold

    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(clf, x_train, y_train, cv=4)
    scores     

    print("Acurácia média + intervalo de confiança de 95%% : %0.2f (+/- %0.2f) %%" % (scores.mean(), scores.std() * 2))

    # Usando pré-processamento

    from sklearn import preprocessing

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train_transformed = scaler.transform(x_train)
    # clf = svm.SVC(C=1).fit(x_train_transformed, y_train)
    x_test_transformed = scaler.transform(x_test)
    # clf.score(x_test_transformed, y_test)  

    # Retreinando com os conjuntos pré-processados

    clf = svm.SVC(kernel='linear', gamma='auto') # or gamma='scale'
    clf.fit(x_train_transformed, y_train)
    acc = clf.score(x_test_transformed, y_test)
    print('Nova acurácia =' , acc*100, '%')

    scores = cross_val_score(clf, x_train_transformed, y_train, cv=4)
    scores  

    print("Nova acurácia média + intervalo de confiança de 95%% : %0.2f (+/- %0.2f) %%" % (scores.mean(), scores.std() * 2))

    # A mesma coisa utilizando pipeline

    from sklearn.pipeline import make_pipeline

    clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(kernel='linear', gamma='auto'))
    cross_val_score(clf, x_train, y_train, cv=4)

    print("Nova acurácia média + intervalo de confiança de 95%% : %0.2f (+/- %0.2f) %%" % (scores.mean(), scores.std() * 2))

    # O modelo com os melhores resultados foi o svm com kernel linear sem pré-processamento

    clf = svm.SVC(kernel='linear', gamma='auto') # or gamma='scale'
    # clf.fit(x_train, y_train)
    clf.fit(x_train_transformed, y_train)

    # Avaliação individual

    image_index = 0 
    plt.imshow(x_test[image_index].reshape(372, 312),cmap='gray')
    pred = int(clf.predict(x_test[image_index].reshape(1, -1)))
    print('Indivíduo:', pred)
    plt.show()

def fp_keras():
    print("\n\nProcessamento com Keras\n")
    # Abre o arquivo
    pkl_file = open('fingerprint_data.pkl', 'rb')
    x_train, x_test, y_train, y_test = pickle.load(pkl_file)

    # Reshape para o formato correto
    x_train = x_train.reshape(x_train.shape[0], 372, 312, 1)
    x_test = x_test.reshape(x_test.shape[0], 372, 312, 1)
    input_shape = (372, 312, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normaliza
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print('Tamanho do conjunto de treinamento', x_train.shape[0])
    print('Tamanho do conjunto de teste', x_test.shape[0])

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(10,10), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(7,7)))
    model.add(Conv2D(64, kernel_size=(4,4), input_shape=(int(input_shape[0]/2), int(input_shape[1]/2),1)))
    model.add(MaxPooling2D(pool_size=(4,4)))
    model.add(Flatten())
    model.add(Dense(256, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(6,activation=tf.nn.softmax))
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    model.summary()

    model.fit(x=x_train,y=y_train, epochs=20)

    # Avaliação do modelo
    acc = model.evaluate(x_test, y_test)
    print('Acurácia = ', acc[1]*100, '%')

    # Avaliação individual

    image_index = 7
    plt.imshow(x_test[image_index].reshape(372, 312),cmap='gray')
    pred = model.predict(x_test[image_index].reshape(1, 372, 312, 1))
    print('Indivíduo:', pred.argmax())
    plt.show()

if __name__ == "__main__":
    main()

else:
    pass
