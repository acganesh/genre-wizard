import wave
import struct
import numpy as np
from scikits.talkbox.features import mfcc
from sklearn.metrics import confusion_matrix

from sklearn import svm
from sklearn import grid_search

import pickle

import matplotlib.pyplot as plt

GENRE_LIST = ["blues", "classical", "country", "disco", "jazz", "metal", "pop", "reggae", "rock"]

def read_wav(data_size, fname):
    wav_file = wave.open(fname, 'r')
    data = wav_file.readframes(data_size)
    wav_file.close()

    data = struct.unpack('{n}h'.format(n=data_size), data)
    data = np.array(data)
    return data

def compute_mfcc_features(data):
    ceps, mspec, spec = mfcc(data)
    num_ceps = ceps.shape[0]
    averaged_mfcc = np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0)
    return averaged_mfcc

def read_file(num, genre):
    num_str = str(num)
    if len(num_str) == 1:
        num_str = "0000"+num_str
    elif len(num_str) == 2:
        num_str = "000"+num_str

    base_path = "./genres/"+genre+"/"+genre+"."
    end_path = ".au.wav"
    full_path = base_path+num_str+end_path

    print full_path

    data = read_wav(DATA_SIZE, full_path)
    mfcc_features = compute_mfcc_features(data)

    return mfcc_features

def process_files():
    DATA_SIZE = int(40000*4100.0/250.0)

    training_data = []
    labels = []

    test_data = []
    test_labels = []

    for genre in GENRE_LIST:
        total_examples = 100
        num_training = 90
        #total_examples = 20
        #num_training = 10
        num = 0

        while num < num_training:
            
            mfcc_features = read_file(num, genre)
            training_data.append(mfcc_features)
            labels.append(genre)

            num += 1

        while num < total_examples:

            mfcc_features = read_file(num, genre)
            test_data.append(mfcc_features)
            test_labels.append(genre)

            num += 1

    data = {'training_data':training_data, 'labels':labels, 'test_data':test_data, 'test_labels':test_labels}

def save_data():
    with open('data.pkl', 'wb') as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

#This grid search empirically searches a space of hyperparameters to find the ones that optimally perform on our dataset.  This only needs to be done once!
def SVM_grid_search():
    parameters = {'kernel':('linear', 'rbf', 'poly', 'sigmoid')}

    power = -5
    C_list = []
    while power < 16:
        C_list.append(pow(2,power))
        power += 2

    power = -15
    gamma_list = []
    while power < 4:
        gamma_list.append(pow(2,power))
        power += 2

    parameters['C'] = C_list
    parameters['gamma'] = gamma_list

    classifier_GS = grid_search.GridSearchCV(classifier, parameters, verbose=10)


    classifier_GS.fit(training_data, labels)

    prediction = classifier_GS.predict(test_data)
    params = classifier_GS.get_params
    print params


#The following function extracts MFCC features from the GTZAN .wav file dataset.  This only needs to be done once, and the data is already stored in data.pkl.
#process_files()
#save_data()
#


with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

training_data = data['training_data']
labels = data['labels']

test_data = data['test_data']
test_labels= data['test_labels']

classifier = svm.SVC(C = 0.125, kernel = 'poly', gamma = 0.125)
classifier.fit(training_data, labels)

prediction = classifier.predict(test_data)

cm = confusion_matrix(prediction, test_labels)

print(cm)

print("CORRECT PREDICTIONS:")
correct = np.trace(cm)
print(correct)
print("TOTAL PREDICTIONS:")
total = np.sum(cm)
print(total)
print("ACCURACY:")
print(float(correct)/float(total))

plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()




