import pickle
import numpy as np

def fn(a):
    cat = ["Benign","Malignant"]
    mod = pickle.load(open('New/model','rb'))
    listt = pickle.load(open('New/dropIdx','rb'))
    scaler = pickle.load(open('New/stdScale','rb'))
    a = np.delete(a,listt)
    a.resize((1,19))
    a = scaler.transform(a)
    pred = mod.predict(a)
    return cat[pred[0]]

