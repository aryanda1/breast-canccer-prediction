import pickle
import numpy as np

def fn(a):
    mod = pickle.load(open('model','rb'))
    listt = pickle.load(open('dropIdx','rb'))
    scaler = pickle.load(open('stdScale','rb'))
    a = np.delete(a,listt)
    a.resize((1,19))
    a = scaler.transform(a)
    pred = mod.predict(a)
    return pred[0]

