import numpy as np
import os
import cv2
import sklearn.metrics import confusion_matrix
from extract_features import *
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier

def load_pacients(n_path, p_path):
    normal = []
    problm = []

    for file in os.listdir(n_path):
        eyes = [cv2.imread(n_path + '/' + file + '/' + eye) for eye in os.listdir(n_path + '/' + file)]
        normal.append(eyes)
    
    for file in os.listdir(p_path):
        eyes = [cv2.imread(p_path + '/' + file + '/' + eye) for eye in os.listdir(p_path + '/' + file)]
        problm.append(eyes)
    
    return normal, problm


def load_masks(normal, nm_path, problm, pm_path):
    normal_m = []
    problm_m = []

    for patient, maskfile in zip(normal, os.listdir(nm_path)):
        eyes = []
        for pac_eye, mask_eye in zip(patient, os.listdir(nm_path + '/' + maskfile)):
            mask = cv2.imread(nm_path + '/' + maskfile + '/' + mask_eye, 0)
            eyem = cv2.bitwise_and(pac_eye, pac_eye, mask=mask)
            eyes.append(eyem)

        normal_m.append(eyes)
    
    for patient, maskfile in zip(problm, os.listdir(pm_path)):
        eyes = []
        for pac_eye, mask_eye in zip(patient, os.listdir(pm_path + '/' + maskfile)):
            mask = cv2.imread(pm_path + '/' + maskfile + '/' + mask_eye, 0)
            eyem = cv2.bitwise_and(pac_eye, pac_eye, mask=mask)
            eyes.append(eyem)

        problm_m.append(eyes)

    return normal_m, problm_m


def filter_transform(normal, problm, filt, spec):
    normal_pre = []
    problm_pre = []
    
    for patient in normal:
        eyes = [filt.apply(eye) for eye in patient]
        eyes = [spec.apply(eye) for eye in eyes]

        normal_pre.append(eyes)
    
    for patient in problm:
        eyes = [filt.apply(eye) for eye in patient]
        eyes = [spec.apply(eye) for eye in eyes]

        problm_pre.append(eyes)

    return normal_pre, problm_pre

def get_features(normal, problm, desc):
    X = []

    for patient in normal:
        feats = [desc.apply(eye) for eye in patient]
        X.append(feats)
    
    for patient in problm:
        feats = [desc.apply(eye) for eye in patient]
        X.append(feats)

    X = np.array(X)
    y = np.concatenate((np.zeros(len(normal), dtype=int),
                        np.ones(len(problm), dtype=int)))
    
    return X, y

def unstack_features(X, y):
    X_unstk = []
    y_unstk = []

    for x_elm, y_elm in zip(X, y):
        for x_elm_sub in x_elm:
            X_unstk.append(x_elm_sub)
            y_unstk.append(y_elm)

    return np.array(X_unstk), np.array(y_unstk)

def distance_calc(x_test, y_pred, x_rgbfeats, test):
    y_pred_final = []

    for i, j in zip(range(0, len(y_pred), 2), range(6)):
        if y_pred[i] == 0 and y_pred[i+1] == 0:
            dist = np.linalg.norm(x_rgbfeats[test[j]][0] - x_rgbfeats[test[j]][1])
            if dist > 700:
                y_pred_final.append(1)
            else:
                y_pred_final.append(0)
        else:
            y_pred_final.append(1)
    
    return y_pred_final


if __name__ == '__main__':

    filt = PreProcess('clahe')
    spec = TransformSpectrum('lab')
    desc = ColorDescriptor('imoments')

    # Getting best features with best params
    normal, problm = load_pacients('./data pacients/olhos/normal',
                                   './data pacients/olhos/problema')

    normal, problm = filter_transform(normal, problm, filt, spec)

    normal, problm = load_masks(normal, './data pacients/marcacoes/normal',
                                problm, './data pacients/marcacoes/problema')

    X, y = get_features(normal, problm, desc)

    # Getting RGB features
    normal, problm = load_pacients('./data pacients/olhos/normal',
                                   './data pacients/olhos/problema')
    
    normal, problm = load_masks(normal, './data pacients/marcacoes/normal',
                                problm, './data pacients/marcacoes/problema')

    X_rgb, _ = get_features(normal, problm, desc)

    # Validation metrics
    pre, sen, esp, acc = ([] for i in range(4))

    # Generating Cross Validation (K = 10)
    cv = KFold(n_splits=10, random_state=40, shuffle=True)

    for train, test in cv.split(X, y):
        x_tran, y_tran = unstack_eyes(X[train], y[train])
        x_test, y_test = unstack_eyes(X[test], y[test])

        model = XGBClassifier(objective='binary:logistic',verbosity=0, use_label_encoder=False).fit(x_tran, y_tran)
        
        y_pred = model.predict(x_test)
        y_pred_final = distance_calc(x_test, y_pred, X_rgb, test)

        cf = np.array([[0, 0], [0, 0]])
        if all(y[test] == y_pred_final):
            cf[0][0] += sum(y_pred == 0)
            cf[1][1] += sum(y_pred == 1)
        else:
            cf += confusion_matrix(y[test], y_pred_final)

        tn, fp, fn, tp = cf.ravel()

        if tp + fp > 0:
            pre.append(tp / (tp + fp))
        else:
            pre.append(0)
        if tp + fn > 0:
            sen.append(tp / (tp + fn))
        else:
            sen.append(0)

        esp.append(tn / (tn + fp))
        acc.append((tp + tn) / (tp + tn + fp + fn))

    print(np.mean(pre), np.mean(sen), np.mean(esp), np.mean(acc))

