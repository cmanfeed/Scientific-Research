import glob
import cv2
import numpy as np

from sklearn.cluster import KMeans
from scipy.stats import skew, kurtosis

from hyperopt import hp
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics
import statistics

class PreProcess():

    def __init__(self, method):
        self.method = method

    def _hist_eqlize(self, img):
        return cv2.merge(([cv2.equalizeHist(ch)
                           for ch in cv2.split(img)]))

    def _cont_clahe(self, img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _oppt_colors(self, img):
        return 255 - img

    def apply(self, img):

        if self.method == 'eqlze':
            return self._hist_eqlize(img)
        
        elif self.method == 'clahe':
            return self._cont_clahe(img)
        
        elif self.method == 'oppcl':
            return self._oppt_colors(img)
        
        elif self.method == 'nopre':
            return img


class TransformSpectrum():

    def __init__(self, method):
        self.method = method

    def apply(self, img):

        if self.method == 'hsv':
            return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        elif self.method == 'lab':
            return cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        
        elif self.method == 'bgr':
            return img


class ApplyMask():

    def __init__(self, method):
        self.method = method

    def _app_mask(self, imgs_n, imgs_p):

        masks_n = [cv2.imread(file, 0)
                   for file in glob.iglob('./data/marcacoes/normal/*.jpg')]
        masks_p = [cv2.imread(file, 0)
                   for file in glob.iglob('./data/marcacoes/problema/*.jpg')]

        n_with_mask = [cv2.bitwise_and(img, img, mask=mask)
                       for img, mask in zip(imgs_n, masks_n)]
        p_with_mask = [cv2.bitwise_and(img, img, mask=mask)
                       for img, mask in zip(imgs_p, masks_p)]

        return n_with_mask, p_with_mask

    def apply(self, imgs_n, imgs_p):

        if self.method == 'appmask':
            return self._app_mask(imgs_n, imgs_p)

        return imgs_n, imgs_p


class ColorDescriptor():

    def __init__(self, method):
        self.method = method

    def _clr_histgram(self, img):
        hist = [cv2.calcHist([ch], [0], None, [256], [0, 256])
                for ch in cv2.split(img)]
        flat = [elm for h in hist for elm in h]

        return np.concatenate(np.array(flat), axis=0)

    def _clr_dominant(self, img):
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        clt = KMeans(n_clusters=3)
        clt.fit(img)

        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)

        hist = hist.astype("float")
        hist /= hist.sum()
        centers = np.concatenate(clt.cluster_centers_, axis=0)

        return np.concatenate((centers, hist), axis=0)

    def _clr_imoment(self, img):

        means = [np.mean(ch) for ch in cv2.split(img)]
        varis = [np.var(ch) for ch in cv2.split(img)]
        skews = [skew(ch.reshape(-1)) for ch in cv2.split(img)]
        kurts = [kurtosis(ch.reshape(-1)) for ch in cv2.split(img)]

        moments = means + varis + skews + kurts
        return np.array(moments)

    def _clr_all(self, img):
        return np.concatenate((self._clr_histgram(img),
                               self._clr_dominant(img),
                               self._clr_imoment(img)), axis=None)
    
    def _clr_imoment_dominant(self, img):
        return np.concatenate((self._clr_dominant(img),
                               self._clr_imoment(img)), axis=None)
    

    def apply(self, img):

        if self.method == 'histogrm':
            return self._clr_histgram(img)

        elif self.method == 'dominant':
            return self._clr_dominant(img)

        elif self.method == 'imoments':
            return self._clr_imoment(img)
        
        elif self.method == 'alldescp':
            return self._clr_all(img)
        
        elif self.method == 'imondomt':
            return self._clr_imoment_dominant(img)


class Classifier():

    def __init__(self, name):
        self.name = name
        
    def _create_XGB(self):
        clf = XGBClassifier(objective='binary:logistic',
                          verbosity=0, use_label_encoder=False)
        return clf
    
    def _create_ridge(self):
        clf = RidgeClassifier()
        return clf

    def _create_linearDA(self):
        clf = LinearDiscriminantAnalysis()
        return clf
    
    def _create_SVM(self):
        clf = SVC(kernel='rbf')
        return clf

    def _create_RandomForest(self):
        clf = RandomForestClassifier()
        return clf


    def get_scores(self, X, y):
        clf = self.apply()
        cv = KFold(n_splits=10, random_state=42, shuffle=True)
        
        acc, sen, esp = ([] for i in range(3))

        for (train, test) in cv.split(X, y):
            model = clf.fit(X[train], y[train])
            predict_test_class = model.predict(X[test])

            tn, fp, fn, tp = metrics.confusion_matrix(y[test], predict_test_class).ravel()
            
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

        return np.mean(acc), np.mean(sen), np.mean(esp)
 
    def apply(self):

        if self.name == 'xgboost':
            return self._create_XGB()
        
        elif self.name == 'ridgecl':
            return self._create_ridge()
        
        elif self.name == 'linarDA':
            return self._create_linearDA()

        elif self.name == 'svm':
            return self._create_SVM()

        elif self.name == 'randomforest':
            return self._create_RandomForest()