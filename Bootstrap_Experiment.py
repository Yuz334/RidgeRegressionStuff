# -*- coding: utf-8 -*-
"""
Author: Yunyi Zhang
file: Bootstrap_Experiment.py
times: 3/3/20211:02 AM
"""

## Generate design matrix and data y
import numpy as np
import os
import json
from sklearn.model_selection import KFold
n = 500
p = 250
p1 = 20
z = 2.2
root = './experiment_500_250_NR'
beta = np.zeros(shape = (p, 1))
Mode = 'Combine'

for i in range(10):
    beta[i, 0] = 0.1 * (i + 7)
def generateData(n, p, p1, z, beta, lag = 10, tau = 0.9, div = 8.0):
    X = np.random.normal(loc = 0, scale = 1.0, size = (n, p))
    M = np.random.normal(loc = 0.5, scale = 0.5, size = (p1, p))
    H = np.zeros(shape = (n, n))
    for i in range(n):
        H[i ,i] = 1.0
        for j in range(i):
            H[i, j] = np.random.uniform(0.6, 0.9) if i - j <= lag else np.random.uniform(0.0, 2.0) / np.power(i - j, z)
    tau1 = np.random.normal(loc  = 0.0, scale = 1.0, size = n + 1)
    tau2 = np.power(tau1[1 : ], 2.0) * np.power(tau1[ : -1], 2.0) - 1.0
    eps = np.matmul(H, tau2.reshape(-1, 1)) / div
    y = np.matmul(X, beta) + eps
    return X, y, M, H

class Bootstrap:
    def __init__(self, rho, b_n, k_n, alpha = 0.05, Boot = 1000, simu = 1000, isForPower = False, bias = 0.0, lag = 50):
        self.rho = rho
        self.b_n = b_n
        self.k_n = k_n
        self.alpha1 = 1 - alpha
        self.Boot = Boot
        self.simu = simu
        self.isForPower = isForPower
        self.bias = bias
        self.lag = lag

    def kernel(self, x):
        return np.exp(- x * x / 2.0)


    def generateK(self, n):
        # This function generates Kn, kernel matrix
        K = np.zeros(shape = (n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = (i - j) / self.k_n
        return self.kernel(K)

    def generateRealResidual(self, H, div = 4.0):
        size = H.shape[0]
        tau1 = np.random.normal(loc=0.0, scale=1.0, size = size + 1)
        tau2 = np.power(tau1[1:], 2.0) * np.power(tau1[: -1], 2.0) - 1.0
        eps = np.matmul(H, tau2.reshape(-1, 1))
        eps2 = tau1[1 :] - 2.0 * tau1[: -1]
        return eps2.reshape((-1, 1))#eps.reshape((-1, 1)) / div


    def generateBootResidual(self, Kernel, len):
        size = Kernel.shape[0]
        return np.random.multivariate_normal(mean = np.zeros(shape = size), cov = Kernel, size = len)



    def truncate(self, a):
        shape = a.shape
        a = a.reshape(-1)
        b = np.zeros_like(a)
        validSet = (np.absolute(a) > self.b_n)
        b[validSet] = a[validSet]
        return b.reshape(shape), validSet

    def BootExp(self, X, beta, M, H):
        n, p = X.shape
        Xbeta = np.matmul(X, beta)
        Mbeta = np.matmul(M, beta) + self.bias
        isPos = 0.0
        quantile = []
        P, lamb, QT = np.linalg.svd(X, full_matrices = False)
        Lamb = np.diag(lamb)
        invmodLamb = np.linalg.inv(Lamb * Lamb + self.rho * np.eye(p))
        Xx = np.matmul(np.linalg.inv(np.matmul(X.T, X) + self.rho * np.eye(p)), X.T)
        FirstHalf = np.eye(p) + self.rho * np.matmul(QT.T, np.matmul(invmodLamb, QT))
        Hat = np.matmul(FirstHalf, Xx)
        lambAdj = lamb / (lamb * lamb + self.rho) + (self.rho * lamb) / np.power(lamb * lamb + self.rho, 2)
        lambAdj2 = lambAdj * lambAdj
        Kernel = self.generateK(n)
        for j in range(self.simu):
            if j % 100 == 3:
                print(isPos / j)
            y = Xbeta + self.generateRealResidual(H)
            beta_tilde = np.matmul(Hat, y)
            betaHat, valid = self.truncate(beta_tilde)
            gammaHat = np.matmul(M, betaHat)
            epsHat = y - np.matmul(X, betaHat)
            # Now we derive tau
            copyM = np.zeros_like(M)
            copyM[:, valid] = M[:, valid]
            C = np.matmul(copyM, QT.T)
            Csq = C * C
            tauHat = np.sqrt(np.matmul(Csq, lambAdj2) + 1.0 / n).reshape(-1)
            # Now perform bootstrap
            varEps = self.generateBootResidual(Kernel, self.Boot)
            XbetaHat = np.matmul(X, betaHat)
            stat = np.amax(np.absolute(gammaHat - Mbeta).reshape(-1) / tauHat)
            bootSamp = []
            for b in range(self.Boot):
                epsStar = varEps[b].reshape((n, 1)) * epsHat
                yStar = XbetaHat + epsStar
                betaStar_tilde = np.matmul(Hat, yStar)
                betaStar_Hat, validStar = self.truncate(betaStar_tilde)
                copyMStar = np.zeros_like(M)
                copyMStar[:, validStar] = M[:, validStar]
                CStar = np.matmul(copyMStar, QT.T)
                CStarsq = CStar * CStar
                tauStarHat = np.sqrt(np.matmul(CStarsq, lambAdj2) + 1.0 / n).reshape(-1)
                gammaStarHat = np.matmul(M, betaStar_Hat)
                statStar = np.amax(np.absolute(gammaStarHat - gammaHat).reshape(-1) / tauStarHat)
                bootSamp.append(statStar)
            bootQuan = np.quantile(bootSamp, q = self.alpha1)
            quantile.append(bootQuan)
            if stat <= bootQuan:
                isPos += 1.0
        if self.isForPower:
            return 1.0 - isPos / self.simu, np.mean(quantile)
        else:
            return isPos / self.simu, np.mean(quantile)


    def resiBootExp(self, X, beta, M, H):
        n, p = X.shape
        Xbeta = np.matmul(X, beta)
        Mbeta = np.matmul(M, beta) + self.bias
        isPos = 0.0
        quantile = []
        P, lamb, QT = np.linalg.svd(X, full_matrices = False)
        Lamb = np.diag(lamb)
        invmodLamb = np.linalg.inv(Lamb * Lamb + self.rho * np.eye(p))
        Xx = np.matmul(np.linalg.inv(np.matmul(X.T, X) + self.rho * np.eye(p)), X.T)
        FirstHalf = np.eye(p) + self.rho * np.matmul(QT.T, np.matmul(invmodLamb, QT))
        Hat = np.matmul(FirstHalf, Xx)
        lambAdj = lamb / (lamb * lamb + self.rho) + (self.rho * lamb) / np.power(lamb * lamb + self.rho, 2)
        lambAdj2 = lambAdj * lambAdj
        for j in range(self.simu):
            if j % 100 == 3:
                print(isPos / j)
            y = Xbeta + self.generateRealResidual(H)
            beta_tilde = np.matmul(Hat, y)
            betaHat, valid = self.truncate(beta_tilde)
            gammaHat = np.matmul(M, betaHat)
            epsHat = y - np.matmul(X, betaHat)
            epsHat = epsHat - np.mean(epsHat)
            # Now we derive tau
            copyM = np.zeros_like(M)
            copyM[:, valid] = M[:, valid]
            C = np.matmul(copyM, QT.T)
            Csq = C * C
            tauHat = np.sqrt(np.matmul(Csq, lambAdj2) + 1.0 / n).reshape(-1)
            # Now perform bootstrap
            XbetaHat = np.matmul(X, betaHat)
            stat = np.amax(np.absolute(gammaHat - Mbeta).reshape(-1) / tauHat)
            bootSamp = []
            for b in range(self.Boot):
                epsStar = np.random.choice(epsHat.reshape(-1), size = (n, 1))
                yStar = XbetaHat + epsStar
                betaStar_tilde = np.matmul(Hat, yStar)
                betaStar_Hat, validStar = self.truncate(betaStar_tilde)
                copyMStar = np.zeros_like(M)
                copyMStar[:, validStar] = M[:, validStar]
                CStar = np.matmul(copyMStar, QT.T)
                CStarsq = CStar * CStar
                tauStarHat = np.sqrt(np.matmul(CStarsq, lambAdj2) + 1.0 / n).reshape(-1)
                gammaStarHat = np.matmul(M, betaStar_Hat)
                statStar = np.amax(np.absolute(gammaStarHat - gammaHat).reshape(-1) / tauStarHat)
                bootSamp.append(statStar)
            bootQuan = np.quantile(bootSamp, q = self.alpha1)
            quantile.append(bootQuan)
            if stat <= bootQuan:
                isPos += 1.0
        if self.isForPower:
            return 1.0 - isPos / self.simu, np.mean(quantile)
        else:
            return isPos / self.simu, np.mean(quantile)

    def wildBootstrap(self, X, beta, M, H):
        n, p = X.shape
        Xbeta = np.matmul(X, beta)
        Mbeta = np.matmul(M, beta) + self.bias
        isPos = 0.0
        quantile = []
        P, lamb, QT = np.linalg.svd(X, full_matrices=False)
        Lamb = np.diag(lamb)
        invmodLamb = np.linalg.inv(Lamb * Lamb + self.rho * np.eye(p))
        Xx = np.matmul(np.linalg.inv(np.matmul(X.T, X) + self.rho * np.eye(p)), X.T)
        FirstHalf = np.eye(p) + self.rho * np.matmul(QT.T, np.matmul(invmodLamb, QT))
        Hat = np.matmul(FirstHalf, Xx)
        lambAdj = lamb / (lamb * lamb + self.rho) + (self.rho * lamb) / np.power(lamb * lamb + self.rho, 2)
        lambAdj2 = lambAdj * lambAdj
        for j in range(self.simu):
            if j % 100 == 3:
                print(isPos / j)
            y = Xbeta + self.generateRealResidual(H)
            beta_tilde = np.matmul(Hat, y)
            betaHat, valid = self.truncate(beta_tilde)
            gammaHat = np.matmul(M, betaHat)
            epsHat = y - np.matmul(X, betaHat)
            epsHat = epsHat - np.mean(epsHat)
            # Now we derive tau
            copyM = np.zeros_like(M)
            copyM[:, valid] = M[:, valid]
            C = np.matmul(copyM, QT.T)
            Csq = C * C
            tauHat = np.sqrt(np.matmul(Csq, lambAdj2) + 1.0 / n).reshape(-1)
            # Now perform bootstrap
            XbetaHat = np.matmul(X, betaHat)
            stat = np.amax(np.absolute(gammaHat - Mbeta).reshape(-1) / tauHat)
            bootSamp = []
            for b in range(self.Boot):
                prd = np.random.normal(loc = 0.0, scale = 1.0, size = n)
                epsStar = (prd * epsHat.reshape(-1)).reshape((-1, 1))
                yStar = XbetaHat + epsStar
                betaStar_tilde = np.matmul(Hat, yStar)
                betaStar_Hat, validStar = self.truncate(betaStar_tilde)
                copyMStar = np.zeros_like(M)
                copyMStar[:, validStar] = M[:, validStar]
                CStar = np.matmul(copyMStar, QT.T)
                CStarsq = CStar * CStar
                tauStarHat = np.sqrt(np.matmul(CStarsq, lambAdj2) + 1.0 / n).reshape(-1)
                gammaStarHat = np.matmul(M, betaStar_Hat)
                statStar = np.amax(np.absolute(gammaStarHat - gammaHat).reshape(-1) / tauStarHat)
                bootSamp.append(statStar)
            bootQuan = np.quantile(bootSamp, q=self.alpha1)
            quantile.append(bootQuan)
            if stat <= bootQuan:
                isPos += 1.0
        if self.isForPower:
            return 1.0 - isPos / self.simu, np.mean(quantile)
        else:
            return isPos / self.simu, np.mean(quantile)


    def faciComb(self, XbetaHat, Hat, lambAdj2, gammaHat, QT, M, epsStar):
        yStar = XbetaHat + epsStar
        betaStar_tilde = np.matmul(Hat, yStar)
        betaStar_Hat, validStar = self.truncate(betaStar_tilde)
        copyMStar = np.zeros_like(M)
        copyMStar[:, validStar] = M[:, validStar]
        CStar = np.matmul(copyMStar, QT.T)
        CStarsq = CStar * CStar
        tauStarHat = np.sqrt(np.matmul(CStarsq, lambAdj2) + 1.0 / n).reshape(-1)
        gammaStarHat = np.matmul(M, betaStar_Hat)
        statStar = np.amax(np.absolute(gammaStarHat - gammaHat).reshape(-1) / tauStarHat)
        return statStar

    def combBootstrap(self, X, beta, M, H):
        n, p = X.shape
        Xbeta = np.matmul(X, beta)
        Mbeta = np.matmul(M, beta) + self.bias
        DepisPos = 0.0
        ResisPos = 0.0
        WidisPos = 0.0
        Depquantile = []
        Widquantile = []
        Resquantile = []
        P, lamb, QT = np.linalg.svd(X, full_matrices=False)
        Lamb = np.diag(lamb)
        invmodLamb = np.linalg.inv(Lamb * Lamb + self.rho * np.eye(p))
        Xx = np.matmul(np.linalg.inv(np.matmul(X.T, X) + self.rho * np.eye(p)), X.T)
        FirstHalf = np.eye(p) + self.rho * np.matmul(QT.T, np.matmul(invmodLamb, QT))
        Hat = np.matmul(FirstHalf, Xx)
        lambAdj = lamb / (lamb * lamb + self.rho) + (self.rho * lamb) / np.power(lamb * lamb + self.rho, 2)
        lambAdj2 = lambAdj * lambAdj
        Kernel = self.generateK(n)
        for j in range(self.simu):
            if j % 100 == 3:
                print(DepisPos / (j + 1.0))
            y = Xbeta + self.generateRealResidual(H)
            beta_tilde = np.matmul(Hat, y)
            betaHat, valid = self.truncate(beta_tilde)
            gammaHat = np.matmul(M, betaHat)
            epsHat = y - np.matmul(X, betaHat)
            epsHatX = epsHat - np.mean(epsHat)
            # Now we derive tau
            copyM = np.zeros_like(M)
            copyM[:, valid] = M[:, valid]
            C = np.matmul(copyM, QT.T)
            Csq = C * C
            tauHat = np.sqrt(np.matmul(Csq, lambAdj2) + 1.0 / n).reshape(-1)
            # Now perform bootstrap
            XbetaHat = np.matmul(X, betaHat)
            varEps = self.generateBootResidual(Kernel, self.Boot)
            stat = np.amax(np.absolute(gammaHat - Mbeta).reshape(-1) / tauHat)
            bootDep = []
            bootRes = []
            bootWid = []
            for b in range(self.Boot):
                epsRes = np.random.choice(epsHatX.reshape(-1), size = (n, 1))
                prd = np.random.normal(loc=0.0, scale=1.0, size=n)
                epsWid = (prd * epsHatX.reshape(-1)).reshape((-1, 1))
                epsStar = varEps[b].reshape((n, 1)) * epsHat
                bootDep.append(self.faciComb(XbetaHat, Hat, lambAdj2, gammaHat, QT, M, epsStar))
                bootRes.append(self.faciComb(XbetaHat, Hat, lambAdj2, gammaHat, QT, M, epsRes))
                bootWid.append(self.faciComb(XbetaHat, Hat, lambAdj2, gammaHat, QT, M, epsWid))
            bootDepQuan = np.quantile(bootDep, q = self.alpha1)
            bootResQuan = np.quantile(bootRes, q = self.alpha1)
            bootWidQuan = np.quantile(bootWid, q = self.alpha1)
            Depquantile.append(bootDepQuan)
            Resquantile.append(bootResQuan)
            Widquantile.append(bootWidQuan)
            if stat <= bootDepQuan:
                DepisPos += 1.0
            if stat <= bootResQuan:
                ResisPos += 1.0
            if stat <= bootWidQuan:
                WidisPos += 1.0
        probList = {}
        quanList = {}
        probList['Dep'] = DepisPos / self.simu
        probList['Res'] = ResisPos / self.simu
        probList['Wid'] = WidisPos / self.simu
        quanList['Dep'] = np.mean(Depquantile)
        quanList['Res'] = np.mean(Resquantile)
        quanList['Wid'] = np.mean(Widquantile)
        return probList, quanList



def truncate(thre, a):
    shape = a.shape
    a = a.reshape(-1)
    b = np.zeros_like(a)
    validSet = (np.absolute(a) > thre)
    b[validSet] = a[validSet]
    return b.reshape(shape), validSet


def thresholdDeb(X, y, rho, b):
    n, p = X.shape
    P, lamb, QT = np.linalg.svd(X)
    lambModf = lamb * lamb + rho
    betaTildeStar = np.matmul(QT.T, np.matmul(QT, np.matmul(X.T, y)) / lambModf.reshape((p, 1)))
    betaTilde = betaTildeStar + rho * np.matmul(QT.T, np.matmul(QT, betaTildeStar) / lambModf.reshape((p, 1)))
    return truncate(b, betaTilde)[0]

def LASSO(X, y, rho):
    n, p = X.shape
    from sklearn.linear_model import Lasso
    lAsso = Lasso(alpha = rho / (2.0 * n), fit_intercept = False)
    lAsso.fit(X, y)
    return lAsso.coef_.reshape((p, 1))

def thresholdLASSO(X, y, rho, b):
    n, p = X.shape
    from sklearn.linear_model import Lasso
    lAsso = Lasso(alpha=rho / (2.0 * n), fit_intercept=False)
    lAsso.fit(X, y)
    betaTilde = np.copy(lAsso.coef_.reshape((p, 1)))
    return truncate(b, betaTilde)[0]

def RIDGE(X, y, rho):
    n, p = X.shape
    from sklearn.linear_model import Ridge
    rIdge = Ridge(alpha = rho)
    rIdge.fit(X, y)
    return rIdge.coef_.reshape((p, 1))

def thresholdRIDGE(X, y, rho, b):
    n, p = X.shape
    from sklearn.linear_model import Ridge
    rIdge = Ridge(alpha=rho, fit_intercept=False)
    rIdge.fit(X, y)
    betaTilde = np.copy(rIdge.coef_.reshape((p, 1)))
    return truncate(b, betaTilde)[0]

def ELASTICNET(X, y, rho):
    n, p = X.shape
    from sklearn.linear_model import ElasticNet
    eLastic = ElasticNet(alpha = rho / (2.0 * n))
    eLastic.fit(X, y)
    return eLastic.coef_.reshape((p, 1))


class ModelSelection:
    def __init__(self, X, y, M, H, beta, rhos, bs):
        self.X = X
        self.y = y
        self.rhos = rhos
        self.bs = bs
        self.M = M
        self.trainX = None
        self.trainy = None
        self.testX = None
        self.testy = None
        self.beta = beta
        self.H = H

    def crossValidation(self, kfold, root = './CrossValidationRes/'):
        Mbeta = np.matmul(self.M, self.beta)
        kFold = KFold(n_splits = kfold)
        sizeRho, sizeB = self.rhos.shape[0], self.bs.shape[0]
        thDebRes = np.zeros((sizeRho, sizeB))
        thLasRes = np.zeros((sizeRho, sizeB))
        thRidRes = np.zeros((sizeRho, sizeB))
        lasRes = np.zeros(sizeRho)
        ridRes = np.zeros(sizeRho)
        elaRes = np.zeros(sizeRho)
        mov = 0.0
        for train, test in kFold.split(self.X):
            self.trainX = self.X[train]
            self.testX = self.X[test]
            self.trainy = self.y[train]
            self.testy = self.y[test]
            for i in range(sizeRho):
                print('\r' + 'Finish' + str(mov * 1.0 / kfold + i / (kfold * sizeRho)), end = '', flush = True)
                for j in range(sizeB):
                    thDebRes[i, j] += np.linalg.norm(self.testy - np.matmul(self.testX, thresholdDeb(self.trainX, self.trainy, self.rhos[i], self.bs[j])))
                    thLasRes[i, j] += np.linalg.norm(self.testy - np.matmul(self.testX, thresholdLASSO(self.trainX, self.trainy, self.rhos[i], self.bs[j])))
                    thRidRes[i, j] += np.linalg.norm(self.testy - np.matmul(self.testX, thresholdRIDGE(self.trainX, self.trainy, self.rhos[i], self.bs[j])))
            for i in range(sizeRho):
                lasRes[i] += np.linalg.norm(self.testy - np.matmul(self.testX, LASSO(self.trainX, self.trainy, self.rhos[i])))
                ridRes[i] += np.linalg.norm(self.testy - np.matmul(self.testX, RIDGE(self.trainX, self.trainy, self.rhos[i])))
                elaRes[i] += np.linalg.norm(self.testy - np.matmul(self.testX, ELASTICNET(self.trainX, self.trainy, self.rhos[i])))
            mov += 1.0
        optthDeb = np.unravel_index(np.argmin(thDebRes, axis=None), thDebRes.shape)
        optthLas = np.unravel_index(np.argmin(thLasRes, axis=None), thLasRes.shape)
        optthRid = np.unravel_index(np.argmin(thRidRes, axis=None), thRidRes.shape)
        optLas = np.argmin(lasRes)
        optRid = np.argmin(ridRes)
        optEla = np.argmin(elaRes)

        optimalPara = {
            'thDeb_rho' : self.rhos[optthDeb[0]],
            'thDeb_b' : self.bs[optthDeb[1]],
            'thLas_rho' : self.rhos[optthLas[0]],
            'thLas_b' : self.bs[optthLas[1]],
            'thRid_rho' : self.rhos[optthRid[0]],
            'thRid_b' : self.bs[optthRid[1]],
            'Las_rho' : self.rhos[optLas],
            'Rid_rho' : self.rhos[optRid],
            'Ela_rho' : self.rhos[optEla]
        }

        for i in range(sizeRho):
            for j in range(sizeB):
                thDebRes[i, j] = np.linalg.norm(Mbeta - np.matmul(M, thresholdDeb(self.X, self.y, self.rhos[i], self.bs[j])))
                thLasRes[i, j] = np.linalg.norm(Mbeta - np.matmul(M, thresholdLASSO(self.X, self.y, self.rhos[i], self.bs[j])))
                thRidRes[i, j] = np.linalg.norm(Mbeta - np.matmul(M, thresholdRIDGE(self.X, self.y, self.rhos[i], self.bs[j])))

        for i in range(sizeRho):
            lasRes[i] = np.linalg.norm(Mbeta - np.matmul(M, LASSO(self.X, self.y, self.rhos[i])))
            ridRes[i] = np.linalg.norm(Mbeta - np.matmul(M, RIDGE(self.X, self.y, self.rhos[i])))
            elaRes[i] = np.linalg.norm(Mbeta - np.matmul(M, ELASTICNET(self.X, self.y, self.rhos[i])))

        if not os.path.exists(root):
            os.makedirs(root)
        np.savetxt(fname = os.path.join(root, 'thDebRes.txt'),  X = thDebRes)
        np.savetxt(fname = os.path.join(root, 'thLasRes.txt'), X = thLasRes)
        np.savetxt(fname=os.path.join(root, 'thRidRes.txt'), X=thRidRes)
        np.savetxt(fname=os.path.join(root, 'lasRes.txt'), X=lasRes)
        np.savetxt(fname=os.path.join(root, 'ridRes.txt'), X=ridRes)
        np.savetxt(fname=os.path.join(root, 'elaRes.txt'), X=elaRes)
        np.savetxt(fname = os.path.join(root, 'Design_Matrix.txt'), X = self.X)
        np.savetxt(fname=os.path.join(root, 'LinComb.txt'), X = self.M)
        np.savetxt(fname=os.path.join(root, 'H.txt'), X=self.H)
        np.savetxt(fname = os.path.join(root, 'rhos.txt'), X = self.rhos)
        np.savetxt(fname = os.path.join(root, 'bs.txt'), X = self.bs)
        np.savetxt(fname = os.path.join(root, 'beta.txt'), X = self.beta)
        jSon = json.dumps(optimalPara)
        files = open(os.path.join(root, 'optimalPara.json'), 'w')
        files.write(jSon)
        files.close()








if __name__ == '__main__':
    if Mode != 'Selection':
        X = np.loadtxt(os.path.join(root, 'Design_Matrix.txt'))
        beta = np.loadtxt(os.path.join(root, 'beta.txt')).reshape((-1, 1))
        M = np.loadtxt(os.path.join(root, 'LinComb.txt'))
        M1 = np.eye(N = M.shape[0], M = M.shape[1])
        n, p = X.shape
        H = np.loadtxt(os.path.join(root, 'H.txt'))
        optimPara = None
        with open(os.path.join(root, 'optimalPara.json')) as json_file:
            optimPara = json.load(json_file)
        rho, bn = optimPara['thDeb_rho'], optimPara['thDeb_b']
        P, la, Q = np.linalg.svd(X, full_matrices=False)
        print(rho)
        print(bn)
        print(la[0])
        print(la[-1])
        print('\n')
        print(M.shape)

    if Mode == 'Selection':
        X, y, M, H = generateData(n, p, p1, z, beta, lag = 10, div = 4)
        P, lamb, QT = np.linalg.svd(X)
        lambmin = lamb[p-1]
        eta = np.log(lambmin) / np.log(n)
        m = 20
        alphabeta = 0.1
        delta = (eta + alphabeta + 0.2) / 2.0
        rho_base = np.power(n, 2.0 * eta - delta)
        v_b = eta - 3.0 / m
        bn_base = np.power(n, -v_b)
        z1 = 2.0 * eta - 2.0 * v_b - 4.0 / m
        z2 = eta / 2.0 - 3.0 / (2.0 * m)
        kn_base = np.power(n, min(z1, z2))
        moving_rho = 0.05 * np.power(1.15, np.linspace(start = 0, stop = 45, num = 40))
        moving_b = 0.1 * np.power(1.15, np.linspace(start = 0, stop = 20, num = 40))
        modelselection = ModelSelection(X, y, M, H, beta, rhos = rho_base * moving_rho, bs = bn_base * moving_b)
        modelselection.crossValidation(kfold = 10, root = root)

    if Mode == 'Test':
        kn = 16.749
        boot = Bootstrap(rho, bn, kn, alpha = 0.10)
        isPos, quan = boot.BootExp(X, beta, M, H)
        print(isPos)
        print(quan)


    if Mode == 'Residual':
        boot = Bootstrap(rho, bn, 0.0, alpha=0.10)
        isPos, quan = boot.resiBootExp(X, beta, M, H)
        print(isPos)
        print(quan)

    if Mode == 'Wild':
        boot = Bootstrap(rho, bn, 0.0, alpha = 0.10)
        isPos, quan = boot.wildBootstrap(X, beta, M, H)
        print(isPos)
        print(quan)

    if Mode == 'Power':
        kn = 3.138
        move = 1
        deltas = [0.05 * i for i in range(move, 12)]
        powers = []
        for delta in deltas:
            boot = Bootstrap(rho, bn, kn, alpha = 0.10, isForPower = True, bias = delta)
            isPos, quan = boot.BootExp(X, beta, M, H)
            powers.append(isPos)
            np.savetxt(fname=os.path.join(root, 'power.txt'), X=np.array(powers))
            move += 1

    if Mode == 'Combine':
        kn = 17.7
        boot = Bootstrap(rho, bn, kn, alpha=0.10)
        pList, Qlist = boot.combBootstrap(X, beta, M1, H)
        print(pList)
        print(Qlist)












