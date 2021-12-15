# -*- coding: utf-8 -*-
"""
Author: Yunyi Zhang
file: New_Estimation.py
times: 7/28/202012:06 PM
"""
import numpy as np
class BootTest:
    rho = 0.0
    b_n = 0.0
    alpha = 0.05
    Bootrep = 2000
    tol = 0.00001
    testTime = 3000
    resiPara = 1.0 # standard deviation of residuals

    def __init__(self, grho, gb_n, galpha, gBootrep, gtol, gTest, gPara, bias = 0.0):
        self.rho = grho
        self.b_n = gb_n
        self.alpha = galpha
        self.Bootrep = gBootrep
        self.tol = gtol
        self.testTime = gTest
        self.resiPara = gPara
        self.bias = bias

    def tightSVD(self, Mat):
        from numpy.linalg import svd
        from numpy import eye
        P_r, Lamb, Q_Tr = svd(Mat)
        move = 0
        while (move < Lamb.shape[0] and Lamb[move] > self.tol) :
            move += 1
        P = P_r[:, 0 : move]
        Q = Q_Tr.T[:, 0 : move]
        Q_perp = Q_Tr.T[:, move :] if move < Mat.shape[1] else None
        lb = eye(move)
        for i in range(move):
            lb[i, i] = Lamb[i]
        lmin = Lamb[move - 1]
        return lb, lmin, P, Q, Q_perp

    def threshold(self, vec):
        from numpy import absolute
        vec1 = vec.reshape(-1)
        l = vec1.shape[0]
        absVec = absolute(vec1)
        nullSet = (absVec <= self.b_n)
        vec1[nullSet] = 0.0
        return vec1.reshape((l, 1)), nullSet
        # Return n x 1 vector and null set

    def calSD(self, X, y, coef):
        from numpy import matmul, inner, sqrt
        size, dimen = X.shape
        resi = (y - matmul(X, coef)).reshape(-1)
        var = inner(resi, resi) / size
        return sqrt(var)

    def generateResidual(self, size):
        # need to change if we need another situation
        from numpy.random import normal, exponential, standard_t
        #err1 = exponential(scale = self.resiPara, size = (size, 1))
        #err2 = exponential(scale = self.resiPara, size = (size, 1))
        #return err1 - err2
        return normal(0.0, scale = self.resiPara, size=(size, 1))


    def calTau(self, M, nullSet, Q, tLamb, size):
        from numpy import matmul, sqrt
        modM = M
        modM[:, nullSet] = 0.0
        C = matmul(modM, Q)
        C_2 = C * C
        tVec = (matmul(C_2, tLamb) + 1 / size).reshape(-1)
        return sqrt(tVec)

    def calQuantile(self, quans):
        from numpy import sort
        if quans.shape[0] == 0:
            return 0.0
        rets = sort(quans)  # Small to big
        size = int((1 - self.alpha) * quans.shape[0])
        crucial1 = rets[size]
        crucial2 = rets[size]
        if size < quans.shape[0] - 1:
            crucial2 = rets[size + 1]
        return (crucial1 + crucial2) / 2.0

    def test(self, X, beta, M):
        from numpy import matmul, eye, zeros, divide, absolute, amax, mean
        from numpy.linalg import inv
        from numpy.random import normal
        # Note that we know the real beta
        Mbeta = matmul(M, beta) + self.bias
        Xbeta = matmul(X, beta)
        size, dimen = X.shape
        invX = inv(matmul(X.T, X) + self.rho * eye(dimen))
        invX = matmul(invX, X.T)
        Lamb, lmin, P, Q, Q_perp = self.tightSVD(X)
        rank = Lamb.shape[0]
        adjMat = inv(Lamb * Lamb + self.rho * eye(rank))
        adjMat = self.rho * matmul(Q, matmul(adjMat, Q.T))
        tauLambda = zeros(shape=(rank, 1))
        realnull = (absolute(beta) <= self.b_n).reshape(-1)
        equalCount = 0.0
        gammaCount = []
        SigmaCount = []
        for i in range(rank):
            lamb = Lamb[i, i]
            tail = lamb * lamb + self.rho
            half = lamb / tail + (self.rho * lamb) / (tail * tail)
            tauLambda[i, 0] = half * half

        # Now, we start testing
        freq = 0.0
        avgLen = zeros(shape= self.testTime)
        for i in range(self.testTime):
            y = Xbeta + self.generateResidual(size)
            thTildeStar = matmul(invX, y)
            thTilde = thTildeStar + matmul(adjMat, thTildeStar)
            thHat, nullTh = self.threshold(thTilde)
            sd = self.calSD(X, y, thHat)
            gaHat = matmul(M, thHat)
            tauHat = self.calTau(M, nullTh, Q, tauLambda, size)
            absGamma = absolute(gaHat - Mbeta).reshape(-1)
            maxStat = amax(divide(absGamma, tauHat))
            XbetaHat = matmul(X, thHat)
            theta_Prep = matmul(Q_perp, matmul(Q_perp.T, thHat)) if Q_perp is not None else zeros(shape = (dimen , 1))
            quan = zeros(shape = self.Bootrep)
            for j in range(self.Bootrep):
                yStar = XbetaHat + normal(0.0, scale = sd, size = (size, 1))
                thTildeStar_ = matmul(invX, yStar)
                thTilde_ = thTildeStar_ + matmul(adjMat, thTildeStar_) + theta_Prep
                thHat_, nullTh_ = self.threshold(thTilde_)
                gaHat_ = matmul(M, thHat_)
                tauHat_ = self.calTau(M, nullTh_, Q, tauLambda, size)
                maxStat_ = amax(divide(absolute(gaHat_ - gaHat).reshape(-1), tauHat_))
                quan[j] = maxStat_
            curQuan = self.calQuantile(quan)
            avgLen[i] = curQuan
            if maxStat <= curQuan:
                freq += 1.0
            if i % 200 == 1:
                print(freq / i)
        return freq / self.testTime #, mean(avgLen), equalCount / self.testTime, np.mean(gammaCount), np.mean(np.absolute(SigmaCount))

    def calResi(self, X, y, beta):
        from numpy import matmul, mean
        eps = (y - matmul(X, beta)).reshape(-1)
        epsMean = mean(eps)
        return eps - epsMean


    def prdTest(self, X, beta, M):
        from numpy import matmul, eye, zeros, divide, absolute, amax, mean
        from numpy.linalg import inv
        from numpy.random import normal, choice
        # Note that we know the real beta
        Mbeta = matmul(M, beta)
        Xbeta = matmul(X, beta)
        size, dimen = X.shape
        numLimb = M.shape[0]
        invX = inv(matmul(X.T, X) + self.rho * eye(dimen))
        invX = matmul(invX, X.T)
        Lamb, lmin, P, Q, Q_perp = self.tightSVD(X)
        rank = Lamb.shape[0]
        adjMat = inv(Lamb * Lamb + self.rho * eye(rank))
        adjMat = self.rho * matmul(Q, matmul(adjMat, Q.T))
        tauLambda = zeros(shape=(rank, 1))
        for i in range(rank):
            lamb = Lamb[i, i]
            tail = lamb * lamb + self.rho
            half = lamb / tail + (self.rho * lamb) / (tail * tail)
            tauLambda[i, 0] = half * half

        # Now, we start testing
        freq = 0.0
        avgQuan = []
        for i in range(self.testTime):
            y = Xbeta + self.generateResidual(size)
            yNew = Mbeta + self.generateResidual(numLimb)
            # Estimation
            thTildeStar = matmul(invX, y)
            thTilde = thTildeStar + matmul(adjMat, thTildeStar)
            thHat, nullTh = self.threshold(thTilde)
            sd = self.calSD(X, y, thHat)
            epsHat = self.calResi(X, y, thHat)
            gaHat = matmul(M, thHat)
            pdRoot = absolute(yNew - gaHat).reshape(-1)
            maxStat = amax(pdRoot)
            XbetaHat = matmul(X, thHat)
            theta_Prep = matmul(Q_perp, matmul(Q_perp.T, thHat)) if Q_perp is not None else zeros(shape=(dimen, 1))
            quan = zeros(shape=self.Bootrep)
            # Bootstrap
            for j in range(self.Bootrep):
                yStar = XbetaHat + normal(0.0, scale = sd, size=(size, 1))
                yNewStar = gaHat + choice(epsHat, size = (numLimb, 1))
                thTildeStar_ = matmul(invX, yStar)
                thTilde_ = thTildeStar_ + matmul(adjMat, thTildeStar_) + theta_Prep
                thHat_, nullTh_ = self.threshold(thTilde_)
                gaHat_ = matmul(M, thHat_)
                pdRoot_ = absolute(yNewStar - gaHat_).reshape(-1)
                maxStat_ = amax(pdRoot_)
                quan[j] = maxStat_
            curQuan = self.calQuantile(quan)
            if maxStat <= curQuan:
                freq += 1.0
            if i % 200 == 1:
                print(freq / i)
            avgQuan.append(curQuan)
        return freq / self.testTime, mean(avgQuan)




if __name__ == '__main__':
    from numpy import zeros, loadtxt, sqrt, matmul
    import os
    file_folder = #
    DesignFileName = os.path.join(file_folder, 'Design_1000_1500_800_N_NEW2_H.txt')
    LimCombName = os.path.join(file_folder, 'LinComb_1000_1500_800_N_NEW2_H.txt')
    Design = loadtxt(DesignFileName)
    LimComb = loadtxt(LimCombName)
    size, dimen = Design.shape
    beta = zeros((dimen, 1))
    beta[0:3, 0] = 1.0
    beta[3:6, 0] = -1.0
    # Take care of beta
    scale = 2.0
    grho = 1.20070904791
    gb_n = 0.228102778079
    galpha = 0.05
    gBoot = 500
    gtol = 0.000001
    gTest = 1000
    gPara = scale
    test = BootTest(grho, gb_n, galpha, gBoot, gtol, gTest, gPara)
    lb, lmin, P, Q, Q_perp = test.tightSVD(Design)
    theta = matmul(Q, matmul(Q.T, beta))
    start = 1
    biases = np.array([0.025 * i for i in range(start, 21)])
    powers = []
    for bias in biases:
        test = BootTest(grho, gb_n, galpha, gBoot, gtol, gTest, gPara, bias = bias)
        freq = test.test(Design, theta, LimComb)
        print('Coverage probability is ' + str(freq) + '\n')
        #print('Average length is' + str(curQuan) + ' For Normal case 1000-500' + '\n')
        powers.append(1.0 - freq)
        np.savetxt(fname = #, X = np.array(powers))
    #freq, avgLen, P, gam, sig = test.prdTest()
    #freq, curQuan = test.prdTest(Design, theta, LimComb)
    #print('Average mischoice prob is ' + str(P) + '\n')
    #print('Average gamma error is ' + str(gam) + '\n')
    #print('Average sigma error is ' + str(sig) + '\n')
    #test.prdTest(Design, beta, LimComb)






