'''Libraries for Prototype selection'''
import numpy as np
import cvxpy as cp
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
import math as mt
from sklearn.model_selection import KFold
import sklearn.metrics
import matplotlib.pyplot as plt


class classifier():
    '''Contains functions for prototype selection'''

    def __init__(self, X, y, epsilon_, lambda_):
        '''Store data points as unique indexes, and initialize
        the required member variables eg. epsilon, lambda,
        interpoint distances, points in neighborhood'''
        self.X = X.copy()
        self.y = y.copy()
        self.epsilon_ = epsilon_
        self.lamda_ = lambda_
        self.coverMismatchErr = 0
        self.n = len(self.y)
        self.D = np.zeros((self.n, self.n))
        self.N = np.zeros((self.n, self.n))
        self.classList = self.getClassList(y)
        for i in range(self.n):
            for j in range(self.n):
                self.D[i][j] = getDistance(X[i], X[j])
                if self.D[i][j] <= self.epsilon_:
                    self.N[i][j] = 1
        self.objectiveValue = 0
        self.labelList = []
        for i in range(max(y) + 1):
            self.labelList.append([])


    '''Implement modules which will be useful in the train_lp() function
    for example
    1) operations such as intersection, union etc of sets of datapoints
    2) check for feasibility for the randomized algorithm approach
    3) compute the objective value with current prototype set
    4) fill in the train_lp() module which will make 
    use of the above modules and member variables defined by you
    5) any other module that you deem fit and useful for the purpose.'''

    def getClassList(self, y):
        classList = []
        listSize = max(y) + 1
        for i in range(listSize):
            classList.append([])
        for i in range(len(y)):
            tmpList = classList[y[i]]
            tmpList.append(i)
        return classList


    def train_lp(self, verbose=False):
        '''Implement the linear programming formulation
        and solve using cvxpy for prototype selection'''
        nTotal = len(self.y)
        for label in range(len(self.classList)):
            n = len(self.classList[label])

            # coefficients
            C1 = np.zeros((1, nTotal))
            for j in range(nTotal):
                C1[0][j] = self.getCost(j, label) + self.lamda_
            C2 = np.ones((1, n))
            B1 = -self.N[self.classList[label]]
            B2 = -np.identity(n)
            # variables
            Alpha = cp.Variable(shape=(nTotal, 1))
            Xi = cp.Variable(shape=(n, 1))

            objective = cp.Minimize(C1 * Alpha + C2 * Xi)
            constraints = [Alpha <= 1,
                           Alpha >= 0,
                           Xi >= 0,
                           B1 * Alpha + B2 * Xi <= -1]
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver=cp.ECOS)
            [A, S, obj] = self.rounding(Alpha.value, Xi.value, result, B1, B2, C1, C2)
            indexes = [k for k, val in enumerate (A) if val == 1]
            for m in range(len(indexes)):
                self.labelList[label].append(indexes[m])


    def rounding(self, alpha, xi, result, B1, B2, C1, C2):
        A = np.zeros((len(alpha), 1))
        S = np.zeros((len(xi), 1))
        obj = 0
        isFeasible = False
        while isFeasible != True:
            for i in range(round(2 * mt.log2(len(xi)))):
                for j in range(len(A)):
                    if alpha[j] > 1:
                        alpha[j] = 1
                    A[j] = max(A[j], np.random.binomial(1, abs(alpha[j])))
                for k in range(len(S)):
                    if xi[k] > 1:
                        xi[k] = 1
                    S[k] = max(S[k], np.random.binomial(1, abs(xi[k])))
                obj = np.dot(C1, A) + np.dot(C2, S)
                if (np.dot(B1, A) + np.dot(B2, S) <= -np.ones((len(B1), 1))).all() and obj <= round(
                        2 * mt.log2(len(xi))) * result:
                    isFeasible = True
        self.objectiveValue += obj
        return A, S, obj


    def getCost(self, curIndex, label):
        count = 0
        neighbors = self.N[curIndex]
        for i in range(len(neighbors)):
            if neighbors[i] == 1 and self.y[i] != label:
                count = count + 1
        return count


    def getSetCoverErr(self):
        return self.coverMismatchErr


    def countPrototype(self):
        count = 0
        for i in range(len(self.labelList)):
            for j in range(len(self.labelList[i])):
                count += 1
        return count


    def objective_value(self):
        '''Implement a function to compute the objective value of the integer optimization
        problem after the training phase'''
        return self.objectiveValue


    def predict(self, instances):
        '''Predicts the label for an array of instances using the framework learnt'''
        labels = np.ones((len(instances), 1))
        for i in range(len(instances)):
            labels[i] = -1
            [misclassication, labels[i]] = self.getNearestLabel(instances[i])
        return labels


    def predictCover(self, instances, targets):
        '''Predicts the cover err for an array of instances using the framework learnt'''
        labels = np.ones((len(instances), 1))
        coverMismatch = 0
        coverMismatchTotal = len(instances)
        for i in range(len(instances)):
            labels[i] = -1
            [misclassication, labels[i]] = self.getNearestLabel(instances[i])
            if labels[i] != targets[i]:
                misclassication = 1
            coverMismatch += misclassication
        self.coverMismatchErr = coverMismatch / coverMismatchTotal
        return labels


    def getNearestLabel(self, instance):
        label = -1
        misclassication = 0
        distance = float('inf')
        for i in range(len(self.labelList)):
            for j in range(len(self.labelList[i])):
                curDistance = getDistance(instance, self.X[self.labelList[i][j]])
                if curDistance < distance:
                    distance = curDistance
                    label = i
        if distance > self.epsilon_:
            misclassication = 1
        return misclassication, label


def getDistance(x1, x2):
    return mt.sqrt(np.sum((x1 - x2) ** 2))


def cross_val(data, target, epsilon_, lambda_, k, verbose):
    '''Implement a function which will perform k fold cross validation
    for the given epsilon and lambda and returns the average test error and number of prototypes'''
    kf = KFold(n_splits=k, random_state=42)
    score = 0
    prots = 0
    obj_val = 0
    coverErr = 0
    for train_index, test_index in kf.split(data):
        '''implement code to count the total number of prototypes learnt and store it in prots'''
        ps = classifier(data[train_index], target[train_index], epsilon_, lambda_)
        ps.train_lp(verbose)
        obj_val += ps.objective_value()
        prots += ps.countPrototype()
        score += sklearn.metrics.accuracy_score(target[test_index], ps.predict(data[test_index]))
        ps.predictCover(data[test_index], target[test_index])
        coverErr += ps.getSetCoverErr()
    score /= k
    prots /= k
    obj_val /= k
    coverErr /= k

    return score, coverErr, prots, obj_val


def test1():
    print("start testing 1")
    iris_data = load_iris()
    data = iris_data.data
    target = iris_data.target
    print(data)
    print(target)

    [score, coverErr, prots, obj_val] = cross_val(data, target, 1, 1.0 / len(target), 4, False)
    print("cross_val result")
    print(score, coverErr, prots, obj_val)



def test2():
    print("start testing 2")
    breast_cancer_data = load_breast_cancer()
    data = breast_cancer_data.data
    target = breast_cancer_data.target
    n = len(data)
    distanceArray = []
    coverErrArray= []
    testErrArray = []
    epsilonArray = []
    protsArray = []
    for i in range(n):
        for j in range(n):
            if i > j:
                distanceArray.append(getDistance(data[i], data[j]))

    for k in range(2, 41, 2):
        epsilon = np.percentile(distanceArray, k)
        epsilonArray.append(epsilon)
        [score, coverErr, prots, obj_val] = cross_val(data, target, epsilon, 1.0 / len(target), 4, False)
        coverErrArray.append(coverErr)
        testErrArray.append(1 - score)
        protsArray.append(prots)

    plt.figure(1)
    plt.plot(epsilonArray, testErrArray, 'k', label='test error')
    plt.plot(epsilonArray, coverErrArray, 'r', label='cover error')
    plt.xlabel('epsilon')
    plt.title('Breast Cancer: Test error and Cover error vs epsilon')
    plt.legend()

    plt.figure(2)
    plt.plot(epsilonArray, protsArray, 'y')
    plt.xlabel('epsilon')
    plt.ylabel('prototype number')
    plt.title('Breast Cancer: Prototype number vs epsilon')

    plt.figure(3)
    plt.plot(protsArray, testErrArray, 'k', label='test error')
    plt.plot(protsArray, coverErrArray, 'r', label='cover error')
    plt.legend()
    plt.xlabel('prototype number')
    plt.ylabel('error')
    plt.title('Breast Cancer: Test error and Cover error vs prototypes')
    plt.show()

def test3():
    print("start testing 3")
    digit_data = load_digits()
    data = digit_data.data
    target = digit_data.target
    epsilonArray = []
    objValueArray = []
    for epsilon in range(20, 41, 2):
        epsilonArray.append(epsilon)
        ps = classifier(data, target, epsilon, 1.0 / len(target))
        ps.train_lp()
        obj_val = ps.objective_value()[0]
        objValueArray.append(obj_val)

    print(epsilonArray, objValueArray)
    plt.figure(1)
    plt.plot(epsilonArray, objValueArray, 'k', label='objective value')
    plt.legend()
    plt.xlabel('epsilon')
    plt.ylabel('objective value')
    plt.title('Digits data: Objective value vs epsilon')
    plt.show()

def test4():
    print("test modification")
    iris_data = load_iris()
    data = iris_data.data
    target = iris_data.target
    n = len(data)
    distanceArray = []
    coverErrArray = []
    testErrArray = []
    epsilonArray = []
    protsArray = []
    for i in range(n):
        for j in range(n):
            if i > j:
                distanceArray.append(getDistance(data[i], data[j]))

    for k in range(2, 41, 2):
        epsilon = np.percentile(distanceArray, k)
        epsilonArray.append(epsilon)
        [score, coverErr, prots, obj_val] = cross_val(data, target, epsilon, 1.0 / len(target), 4, False)
        coverErrArray.append(coverErr)
        testErrArray.append(1 - score)
        protsArray.append(prots)

    plt.figure(1)
    plt.plot(epsilonArray, testErrArray, 'k', label='test error')
    plt.plot(epsilonArray, coverErrArray, 'r', label='cover error')
    plt.xlabel('epsilon')
    plt.title('Breast Cancer: Test error and Cover error vs epsilon')
    plt.legend()

    plt.figure(2)
    plt.plot(epsilonArray, protsArray, 'y')
    plt.xlabel('epsilon')
    plt.ylabel('prototype number')
    plt.title('Breast Cancer: Prototype number vs epsilon')

    plt.figure(3)
    plt.plot(protsArray, testErrArray, 'k', label='test error')
    plt.plot(protsArray, coverErrArray, 'r', label='cover error')
    plt.legend()
    plt.xlabel('prototype number')
    plt.ylabel('error')
    plt.title('iris data: Test error and Cover error vs prototypes')
    plt.show()


if __name__ == "__main__":
    test4()
