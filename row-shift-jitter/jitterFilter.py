import numpy as np

class JitterFilter(object):

    def __init__(self, array, hight, jitterRadius):
        self.array = array
        self.hight = hight
        self.jitterRadiusList = np.arange(-jitterRadius, jitterRadius+1)
    
    def linearJitter(self):
        arrayCopy = np.copy(self.array)
        self.jitterVector = np.random.choice(self.jitterRadiusList, size=self.hight)

        for idx in range(self.hight):
            arrayCopy[idx] = np.roll(arrayCopy[idx], self.jitterVector[idx])
        return arrayCopy 

    def printJitterVector(self):
        print(self.jitterVector)

if __name__ == "__main__":
    N = 7 
    testArray = np.arange(0, N**2).reshape(N,N)
    testClass = JitterFilter(testArray, N, 1)
    jitteredArray = testClass.linearJitter()

    print(testArray)
    print('/' * 30)
    print(jitteredArray)
    print('/' * 30)
    testClass.printJitterVector()
