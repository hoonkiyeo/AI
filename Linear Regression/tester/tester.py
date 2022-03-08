import numpy as np

if __name__=="__main__":
    test1=np.loadtxt("test.txt")
    test2=np.loadtxt("correct.txt")
    if np.all(np.isclose(test1,test2,atol=0.001)):
        print("Correct")
    else:
        print("Wrong")
