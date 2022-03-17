from itertools import count
import numpy as np
import matplotlib.pyplot as plt
import random

from numpy.random import randn as rd

def new_data(mA, mB, sigmaA, sigmaB, n, data_type, rng):

    if data_type == "linear":
        classA = rng.standard_normal((2, n))*sigmaA + np.repeat(mA, n, axis=1)
        classB = rng.standard_normal((2, n))*sigmaB + np.repeat(mB, n, axis=1)

    elif data_type == "nonlinear":
        N = n+n

        #First coord classA
        classA00 = rng.standard_normal((1, int(n / 2))) * sigmaA + mA[0]
        classA01 = rng.standard_normal((1, int(n / 2))) * sigmaA - mA[0]
        classA0 = np.concatenate((classA00, classA01), axis=1) #left and right clusters
        #Second coord classA
        classA1 = rng.standard_normal((1, n))*sigmaA + mA[1]

        classA = np.concatenate((classA0, classA1), axis=0)
        classB = rng.standard_normal((2, n))*sigmaB + np.repeat(mB, n, axis=1)      


    X2D = np.concatenate((classA, classB), axis = 1)
    X = np.concatenate((X2D, np.ones((1, n+n))), axis = 0)

    T = np.concatenate((np.ones(n), -np.ones(n)))

    shuffler = rng.permutation(n+n)
    X = X[:, shuffler]
    T = T[shuffler]

    return X, T


def plot_data(X, T):
    plt.scatter(X[0, :], X[1, :], c=T)
    plt.xlim([-1.7, 1.7])
    plt.ylim([-1.7, 1.7])
    plt.grid()


def plot_boundaries(W):
    x = np.linspace(-1, 3, 100)
    y = - ( W[2] + W[0]*x ) / W[1]
    # y = - ( W[0]*x ) / W[1]

    plt.plot(x, y, color='red')


"""
def subsample(n, scenario, rng, sigmaA, sigmaB, mA, mB):
    N = n + n

    # First coord classA
    classA00 = rng.standard_normal((1, int(n / 2))) * sigmaA + mA[0]
    classA01 = rng.standard_normal((1, int(n / 2))) * sigmaA - mA[0]
    classA0 = np.concatenate((classA00, classA01), axis=1)  # left and right clusters   #X-axis
    # Second coord classA
    classA1 = rng.standard_normal((1, n)) * sigmaA + mA[1]      #Y-axis
    
    classB = rng.standard_normal((2, n)) * sigmaB + np.repeat(mB, n, axis=1)

    if scenario == 1:
        a,b = classA0.shape
        classA0=list(classA0[0])
        classA1=list(classA1[0])
        indice = round(b*0.25)
        classA_validation = [[],[]]
        
        
        for i in range(indice):

            integer = random.randrange(0,len(classA0),1)

            classA_validation[0].append(classA0[integer])
            classA_validation[1].append(classA1[integer])

            classA0 = np.delete(classA0,[integer])
            classA1 = np.delete(classA1,[integer])

        classB0 = classB[0][:]
        classB1 = classB[1][:]


        classB_validation = [[],[]]

        for i in range(indice):
            integer = random.randrange(0,len(classB0),1)
            classB_validation[0].append(classB0[integer])
            classB_validation[1].append(classB1[integer])
            classB0 = np.delete(classB0,[integer])
            classB1 = np.delete(classB1,[integer])
        
    
        training = np.array([list(classA0) + list(classB0), list(classA1) + list(classB1)])


        validation = [list(classA_validation[0])+list(classB_validation[0]), list(classA_validation[1])+list(classB_validation[1])]
        validation = np.array(validation)


        T_validation = np.concatenate((np.ones(len(classA_validation[0])), -np.ones(len(classB_validation[0]))))
        T_training = np.concatenate((np.ones(len(classA0)), -np.ones(len(classB0))))


    elif scenario == 2:
        a,b = classA0.shape
        classA0=list(classA0[0])
        classA1=list(classA1[0])
        
        indice = round(b*0.5)
        classA_validation = [[],[]]
        
        for i in range(indice):
            integer = random.randrange(0,len(classA0),1)
            classA_validation[0].append(classA0[integer])
            classA_validation[1].append(classA1[integer])
            classA0 = np.delete(classA0,[integer])
            classA1 = np.delete(classA1,[integer])

        classA_validation = np.array(classA_validation)
        classA_training = np.array([classA0,classA1])
                

        classB = np.array(rng.standard_normal((2, n)) * sigmaB + np.repeat(mB, n, axis=1))
        classB0 = classB[0]
        classB1 = classB[1]

        validation = classA_validation

        training = np.array([list(classA_training[0])+list(classB0), list(classA_training[1])+list(classB1)])

        
        T_validation = np.ones(indice)
        T_training = np.concatenate((np.ones(len(classA_training[0])), -np.ones(len(classB0))))


    elif scenario == 3:
        l = len(classA0[0])
        classA0=list(classA0[0])
        classA1=list(classA1[0])
        subset_negative = [[],[]]
        training_A = [[],[]]
        count_negative = 0
        subset_positive = [[],[]]
        count_positive = 0
        for i in range(l):
            
            if classA0[i]<0:
                count_negative += 1  
                if count_negative%5 != 0:   #every 1/5 times
                    subset_negative[0].append(classA0[i])
                    subset_negative[1].append(classA1[i])
                else : 
                    training_A[0].append(classA0[i])
                    training_A[1].append(classA1[i])
            
            else:
                count_positive +=1
                if count_negative%5 != 0:  #every 4/5 times:
                    subset_positive[0].append(classA0[i])
                    subset_positive[1].append(classA1[i])
                    
                else:
                    training_A[0].append(classA0[i])
                    training_A[1].append(classA1[i])
        

        classB = np.array(rng.standard_normal((2, n)) * sigmaB + np.repeat(mB, n, axis=1))
        classB0 = classB[0]
        classB1 = classB[1]

        validation = np.array([subset_negative[0] + subset_positive[0], subset_negative[1] + subset_positive[1]])
        training = np.array([list(training_A[0])+list(classB0), list(training_A[1])+list(classB1)])

        T_validation = np.ones(len(subset_negative[0]) + len(subset_positive[0]))
        T_training = np.concatenate((np.ones(len(training_A[0])), -np.ones(len(classB0))))




    else:
        print("Please enter a correct scenario : 1, 2 or 3.")

    at,bt = training.shape
    av,bv = validation.shape

    validation = np.concatenate((validation,np.ones((1,bv))),axis=0)
    training = np.concatenate((training,np.ones((1,bt))),axis=0)                #adding the bias

    validation= np.array(validation)
    training = np.array(training)

    return training, T_training, validation, T_validation
"""

def subsample(n, scenario, rng, sigmaA, sigmaB, mA, mB):
    N = n + n

    # First coord classA
    classA00 = rng.standard_normal((1, int(n / 2))) * sigmaA + mA[0]
    classA01 = rng.standard_normal((1, int(n / 2))) * sigmaA - mA[0]
    classA0 = np.concatenate((classA00, classA01), axis=1)  # left and right clusters   #X-axis
    # Second coord classA
    classA1 = rng.standard_normal((1, n)) * sigmaA + mA[1]      #Y-axis
    
    classB = rng.standard_normal((2, n)) * sigmaB + np.repeat(mB, n, axis=1)

    if scenario == 1:
        a,b = classA0.shape
        classA0=list(classA0[0])
        classA1=list(classA1[0])
        indice = round(b*0.25)
        classA_t = [[],[]]
        
        
        for i in range(indice):

            integer = random.randrange(0,len(classA0),1)

            classA_t[0].append(classA0[integer])
            classA_t[1].append(classA1[integer])

            classA0 = np.delete(classA0,[integer])
            classA1 = np.delete(classA1,[integer])

        classB0 = classB[0][:]
        classB1 = classB[1][:]


        classB_t= [[],[]]

        for i in range(indice):
            integer = random.randrange(0,len(classB0),1)
            classB_t[0].append(classB0[integer])
            classB_t[1].append(classB1[integer])
            classB0 = np.delete(classB0,[integer])
            classB1 = np.delete(classB1,[integer])
        
    
        validation_set = np.array([list(classA0) + list(classB0), list(classA1) + list(classB1)])


        train_set = [list(classA_t[0])+list(classB_t[0]), list(classA_t[1])+list(classB_t[1])]
        train_set = np.array(train_set)


        T_training = np.concatenate((np.ones(len(classA_t[0])), -np.ones(len(classB_t[0]))))
        T_validation = np.concatenate((np.ones(len(classA0)), -np.ones(len(classB0))))


    elif scenario == 2:
        a,b = classA0.shape
        classA0=list(classA0[0])
        classA1=list(classA1[0])
        
        indice = round(b*0.5)
        classA_t = [[],[]]
        
        for i in range(indice):
            integer = random.randrange(0,len(classA0),1)
            classA_t[0].append(classA0[integer])
            classA_t[1].append(classA1[integer])
            classA0 = np.delete(classA0,[integer])
            classA1 = np.delete(classA1,[integer])

        classA_t = np.array(classA_t)
        classA_v = np.array([classA0,classA1])
                

        classB = np.array(rng.standard_normal((2, n)) * sigmaB + np.repeat(mB, n, axis=1))
        classB0 = classB[0]
        classB1 = classB[1]

        train_set = classA_t

        validation_set = np.array([list(classA_v[0])+list(classB0), list(classA_v[1])+list(classB1)])

        
        T_training= np.ones(indice)
        T_validation = np.concatenate((np.ones(len(classA_v[0])), -np.ones(len(classB0))))


    elif scenario == 3:
        l = len(classA0[0])
        classA0=list(classA0[0])
        classA1=list(classA1[0])
        subset_negative = [[],[]]
        validation_A = [[],[]]
        count_negative = 0
        subset_positive = [[],[]]
        count_positive = 0
        for i in range(l):
            
            if classA0[i]<0:
                count_negative += 1  
                if count_negative%5 != 0:   #every 1/5 times
                    subset_negative[0].append(classA0[i])
                    subset_negative[1].append(classA1[i])
                else : 
                    validation_A[0].append(classA0[i])
                    validation_A[1].append(classA1[i])
            
            else:
                count_positive +=1
                if count_negative%5 != 0:  #every 4/5 times:
                    subset_positive[0].append(classA0[i])
                    subset_positive[1].append(classA1[i])
                    
                else:
                    validation_A[0].append(classA0[i])
                    validation_A[1].append(classA1[i])
        

        classB = np.array(rng.standard_normal((2, n)) * sigmaB + np.repeat(mB, n, axis=1))
        classB0 = classB[0]
        classB1 = classB[1]

        train_set = np.array([subset_negative[0] + subset_positive[0], subset_negative[1] + subset_positive[1]])
        validation_set = np.array([list(validation_A[0])+list(classB0), list(validation_A[1])+list(classB1)])

        T_training = np.ones(len(subset_negative[0]) + len(subset_positive[0]))
        T_validation = np.concatenate((np.ones(len(validation_A[0])), -np.ones(len(classB0))))




    else:
        print("Please enter a correct scenario : 1, 2 or 3.")

    at,bt = train_set.shape
    av,bv = validation_set.shape

    validation_set = np.concatenate((validation_set,np.ones((1,bv))),axis=0)
    train_set = np.concatenate((train_set,np.ones((1,bt))),axis=0)                #adding the bias

    validation_set= np.array(validation_set)
    train_set = np.array(train_set)

    return train_set, T_training, validation_set, T_validation


def missclassified_rate(X, T, W):
    predictions = np.where(W@X>0, 1, 0)
    T_01 = np.where(T==1, 1, 0)
    accuracy = np.zeros((2, 2))
    for i in range(X.shape[1]):
        target = T_01[i]
        if predictions[i] == target:
            accuracy[target, target] += 1
        elif predictions[i] > target:
            accuracy[0, 1] += 1
        else:
            accuracy[1, 0] += 1
    accuracy[0, :] *= 100/(accuracy[0, 0] + accuracy[0, 1])
    accuracy[1, :] *= 100/(accuracy[1, 0] + accuracy[1, 1])

    return accuracy


"""def compute_accuracy(out, targets):
    h,l = out.shape
    size = h*l
    predictions = np.resize(np.where(out>0, 1, 0), (size, 1))
    T_01 = np.where(targets==1, 1, 0)
    accuracy = np.zeros((2, 2))
    for i in range(out.shape[1]):
        t = T_01[i]
        if predictions[i] == t:
            accuracy[t, t] += 1
        elif predictions[i] > t: #t = 0 and prediction = 1
            accuracy[0, 1] += 1
        else:                    #t = 1 and prediction = 0
            accuracy[1, 0] += 1
    if (accuracy[0, 0] + accuracy[0, 1]) != 0:
        accuracy[0, :] *= 100/(accuracy[0, 0] + accuracy[0, 1])
    if (accuracy[1, 0] + accuracy[1, 1]) != 0:
        accuracy[1, :] *= 100/(accuracy[1, 0] + accuracy[1, 1])

    return accuracy"""

def compute_accuracy(out, targets):
    predictions = np.resize(np.where(out>0, 1, 0), (200, 1))
    T_01 = np.where(targets==1, 1, 0)
    accuracy = np.zeros((2, 2))
    for i in range(out.shape[1]):
        t = T_01[i]
        if predictions[i] == t:
            accuracy[t, t] += 1
        elif predictions[i] > t: #t = 0 and prediction = 1
            accuracy[0, 1] += 1
        else:                    #t = 1 and prediction = 0
            accuracy[1, 0] += 1
    if (accuracy[0, 0] + accuracy[0, 1]) != 0:
        accuracy[0, :] *= 100/(accuracy[0, 0] + accuracy[0, 1])
    if (accuracy[1, 0] + accuracy[1, 1]) != 0:
        accuracy[1, :] *= 100/(accuracy[1, 0] + accuracy[1, 1])

    return accuracy


def backprop_training(patterns, ndata, targets, Nhidden, alpha, eta, epochs_nb, accuracy):

    # Nhidden : number of nodes in the hidden layer
    # W       : Weight for the first layer
    # V       : Weight for the second layer 
    # alpha   : Control factor (weighting of old weight importance)
    # eta     : Learning Rate



    W = np.random.randn(Nhidden, patterns.shape[0]) #bias included in patterns = X
    V = np.random.randn(1, Nhidden+1) #adding bias to hidden layer
    dw = np.zeros((Nhidden, patterns.shape[0]))
    dv = np.zeros((1, Nhidden+1))


    accuraciesA = []
    accuraciesB = []
    mse = []
    #mem_mse = np.inf

    for epoch in range(epochs_nb):
        
        #Step 1 - Forward Pass
        hin = W @ patterns
        hout = np.concatenate((2 / ( 1 + np.exp(-hin) ) - 1, np.ones((1, np.shape(patterns)[1]))), axis=0) # Out of the first layer
        oin = V @ hout
        out = 2 / ( 1 + np.exp(-oin) ) - 1 # Out of the second layer)

        #Step 2 - Backward Pass
        delta_o = 0.5 * (out - targets) * ((1 + out) * (1 - out))
        delta_h = 0.5 * (V.T @ delta_o) * ((1 + hout) * (1 - hout))
        delta_h = delta_h[:Nhidden, :]

        # #Step 3 - Weight Update
        dw = alpha * dw - ( 1 - alpha ) * ( delta_h @ patterns.T )
        dv = alpha * dv - ( 1 - alpha ) * ( delta_o @ hout.T )
        W += dw * eta
        V += dv * eta

        #eta = eta/(epoch+1)

        if accuracy == True :
            ## ACCURACY
            current_accuracy = compute_accuracy(out, targets)
            accuraciesA.append(current_accuracy[1, 1])
            accuraciesB.append(current_accuracy[0, 0])

        ##MSE
        #if mem_mse > 0.7*np.mean((out-targets)**2):
        mse.append(np.mean((out-targets)**2))
            #mem_mse = np.mean((out-targets)**2)
        #else:
           # break
        

    return mse, accuraciesA, accuraciesB, W, V

"""
def backprop_validation(patterns, ndata, targets, Nhidden, alpha, eta, epochs_nb, W, V):

    # Nhidden : number of nodes in the hidden layer
    # W       : Weight for the first layer
    # V       : Weight for the second layer 
    # alpha   : Control factor (weighting of old weight importance)
    # eta     : Learning Rate


    dw = np.zeros((Nhidden, patterns.shape[0]))
    dv = np.zeros((1, Nhidden+1))

    # accuracies = np.zeros((epochs_nb, 2, 2))
    accuraciesA = []
    accuraciesB = []
    mse = []

    for epoch in range(epochs_nb):

        #Step 1 - Forward Pass
        hin = W @ patterns
        hout = np.concatenate((2 / ( 1 + np.exp(-hin) ) - 1, np.ones((1, np.shape(patterns)[1]))), axis=0) # Out of the first layer
        print(hout.shape)
        print(V.shape)
        oin = V @ hout
        out = 2 / ( 1 + np.exp(-oin) ) - 1 # Out of the second layer

        #Step 2 - Backward Pass
        delta_o = 0.5 * (out - targets) * ((1 + out) * (1 - out))
        delta_h = 0.5 * (V.T @ delta_o) * ((1 + hout) * (1 - hout))
        delta_h = delta_h[:Nhidden, :]

        # #Step 3 - Weight Update
        dw = alpha * dw - ( 1 - alpha ) * ( delta_h @ patterns.T )
        dv = alpha * dv - ( 1 - alpha ) * ( delta_o @ hout.T )
        W += dw * eta
        V += dv * eta

        ## ACCURACY
        current_accuracy = compute_accuracy(out, targets)
        accuraciesA.append(current_accuracy[1, 1])
        accuraciesB.append(current_accuracy[0, 0])

        ##MSE
        mse.append(np.mean((out-targets)**2))

    return mse, accuraciesA, accuraciesB

"""

def backprop_trainvalide(train_set, validation_set, ndata, train_targets, validation_targets, Nhidden, alpha, eta, epochs_nb):

    # Nhidden : number of nodes in the hidden layer
    # W       : Weight for the first layer
    # V       : Weight for the second layer 
    # alpha   : Control factor (weighting of old weight importance)
    # eta     : Learning Rate

    """print('-------------')
    print(train_set.shape)
    print(validation_set.shape)
    print(train_targets.shape)
    print(validation_targets.shape)
    print('-------------')"""


    W = np.random.randn(Nhidden, train_set.shape[0]) #bias included in trainset = X
    V = np.random.randn(1, Nhidden+1) #adding bias to hidden layer
    dw = np.zeros((Nhidden, train_set.shape[0]))
    dv = np.zeros((1, Nhidden+1))

    accuraciesA_t = []
    accuraciesB_t = []
    mse_t = []
    accuraciesA_v = []
    accuraciesB_v = []
    mse_v = []

    for epoch in range(epochs_nb):

        #Step 1 - Forward Pass
        hin_t = W @ train_set
        hout_t = np.concatenate((2 / ( 1 + np.exp(-hin_t) ) - 1, np.ones((1, np.shape(train_set)[1]))), axis=0) # Out of the first layer
        oin_t = V @ hout_t
        out_t = 2 / ( 1 + np.exp(-oin_t) ) - 1 # Out of the second layer

        hin_v = W @ validation_set
        hout_v = np.concatenate((2 / ( 1 + np.exp(-hin_v) ) - 1, np.ones((1, np.shape(validation_set)[1]))), axis=0) # Out of the first layer
        oin_v = V @ hout_v
        out_v = 2 / ( 1 + np.exp(-oin_v) ) - 1 # Out of the second layer
        """print(out_v.shape)
        print(validation_targets.shape)"""

        #Step 2 - Backward Pass
        delta_o = 0.5 * (out_t - train_targets) * ((1 + out_t) * (1 - out_t))
        delta_h = 0.5 * (V.T @ delta_o) * ((1 + hout_t) * (1 - hout_t))
        delta_h = delta_h[:Nhidden, :]

        # #Step 3 - Weight Update
        dw = alpha * dw - ( 1 - alpha ) * ( delta_h @ train_set.T )
        dv = alpha * dv - ( 1 - alpha ) * ( delta_o @ hout_t.T )
        W += dw * eta
        V += dv * eta

        ## ACCURACY & MSE : TRAINING SET
        current_accuracy_t = compute_accuracy(out_t, train_targets)
        accuraciesA_t.append(current_accuracy_t[1, 1])
        accuraciesB_t.append(current_accuracy_t[0, 0])

        mse_t.append(np.mean((out_t-train_targets)**2))

        ## ACCURACY & MSE : VALIDATION SET
        current_accuracy_v = compute_accuracy(out_v, validation_targets)
        accuraciesA_v.append(current_accuracy_v[1, 1])
        accuraciesB_v.append(current_accuracy_v[0, 0])
        
        """print(out_v.shape)
        print(validation_targets.shape)"""
        print(current_accuracy_t)

        mse_v.append(np.mean((out_v-validation_targets)**2))
        

    return (mse_t, mse_v), ((accuraciesA_t, accuraciesB_t), (accuraciesA_v, accuraciesB_v)), W, V

def forward_pass(V, W, patterns):
    hin = W @ patterns
    hout = np.concatenate((2 / ( 1 + np.exp(-hin) ) - 1, np.ones((1, np.shape(patterns)[1]))), axis=0) # Out of the first layer
    oin = V @ hout
    out = 2 / ( 1 + np.exp(-oin) ) - 1 # Out of the second layer)
    return(out)


def subsample_function_approx(patterns_used,targets_used,pourcentage):
    random.seed(0)
    random.shuffle(patterns_used[0])
    random.seed(0)
    random.shuffle(patterns_used[1])
    random.seed(0)
    random.shuffle(targets_used)

    l = len(targets_used)
    indice = round(l*pourcentage)

    patterns_training = np.array([patterns_used[0][:indice], patterns_used[1][:indice], patterns_used[2][:indice]])
    targets_training = np.array(targets_used[:indice])
    patterns_validation = np.array([patterns_used[0][indice:], patterns_used[1][indice:], patterns_used[2][indice:]])
    targets_validation = np.array(targets_used[indice:])

    return patterns_training, targets_training, patterns_validation, targets_validation


def backprop_trainvalide_funct_approx(train_set, validation_set, ndata, train_targets, validation_targets, Nhidden, alpha, eta, epochs_nb):

        # Nhidden : number of nodes in the hidden layer
    # W       : Weight for the first layer
    # V       : Weight for the second layer 
    # alpha   : Control factor (weighting of old weight importance)
    # eta     : Learning Rate


    W = np.random.randn(Nhidden, train_set.shape[0]) #bias included in trainset = X
    V = np.random.randn(1, Nhidden+1) #adding bias to hidden layer
    dw = np.zeros((Nhidden, train_set.shape[0]))
    dv = np.zeros((1, Nhidden+1))

    mse_t = []
    mse_v = []

    for epoch in range(epochs_nb):

        #Step 1 - Forward Pass
        hin_t = W @ train_set
        hout_t = np.concatenate((2 / ( 1 + np.exp(-hin_t) ) - 1, np.ones((1, np.shape(train_set)[1]))), axis=0) # Out of the first layer
        oin_t = V @ hout_t
        out_t = 2 / ( 1 + np.exp(-oin_t) ) - 1 # Out of the second layer

        hin_v = W @ validation_set
        hout_v = np.concatenate((2 / ( 1 + np.exp(-hin_v) ) - 1, np.ones((1, np.shape(validation_set)[1]))), axis=0) # Out of the first layer
        oin_v = V @ hout_v
        out_v = 2 / ( 1 + np.exp(-oin_v) ) - 1 # Out of the second layer
    

        #Step 2 - Backward Pass
        delta_o = 0.5 * (out_t - train_targets) * ((1 + out_t) * (1 - out_t))
        delta_h = 0.5 * (V.T @ delta_o) * ((1 + hout_t) * (1 - hout_t))
        delta_h = delta_h[:Nhidden, :]

        # #Step 3 - Weight Update
        dw = alpha * dw - ( 1 - alpha ) * ( delta_h @ train_set.T )
        dv = alpha * dv - ( 1 - alpha ) * ( delta_o @ hout_t.T )
        W += dw * eta
        V += dv * eta

        #eta = eta/(epoch+1)

        ## MSE : TRAINING SET

        mse_t.append(np.mean((out_t-train_targets)**2))

        ## MSE : VALIDATION SET

        mse_v.append(np.mean((out_v-validation_targets)**2))
        

    return (mse_t, mse_v), W, V