
import torch 
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def mackey_glass(max_time, beta=0.2, gamma=0.1, n=10, theta=25):
    x = np.empty(max_time+1)
    x[0] = 1.5
    for t in range(1, max_time+1):
        x_1 = x[t-1]
        x_theta = 0 if t<theta else x[t-theta]
        x[t] = x_1+(beta*x_theta)/(1+x_theta**n)-gamma*x_1
    return x

t = np.arange(301, 1501)
x = mackey_glass(1506)
plt.plot(t, x[t], 'black')
plt.title("Mackey Glass Series")
plt.xlabel("t")
plt.ylabel("x(t)")

x = mackey_glass(1506)
noise = np.random.normal(0,0.15,size=x.shape)
x_noisy = x + noise
patterns_noisy = np.array([x_noisy[t-lag] for lag in [20,15,10,5,0]])

t = np.arange(301, 1501)
plt.plot(t, x_noisy[t], 'orange')
plt.plot(t, x[t], 'black')
plt.title("Mackey Glass Series with gaussian noise (0,0.15)")
plt.xlabel("t")
plt.ylabel("x(t)")

patterns = np.array([x_noisy[t-lag] for lag in [20,15,10,5,0]])
targets = x_noisy[t+5]
n_train = 500
n_val = 1000
n_test = 1200
patterns_train, targets_train = patterns[:,0:n_train], targets[0:n_train].reshape(1, n_train)
patterns_val, targets_val  = patterns[:,n_train:n_val], targets[n_train:n_val].reshape(1, n_val-n_train)
patterns_test, targets_test  = patterns[:,n_val:n_test], targets[n_val:n_test].reshape(1, n_test-n_val)

plt.plot(t[:n_train], targets_train[0,:], label="Training Data", color='blue')
plt.plot(t[n_train:n_val], targets_val[0,:], label="Validation Data", color='magenta')
plt.plot(t[n_val:n_test], targets_test[0,:], label="Test Data", color='gray')

class Neural_Net(nn.Module):
    # Define the network
    def __init__(self):
        super(Neural_Net, self).__init__()

        self.layer1 = nn.Linear(in_features=5, out_features=3, bias=True)  # Hidden Layer 1
        self.layer2 = nn.Linear(in_features=3, out_features=6, bias=True)  # Hidden Layer 2
        self.layer3 = nn.Linear(in_features=6, out_features=1, bias=True)  # Layer Out

        self.layer1_activation = nn.Sigmoid()  # Sigmoidal activation for hidden layer 1
        self.layer2_activation = nn.Sigmoid()  # Sigmoidal activation for hidden layer 2

    def forward(self, x):
        # First layer
        y1 = self.layer1(x)
        y1 = self.layer1_activation(y1)

        # Second layer
        y2 = self.layer2(y1)
        y2 = self.layer2_activation(y2)

        # Final output
        out = self.layer3(y2)
        return out


class Agent():
    def __init__(self):
        super(Agent,self).__init__()

        self.NN = Neural_Net()

        self.optimizer = torch.optim.SGD(self.NN.parameters(), lr=0.01, weight_decay=1e-3)
        # Adam optimizer is also cool
        self.criterion = nn.MSELoss()
    

    def forward(self, x):
        return(self.NN.forward(x))

    def backward(self, target, output):

        # Compute gradients
        self.optimizer.zero_grad() #Reboot gradient
        loss = nn.functional.mse_loss(target, output)
        loss.backward()
        #print(f"Loss: {loss}")
        #Gradient = NN.parameters().grad
        self.optimizer.step()
        return(loss)



agent = Agent()
Prediction = []
LOSS_Train =[]
LOSS_Validation =[]

# VERY IMPORTANT STEP : TRANSFORM EVERY DATA IN TENSOR
Xpatterns_train = torch.tensor(patterns_train,dtype=torch.float)
Ytargets_train = torch.tensor(targets_train,dtype=torch.float)
Xpatterns_validation = torch.tensor(patterns_val,dtype=torch.float)
Ytarget_validation = torch.tensor(targets_val,dtype=torch.float)

nb_epoch = 50
loss_over_epoch_train = []
loss_over_epoch_validation = []

iteration_nb = 10
Courbes = np.zeros((iteration_nb,patterns_train.shape[1]))
Courbe_global=[]
MSEs_train = np.zeros((iteration_nb,nb_epoch))
MSE_global_train = []
MSEs_validation = np.zeros((iteration_nb,nb_epoch))
MSE_global_validation = []

for iterations in range(iteration_nb): # STEP FOR CREATING AN AVERAGE OVER SEVERAL TRAINING
    for epoch in range(nb_epoch): # TEST OVER SEVERAL EPOCH
        LOSS_Train = []
        LOSS_Validation = []
        for id in range(len(patterns_train[0])): # BECAUSE WE HAVE A TIME SERIE AND WE HAVE TO TRAIN FOR EVERY DATAPOINT

            #Loss train set
            out_train = agent.NN.forward(Xpatterns_train[:,id])
            loss = agent.backward(Ytargets_train[:,id],out_train)
            LOSS_Train.append(loss.detach().numpy())

            #Loss Validation set
            out_validation = agent.NN.forward(Xpatterns_validation[:,id])
            LOSS_Validation.append(torch.nn.functional.mse_loss(Ytarget_validation[:,id],out_validation).detach().numpy())

        loss_over_epoch_train.append(np.mean(LOSS_Train))
        loss_over_epoch_validation.append(np.mean(LOSS_Validation))
    MSEs_train[iterations,:] = loss_over_epoch_train[-nb_epoch:]
    MSEs_validation[iterations,:] = loss_over_epoch_validation[-nb_epoch:]

    for id in range(len(patterns_val[0])):
        out = agent.NN.forward(Xpatterns_validation[:,id])
        Prediction.append(out.detach().numpy())
    print(np.shape(Courbes),np.shape(Prediction))
    Courbes[iterations,:] = np.transpose(Prediction[-500:])
    

Courbe_global = np.mean(Courbes,axis=0)
#print(np.shape(Courbe_global))
MSE_global_train = np.mean(MSEs_train,axis=0)
MSE_global_validation = np.mean(MSEs_validation,axis=0)
#print(np.shape(MSE_global_train))

plt.figure()

plt.subplot(2,1,1)
plt.plot(targets_val[0,:], 'black')
plt.plot(Courbe_global, 'red')
plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid(True)
plt.legend(["mackey_glass","prediction"])
plt.title("Prediction with config (nh1xnh2)=(3,6)")
plt.subplot(2,1,2)
#plt.plot(MSE_global_train,"blue")
#plt.plot(MSE_global_validation,"green")
plt.plot(MSEs_train[0,:],"blue")
plt.plot(MSEs_validation[0,:],"green")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend(["MSE_Train","MSE_Validation"])
plt.grid(True)
plt.savefig("nh1=3_nh2=6_noisy.png")