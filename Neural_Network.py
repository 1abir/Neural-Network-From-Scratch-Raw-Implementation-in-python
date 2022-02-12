# %%
import numpy as np
import pandas as pd
import pickle

# %%
class Layer:
    def __init__(self,fan_in,fan_out):
        self.weight = np.random.randn(fan_out,fan_in).T
        self.bias = np.zeros(shape=(1,fan_out))
        self.activation_function = self.sigmoid
        self.activation_function_derivative = self.sigmoid_derivative

    def forward(self,inputs):
        return  np.dot(inputs,self.weight) + self.bias
    

    def sigmoid(self, val):
        return 1 / (1 + np.exp(-val))

    def sigmoid_derivative(self, val):
        return np.exp(-val)/(1 + np.exp(-val))**2

    
    

# %%
class NeuralNetwork:
    def __init__(self,hidden_layers) -> None:
        self.network_structure = hidden_layers
        self.hidden_layers = []
        for i in range(len(hidden_layers)-1):
            self.hidden_layers.append(Layer(hidden_layers[i],hidden_layers[i+1]))

        # self.hidden_layers[-1].activation_function = self.hidden_layers[-1].softmax
        # self.hidden_layers[-1].activation_function_derivative = self.hidden_layers[-1].crossEntropyloss

    def forward(self,inputs):
        self.layer_outputs = []
        self.layer_outputs_activation = []

        self.layer_outputs_activation.append(inputs)

        self.layer_outputs.append(self.hidden_layers[0].forward(inputs))
        self.layer_outputs_activation.append(self.hidden_layers[0].activation_function(self.layer_outputs[0]))
        
        for i in range(1,len(self.hidden_layers)):
            layer_output = self.hidden_layers[i].forward(self.layer_outputs_activation[i])
            self.layer_outputs.append(layer_output)
            
            layer_output = self.hidden_layers[i].activation_function(layer_output)
            self.layer_outputs_activation.append(layer_output)

    def delta(self, layer_no, delta_i_1, y_train):

        derivative = self.hidden_layers[layer_no].activation_function_derivative(self.layer_outputs[layer_no])
        if layer_no == len(self.hidden_layers)-1:
            delta_i = self.layer_outputs_activation[layer_no +1] - y_train
        else:
            delta_i = np.dot(delta_i_1,self.hidden_layers[layer_no + 1].weight.T)
        delta_i = derivative * delta_i
        
        return delta_i

    def backward(self, y_train, alpha):
        nabla = None
        for i in range(len(self.layer_outputs)-1, -1, -1):
            nabla = self.delta(i, nabla, y_train)
            self.hidden_layers[i].weight -= alpha * np.dot(self.layer_outputs_activation[i].T, nabla)
            self.hidden_layers[i].bias -= alpha * np.sum(nabla, axis=0, keepdims=True)

    def print_weight(self):
        for i in range(len(self.hidden_layers)-1,-1,-1):
            print(self.hidden_layers[i].weight.T)

    def print_activationlayer_outputs_activation(self):
        for i in range(len(self.layer_outputs_activation)-1,-1,-1):
            print(self.layer_outputs_activation[i])

    def cost(self, y):
        error = ((self.layer_outputs_activation[-1] - y) ** 2)
        error = np.sum(error)
        return error / len(y[0])

    def train(self,train_x,train_y,epocs=1000,alpha=.01):
        y = pd.get_dummies(train_y).values
        for i in range(epocs):
            self.forward(train_x)
            self.backward(y,alpha)
            # if i % 100 == 0:
            #     print(self.cost(y))   

            alpha -= (alpha - .001) / 1000

    def predict(self,inputs):
        layer_outputs = []
        layer_outputs_activation = []

        layer_outputs_activation.append(inputs)

        layer_outputs.append(self.hidden_layers[0].forward(inputs))
        layer_outputs_activation.append(self.hidden_layers[0].activation_function(layer_outputs[0]))
        
        for i in range(1,len(self.hidden_layers)):
            layer_output = self.hidden_layers[i].forward(layer_outputs_activation[i])
            layer_outputs.append(layer_output)
            
            layer_output = self.hidden_layers[i].activation_function(layer_output)
            layer_outputs_activation.append(layer_output)

        retval = np.argmax(layer_outputs_activation[-1],axis=1) + 1
        
        return retval

    def accuracy(self,y_pred,test_y):
        return np.equal(y_pred,test_y).mean()

# %%
train_df = pd.read_csv('trainNN.txt',delim_whitespace=True,header=None)
test_df = pd.read_csv('testNN.txt',delim_whitespace=True,header=None)
train_x = train_df.iloc[:,:-1].values
train_y = train_df.iloc[:,-1].values
test_x = test_df.iloc[:,:-1].values
test_y = test_df.iloc[:,-1].values
train_x = (train_x - train_x.mean(axis=0) ) / train_x.std(axis=0)
test_x = ( test_x - test_x.mean(axis=0)) / test_x.std(axis=0)

nFeatures = train_x.shape[1]
nClasses = len(np.unique(train_y))

# %%
def test_module(nNetwork,test_x,test_y):
    df2 = pd.DataFrame(columns=['no. of layer','no. of nodes/layer','accuracy'])
    
    with open('1605104_2.txt','w') as f:
        pass

    for i in range(nNetwork):
        nn = None
        with open(f'nn_{i+1}.parameters','rb') as f:
            nn = pickle.load(f)

        y_pred = nn.predict(test_x)
        accuracy = nn.accuracy(y_pred,test_y)

        df = pd.DataFrame()
        df['Sample No'] = (y_pred != test_y).nonzero()[0] + 1
        df['Feature'] = test_x[df['Sample No']-1,:].tolist()
        df['Predicted Class'] = y_pred[(y_pred != test_y).nonzero()]
        df['Actual Class'] = test_y[(y_pred != test_y).nonzero()]
        # print(df.to_string(index=False))
        with open(f'nn_{i+1}.parameters','wb') as f:
            pickle.dump(nn,f)
        df2 = df2.append({'no. of layer':len(networks[i])-1,'no. of nodes/layer':networks[i][1:],'accuracy':accuracy},ignore_index=True)
        
        with open('1605104_2.txt','a') as f:
            f.write(f'Number of layers: {len(nn.network_structure)-1}\n')
            f.write(f'Network Structure: {nn.network_structure}\n')
            f.write(f'Correctly classified samples: {(y_pred == test_y).sum()}\n')
            f.write(f'Missclassified samples: {(y_pred != test_y).sum()}\n')
            f.write(f'Accuracy: {accuracy}\n')
            if not df.empty:
                f.write(df.to_string(index=False))

            f.write('\n')
    
    with open('1605104_2.txt','a') as f:
        f.write(df2.to_string(index=False))

# %%
np.random.seed(2)
networks = [
    [nFeatures,2,2,2,nClasses],
    [nFeatures,3,3,nClasses],
    [nFeatures,4,4,4,nClasses],
]

with open('1605104.txt','w') as f:
    pass

df2 = pd.DataFrame(columns=['no. of layer','no. of nodes/layer','accuracy'])

for i in range(len(networks)):
    nn = NeuralNetwork(networks[i])
    nn.train(train_x,train_y,epocs=100,alpha=.01)
    y_pred = nn.predict(test_x)
    
    accuracy = nn.accuracy(y_pred,test_y)

    df = pd.DataFrame()
    df['Sample No'] = (y_pred != test_y).nonzero()[0] + 1
    df['Feature'] = test_x[df['Sample No']-1,:].tolist()
    df['Predicted Class'] = y_pred[(y_pred != test_y).nonzero()]
    df['Actual Class'] = test_y[(y_pred != test_y).nonzero()]

    df2 = df2.append({'no. of layer':len(networks[i])-1,'no. of nodes/layer':networks[i][1:],'accuracy':accuracy},ignore_index=True)
    
    # print(df.to_string(index=False))
    with open(f'nn_{i+1}.parameters','wb') as f:
        987\pickle.dump(nn,f)
    
    with open('1605104.txt','a') as f:
        f.write(f'Number of layers: {len(networks[i])-1}\n')
        f.write(f'Network Structure: {networks[i]}\n')
        f.write(f'Correctly classified samples: {(y_pred == test_y).sum()}\n')
        f.write(f'Missclassified samples: {(y_pred != test_y).sum()}\n')
        f.write(f'Accuracy: {accuracy}\n')
        if not df.empty:
            f.write(df.to_string(index=False))

        f.write('\n')

with open('1605104.txt','a') as f:
    f.write(df2.to_string(index=False))


test_module(len(networks),test_x,test_y)



