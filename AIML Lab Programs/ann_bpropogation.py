import numpy as np

x = np.array(([2,9], [1,5], [3,6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

x = x/np.amax(x,axis=0)

y = y/100

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivative_sigmoid(x):
    return x*(1-x)

epoch = 7000
lr = 0.2

ip_layer_neurons = 2
hdn_layer_nrns = 3
op_layer_nrns = 1

wh = np.random.uniform(size=(ip_layer_neurons, hdn_layer_nrns))
bh = np.random.uniform(size=(1,hdn_layer_nrns))

wout = np.random.uniform(size=(hdn_layer_nrns, op_layer_nrns))
bout = np.random.uniform(size=(1, op_layer_nrns))

for i in range(epoch):
    hinp1= np.dot(x,wh)
        #first hidden layer input is cal by a dot product
    hinp = hinp1 + bh 
        #hidden layer input

    hlayer_act = sigmoid(hinp)
        #result of activation from hidden will be input to output layer

    outinp1 = np.dot(hlayer_act, wout)
    outinp = outinp1 + bout
    output = sigmoid(outinp)

    #Back Propagation
    EO = y-output
        #Error output layer
    outgrad = derivative_sigmoid(output)
        #output gradient

    d_output = EO*outgrad

    EH = d_output.dot(wout.T)
        #error hidden layer
        # .Target

    hiddengrad = derivative_sigmoid(hlayer_act)

    d_hiddenlayer = EH*hiddengrad

    wout += hlayer_act.T.dot(d_output)*lr
    #bout += np.sum(d_output, axis=0, keepdims=True)*lr

    wh+= x.T.dot(d_hiddenlayer)*lr
    #bh += np.sum(d_hiddenlayer, axis=0, keepdims=True)*lr

print("Input: ", x)
print("Actual Output: ",y)
print("Predicted Output: ", output)
