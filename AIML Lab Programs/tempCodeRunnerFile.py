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
