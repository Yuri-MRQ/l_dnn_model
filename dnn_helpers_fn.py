#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    
    A = (1/(1+np.exp(-z)))
    cache = z
    
    return A, cache


# In[3]:


def relu(z):
    """
    Compute the relu of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    
    A = np.maximum(0,z)
    assert(A.shape == z.shape)
    cache = z
    
    return A, cache


# In[4]:


def tanh(z):
    """
    Compute the tanh of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    
    A = (2/(1+ np.exp(-2*z)))-1
    cache = z
    
    return A, cache


# In[5]:


def softmax(z):
    
    """Compute the softmax
    
    Arguments:
    z- A scalar or numpy arry of any size.
    
    Return:
    sm -- softmax(z)
    """
    z = z - np.max(z)
    A = np.divide(np.exp(z), np.sum(np.exp(z), axis=0, keepdims=True))
    cache = z
    
    return A, cache


# In[6]:


def back_sigmoid(dA, Z):
    """
    Compute the derivative of sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.
    dA --

    Return:
    s -- sigmoid(z)
    """
    
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    assert (dZ.shape == Z.shape)
    
    
    return dZ


# In[7]:


def back_relu(dA, Z):
    """
    Compute the derivative of relu of z

    Arguments:
    z -- A scalar or numpy array of any size.
    dA --

    Return:
    s -- sigmoid(z)
    """
    
    dZ = np.array(dA, copy=True)  
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    
    return dZ


# In[8]:


def back_tanh(dA, Z):
    """
    Compute the derivative of tanh of z

    Arguments:
    z -- A scalar or numpy array of any size.
    dA -- 

    Return:
    s -- sigmoid(z)
    """
    dZ = dAL*(1 - np.power(np.tanh(Z), 2))
    
    assert (dZ.shape == Z.shape)
    
    return dZ


# In[9]:


def back_softmax(A, Y, Z):
    
    """Compute the derivative of softmax
    
    Arguments:
    z- A scalar or numpy arry of any size.
    A -- 
    Y -- 
    Z -- 
    
    Return:
    sm -- softmax(z)
    """
    dZ = A - Y
    
    assert (dZ.shape == Z.shape)
    
    return dZ


# In[10]:


def init_parameters(model_shape, activation):
    """
    Inizialize the model paramaters w, b
    
    Arguments:
    model_shape -- A list with all the layers shapes. Ex.: (10, 5, 4, 1), in this example we have a 3-layer model with 
    2 hidden layer (5 units and 4 units) and 1 output layer (binarie)
    
    Return:
    parameters -- return a dictionare of paramaters, the key is the paramater + l (layer number)
    """
    np.random.seed(3)
    parameters = {}
    for l in range(1, len(model_shape)):
        
        if activation[-1] == 'sigmoid':
            parameters['W'+str(l)] = np.random.randn(model_shape[l], model_shape[l-1])/ np.sqrt(model_shape[l-1])
        else:
            parameters['W'+str(l)] = np.random.randn(model_shape[l], model_shape[l-1]) * 0.01
        
        parameters['b'+str(l)] = np.zeros((model_shape[l], 1))
        
        assert(parameters['W'+str(l)].shape == (model_shape[l], model_shape[l-1]))
        assert(parameters['b'+str(l)].shape == (model_shape[l], 1))
        
    return parameters


# In[11]:


def linear_foward(A, W, b):
    Z = np.dot(W, A)+b
    cache = (A, W, b)
    assert(Z.shape == (W.shape[0], A.shape[1]))
    
    return Z, cache


# In[12]:


def activation_foward(A_prev, W, b, activation):
    
    if activation == 'relu':
        Z, linear_cache = linear_foward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == 'sigmoid':
        Z, linear_cache = linear_foward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == 'tanh':
        Z, linear_cache = linear_foward(A_prev, W, b)
        A, activation_cache = tanh(Z)
    else:
        Z, linear_cache = linear_foward(A_prev, W, b)
        A, activation_cache = softmax(Z)
        
    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    
    return A, cache


# In[13]:


def foward_prop(X, parameters, activation):
    """
    Forward propagation
    
    Arguments:
    
    X -- array of inputs
    parameters -- dict of parameters (W1, W2 ... WL), (b1, b2 ... bL)
    activation -- list of layers activations
    
    Return:
    
    AL -- last A from the output
    cache -- list with all caches
    """
    caches = []
    L = len(parameters)//2

    A = X
    
    for l in range(1, L):
        A_prev = A
        
        #linear foward propagation
        A, cache = activation_foward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activation[l-1])
        caches.append(cache)
        
    #last layer
    AL, cache = activation_foward(A,  parameters['W'+str(L)], parameters['b'+str(L)], activation[L-1])
    caches.append(cache)
    
    assert(AL.shape == (parameters['W'+str(L)].shape[0],X.shape[1]))
    
    return AL, caches
        


# In[14]:


def cost_comput(AL, Y, activation):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    
    """
    #if last layer == softmax
    L = len(activation)
    m = Y.shape[1]
    
    if activation[L-1] == 'softmax':
        cost = -1/m * np.sum(np.multiply(Y, np.log(AL)))    
        
    else:

        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        
    
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost


# In[15]:


def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)

    assert(dW.shape == W.shape)
    assert(db.shape == (dZ.shape[0], 1))
    assert(dA_prev.shape == A_prev.shape)
    
    return dW, db, dA_prev


# In[16]:


def activation_backward(dA, Y, A, cache, activation):
    linear_cache, activation_cache = cache
    Z = activation_cache
    if activation == 'relu':
        dZ = back_relu(dA, Z)
        dW, db, dA_prev = linear_backward(dZ,linear_cache)

    elif activation == 'sigmoid':
        dZ = back_sigmoid(dA, Z)
        dW, db, dA_prev = linear_backward(dZ,linear_cache)

    elif activation == 'tanh':
        dZ = back_tanh(dA, Z)
        dW, db, dA_prev = linear_backward(dZ,linear_cache)
        
    else:
        dZ = back_softmax(A, Y, Z)
        dW, db, dA_prev = linear_backward(dZ,linear_cache)
        
    assert(dZ.shape == Z.shape)
        
    return dW, db, dA_prev


# In[17]:


def back_prop(Y, AL, caches, activation):
    """
    Calculate backward propagation
    
    Arguments:
    
    Y -- True labels
    AL -- Model lavel predictions
    caches -- cache contains z, w, b, 0 index
    activation -- list of layers activations
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    if activation[L-1] == 'sigmoid':
        #derivative of cost with respect to AL
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    else:
        dAL = 0
    #We have to start from the last layer, L in this case, going down to the first layer 0. 
    #For this we have to initiate with the last layer cache
    current_cache = caches[L-1]
    #unpacking cache

    grads['dW'+str(L)], grads['db'+str(L)], grads['dA'+str(L-1)] =  activation_backward(dAL, Y, AL, current_cache, activation[L-1])
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        #unpacking cache
        
        linear_cache, activation_cache = current_cache
        A_prev, W, b= linear_cache
        Z = activation_cache
        
        grads['dW'+str(l+1)], grads['db'+str(l+1)], grads['dA'+str(l)]  = activation_backward(grads['dA'+str(l+1)], Y, AL, current_cache, activation[l])    
        
        
    
    return grads


# In[18]:


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    
    L = len(parameters)//2
    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate*grads['db'+str(l+1)]
        
    return parameters
            


# In[19]:


#from Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow, Aurélien Géron
#and some minors modifications
def print_status_bar(iteration, print_time, total, loss, acc, metrics=None):
    metrics_loss = " - ".join(["{}: {:.4f}".format(m, np.squeeze(loss[m][int(iteration/print_time)])) for m in loss ])
    metric_acc = " - ".join(["{}: {:.4f}".format(m, np.squeeze(acc[m][int(iteration/print_time)])) for m in acc ])
    end = "" if iteration < total else "\n"
    print("\r{}/{}  - ".format(iteration, total) + metrics_loss + ' / ' + metric_acc, end=end)


# In[20]:


def l_dnn(X_train, Y_train, X_dev, Y_dev, learning_rate, num_iterations, model_shape, activation, print_cost=True, verbose=True):
    """
    Model training
    
    Arguments:
    X -- np.array with the inputs
    Y -- true labels
    learning_rate -- for update the parameters - float
    num_iterations -- number of times the paramater going to be update - int
    model_shape -- python list, the len(model_shape) is the number of layer and the model_shape[i] is
    the number of units
    activation -- python list with the activation function for the layers
    
    Return:
    
    parameters -- model parameters
    """
    np.random.seed(1)
    costs = {"train_cost" : [], "dev_cost" : []}  
    acc = {"train_acc" : [], "dev_acc" : []}
    #inizialize paramater
    parameters = init_parameters(model_shape, activation)
    
    for i in range(num_iterations):
        #foward prop
        AL, caches = foward_prop(X_train, parameters, activation)
        AL_dev, caches_dev = foward_prop(X_dev, parameters, activation)
        #compute cost
        cost_train = cost_comput(AL, Y_train, activation)
        cost_dev = cost_comput(AL_dev, Y_dev, activation)
        #back prop
        grads = back_prop(Y_train, AL, caches, activation)
        #parameters update
        parameters = update_parameters(parameters, grads, learning_rate)
        #predict onto train and dev dataset
        pred_train, probs_train = predict(X_train, parameters, activation)
        pred_dev, probs_dev = predict(X_dev, parameters, activation)
        train_acc = evaluate_prediction(pred_train, Y_train)
        dev_acc = evaluate_prediction(pred_dev, Y_dev)
        # Print the cost every 100 training example
        if verbose:
            if print_cost and i % 10 == 0:
                costs['train_cost'].append(cost_train)
                costs['dev_cost'].append(cost_dev)
                acc["train_acc"].append(train_acc)
                acc["dev_acc"].append(dev_acc)
                print_status_bar(i, 10, num_iterations, loss=costs, acc=acc, metrics=None)
                
    # plot the cost
    fig,ax = plt.subplots(1)
    l1, = ax.plot(np.squeeze(costs['train_cost']), color='blue')
    l2, = ax.plot(np.squeeze(costs['dev_cost']), color='red')
    l1.set_label('train_cost')
    l2.set_label('dev_cost')
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    ax.legend()
    plt.show()
    
    #plot the acc
    fig,ax = plt.subplots(1)
    l1, = ax.plot(np.squeeze(acc['train_acc']), color='blue')
    l2, = ax.plot(np.squeeze(acc['dev_acc']), color='red')
    l1.set_label('train_acc')
    l2.set_label('dev_acc')
    plt.ylabel('acc')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    ax.legend()
    plt.show()
    
    return parameters, activation
    
    


# In[21]:


#helpd by this great git hub https://charleslow.github.io/softmax_from_scratch/
def predict(X, parameters, activation):
    # Forward propagation
    probabilities, caches = foward_prop(X, parameters, activation)
    
    # Calculate Predictions (the highest probability for a given example is coded as 1, otherwise 0)
    predictions = (probabilities == np.amax(probabilities, axis=0, keepdims=True))
    predictions = predictions.astype(float)

    return predictions, probabilities

def evaluate_prediction(predictions, Y):
    m = Y.shape[1]
    predictions_class = predictions.argmax(axis=0).reshape(1, m)
    Y_class = Y.argmax(axis=0).reshape(1, m)
    
    return np.sum((predictions_class == Y_class) / (m))


# In[ ]:




