import numpy as np
import pickle
def predictions(x,theta):
    z = x.dot(theta)
    predic = 1 / (1 + np.exp(-z))
    return predic

# if y = 1 => -log(h)
# if y = 0 => -log(1-h)
# (y * -log(h)) + (1 - y) * (-log(1-h))

def compute_cost(x,y,theta):
    m = len(y)
    prediction = predictions(x,theta)
    epsilon = 1e-15
    prediction = np.clip(prediction,epsilon,1-epsilon)
    cost = (-1 / m) * np.sum((y * np.log(prediction)) + ((1 - y) * np.log(1 - prediction)))
    return cost


def gradient_descent(x,y,alpha=0.1,iters=5000):
    m = len(y)
    epsilon = 1e-15
    theta = np.zeros((x.shape[1],1))
    for i in range(iters):       
        prediction = predictions(x,theta)
        prediction = np.clip(prediction,epsilon,1-epsilon)
        gradient = (1 / m) * x.T.dot(prediction - y) 
        theta -= alpha * gradient
        cost = compute_cost(x,y,theta)
        print(f"Cost at iteration {i} = {cost}")
    np.save("src\\data\\opt_theta.npy",theta)
    with open("src\\data\\linear_model.pkl", "wb") as f:
        pickle.dump(theta, f)   
    return theta


