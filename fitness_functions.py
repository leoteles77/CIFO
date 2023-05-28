import numpy as np

def brier_score(y_true,y_pred):
    n = len(y_pred)  # Number of instances
    r = 10  # Number of possible classes
    
    bs = 0.0
    for t in range(n):
        for i in range(r):
            f_ti = y_pred[t][i] 
            o_ti = y_true[t][i]  
            bs += (f_ti - o_ti) ** 2
    
    return bs/n

def cross_entropy_loss(y_true, y_pred):
    # Ensure the inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Clip the predicted values to avoid log(0) errors
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    # Compute the cross-entropy loss
    loss = -np.mean(y_true * np.log(y_pred))
    
    return loss