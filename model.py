"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """
    # TODO: Return (X, y) under one shared seeded row permutation
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    X_shuffled, y_shuffled = X[idx], y[idx]
    return X_shuffled, y_shuffled

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    # TODO: Slice already-shuffled data into contiguous train/val/test partitions...
    n = len(X)
    n_train, n_val = int(n*train_frac), int(n*val_frac)
    n_test = n - n_train - n_val
    X_tr, y_tr, X_va, y_va, X_te, y_te = X[:n_train], y[:n_train], X[n_train:n_train+n_val], y[n_train:n_train+n_val], X[n_train+n_val:n_train+n_val+n_test], y[n_train+n_val:n_train+n_val+n_test]
    return X_tr,y_tr,X_va,y_va,X_te,y_te

# Step 3 - compute_feature_stats
def compute_feature_stats(X):
    # TODO: Compute per-feature mean and std; replace std of 0 with 1
    feature_mean = np.mean(X,axis=0)
    feature_std = np.std(X,axis=0)
    feature_ones = np.ones_like(feature_std)
    feature_std_fix = np.where(feature_std == 0.0,feature_ones,feature_std)
    return feature_mean,feature_std_fix

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    # TODO: Apply z-score normalization using precomputed training mean and std.
    X_scaled = (X-mean)/std
    return X_scaled

# Step 5 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to feature matrix X
    n,d = X.shape
    bias_col = np.ones((n,1))
    X_biased = np.hstack([bias_col,X])
    return X_biased

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    # TODO: Standardize features then add the bias column to form the design matrix.
    X_scaled = standardize_features(X,mean,std)
    X_biased = add_bias_column(X_scaled)
    return X_biased

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    # TODO: Return the predicted target vector from X and weights
    return X @ weights

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    # TODO: Return the average of squared residuals as a scalar float.
    return np.mean((y_true - y_pred) ** 2)

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    # TODO: Return the analytic MSE gradient w.r.t. weights: (2/n) X^T (y_pred - y_true)
    n = X.shape[0]
    return 2 / n * X.T @ (y_pred - y_true)

# Step 10 - normal_equation
def normal_equation(X, y):
    # TODO: Solve for the closed-form least-squares weights via the normal equation.
    return np.linalg.solve(X.T @ X, X.T @ y)

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    # TODO: Return (n_features,) weights sampled from N(0, 0.01)
    np.random.seed(seed=seed)
    return np.random.normal(0,0.01,n_features)

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    # TODO: return the updated weight vector after one MSE gradient step
    y_pred = predict_linear(X,weights)
    dg = mse_gradient(X,y,y_pred)
    weights -= lr * dg
    return weights

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """
    # TODO: return the pair (train_loss, val_loss) as MSE floats
    y_train_pre, y_val_pre = predict_linear(X_train,weights), predict_linear(X_val,weights)
    train_loss, val_loss = mse_loss(y_train_pre,y_train), mse_loss(y_val_pre,y_val)
    return train_loss, val_loss

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    # TODO: Update best weights and patience counter; signal stop when val loss stalls...
    stop = False
    if val_loss < best_val_loss:
        wait = 0
        best_val_loss, best_weights = val_loss, weights.copy()
    else:
        wait += 1
        if wait >= patience:
            stop = True
    return best_val_loss, wait, best_weights, stop

# Step 15 - init_training_state
def init_training_state(n_features, seed=None):
    # TODO: Build the initial training-state dictionary for the GD epoch loop.
    weights = initialize_weights(n_features,seed)
    best_weights = weights.copy()
    best_val_loss = np.inf
    wait = 0
    train_losses, val_losses = [], []
    stopped = False
    metrics_dict = {"weights":weights,\
                    "best_weights":best_weights,\
                    "best_val_loss":best_val_loss,\
                    "wait":wait,\
                    "train_losses":train_losses,\
                    "val_losses":val_losses,\
                    "stopped":stopped
                    } 
    return metrics_dict

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """Perform one GD step, log losses, and refresh early-stopping on state.

    Args:
        state: Dict with keys weights, best_weights, best_val_loss, wait,
            stopped, train_losses, val_losses.
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        lr: Learning rate (float).
        patience: Early-stopping patience (int).

    Returns:
        Updated state dict.
    """
    # TODO: Take one GD step, log train/val losses, refresh early-stopping fields...
    # GD step
    weights = state["weights"]
    gd_step(X_train,y_train,weights,lr)
    # log train/val losses
    train_loss, val_loss = epoch_train_val_losses(X_train,y_train,X_val,y_val,weights)
    state["train_losses"].append(train_loss)
    state["val_losses"].append(val_loss)
    #  refresh early-stopping fields
    best_val_loss, wait, best_weights, stop = update_early_stop_state(val_loss,state["best_val_loss"],state["wait"],weights,state["best_weights"],patience)
    # update state
    state["best_val_loss"],state["wait"],state["best_weights"],state["stopped"] = best_val_loss, wait, best_weights, stop
    return state

# Step 17 - train_batch_gd
def train_batch_gd(X_train, y_train, X_val, y_val, lr, epochs, patience, seed=None):
    # TODO: Train weights with full-batch GD for up to epochs, with early stopping.
    n_features = X_train.shape[1]
    state = init_training_state(n_features,seed)
    for epoch in range(epochs):
        state = run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience)
        if state["stopped"]:
            break
    return state["best_weights"],state["train_losses"],state["val_losses"]

# Step 18 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 19 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 20 - r_squared (not yet solved)
# TODO: implement

# Step 21 - evaluate_regression (not yet solved)
# TODO: implement

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

