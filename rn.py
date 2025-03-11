import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Caricamento del dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocessing
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def initialize_weights(input_dim, hidden_dim, output_dim):
    weights = {
        "W1": np.random.randn(input_dim, hidden_dim) * 0.1,
        "b1": np.zeros(hidden_dim),
        "W2": np.random.randn(hidden_dim, output_dim) * 0.1,
        "b2": np.zeros(output_dim)
    }
    return weights

def forward_pass(X, weights):
    z1 = np.dot(X, weights["W1"]) + weights["b1"]
    a1 = np.tanh(z1)
    z2 = np.dot(a1, weights["W2"]) + weights["b2"]
    a2 = softmax(z2)
    return z1, a1, z2, a2

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def compute_loss(y, a2):
    m = y.shape[0]
    log_likelihood = -np.log(a2[range(m), np.argmax(y, axis=1)])
    loss = np.sum(log_likelihood) / m
    return loss

def backward_pass(X, y, z1, a1, z2, a2, weights):
    m = X.shape[0]
    delta2 = a2 - y
    dW2 = np.dot(a1.T, delta2) / m
    db2 = np.sum(delta2, axis=0) / m
    delta1 = np.dot(delta2, weights["W2"].T) * (1 - np.tanh(z1) ** 2)
    dW1 = np.dot(X.T, delta1) / m
    db1 = np.sum(delta1, axis=0) / m
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

def rprop_update(weights, grads, prev_grads, prev_steps, eta_pos, eta_neg):
    for key in weights:
        grad_sign = np.sign(grads[key])
        prev_grad_sign = np.sign(prev_grads[key])
        update_step = prev_steps[key] * np.where(grad_sign == prev_grad_sign, eta_pos, eta_neg)
        prev_steps[key] = update_step
        weights[key] -= np.sign(grads[key]) * update_step
        prev_grads[key] = grads[key]
    return weights, prev_grads, prev_steps

def train_model_rprop_with_early_stopping(x_train, y_train, x_val, y_val, hidden_dim, epochs, eta_pos, eta_neg, patience=10):
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    weights = initialize_weights(input_dim, hidden_dim, output_dim)
    prev_grads = {key: np.zeros_like(val) for key, val in weights.items()}
    prev_steps = {key: np.ones_like(val) * 0.1 for key, val in weights.items()}
    history = {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []}
    best_loss = np.inf
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        z1, a1, z2, a2 = forward_pass(x_train, weights)
        train_loss = compute_loss(y_train, a2)
        grads = backward_pass(x_train, y_train, z1, a1, z2, a2, weights)
        weights, prev_grads, prev_steps = rprop_update(weights, grads, prev_grads, prev_steps, eta_pos, eta_neg)
        
        predictions_train = np.argmax(a2, axis=1)
        targets_train = np.argmax(y_train, axis=1)
        train_accuracy = np.mean(predictions_train == targets_train)
        
        z1_val, a1_val, z2_val, a2_val = forward_pass(x_val, weights)
        val_loss = compute_loss(y_val, a2_val)
        predictions_val = np.argmax(a2_val, axis=1)
        targets_val = np.argmax(y_val, axis=1)
        val_accuracy = np.mean(predictions_val == targets_val)
        
        history["accuracy"].append(train_accuracy)
        history["loss"].append(train_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} - Train Acc: {train_accuracy:.4f}, Loss: {train_loss:.4f} | Val Acc: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = weights.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return best_weights, history


def grid_search_cross_validation(x_train, y_train, param_grid, num_folds=5, patience=5, epochs=100):
    best_params = None
    best_score = -np.inf
    results = []
    
    # Genera tutte le combinazioni di parametri
    param_combinations = [
        {'hidden_dim': hd, 'eta_pos': ep, 'eta_neg': en}
        for hd in param_grid['hidden_dim']
        for ep in param_grid['eta_pos']
        for en in param_grid['eta_neg']
    ]
    
    for params in param_combinations:
        print(f"\nTesting parameters: {params}")
        kfold = KFold(n_splits=num_folds, shuffle=True)
        fold_accuracies = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train), 1):
            x_tr, y_tr = x_train[train_idx], y_train[train_idx]
            x_vl, y_vl = x_train[val_idx], y_train[val_idx]
            
            model, history = train_model_rprop_with_early_stopping(
                x_tr, y_tr, x_vl, y_vl,
                params['hidden_dim'], epochs, params['eta_pos'], params['eta_neg'], patience
            )
            
            _, _, _, a2_val = forward_pass(x_vl, model)
            val_accuracy = np.mean(np.argmax(a2_val, axis=1) == np.argmax(y_vl, axis=1))
            fold_accuracies.append(val_accuracy)
            print(f"Fold {fold} Val Accuracy: {val_accuracy:.4f}")
        
        mean_accuracy = np.mean(fold_accuracies)
        if mean_accuracy > best_score:
            best_score = mean_accuracy
            best_params = params
        
        results.append({
            'params': params,
            'mean_accuracy': mean_accuracy,
            'fold_accuracies': fold_accuracies
        })
        print(f"Mean validation accuracy: {mean_accuracy:.4f}")
    
    return best_params, results

# Parametri per la grid search
param_grid = {
    'hidden_dim': [64, 128],
    'eta_pos': [1.1, 1.2],
    'eta_neg': [0.4, 0.5]
}

#grid search
best_params, grid_results = grid_search_cross_validation(
    x_train, y_train,
    param_grid=param_grid,
    num_folds=5,
    patience=5,
    epochs=100
)

print("\n" + "="*50)
print(f"Migliori parametri trovati: {best_params}")
print("="*50)

# Addestramento finale con i migliori parametri
x_train_final, x_val_final, y_train_final, y_val_final = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42
)

final_model, final_history = train_model_rprop_with_early_stopping(
    x_train_final, y_train_final,
    x_val_final, y_val_final,
    best_params['hidden_dim'], 200,
    best_params['eta_pos'], best_params['eta_neg'],
    patience=5
)

# Valutazione finale sul test set
_, _, _, a2_test = forward_pass(x_test, final_model)
test_accuracy = np.mean(np.argmax(a2_test, axis=1) == np.argmax(y_test, axis=1))
test_loss = compute_loss(y_test, a2_test)

print("\n" + "="*100)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print("="*100)


def plot_metrics(histories):
    num_epochs = min(len(h['accuracy']) for h in histories)
    avg_loss = np.mean([h['loss'][:num_epochs] for h in histories], axis=0)
    avg_val_loss = np.mean([h['val_loss'][:num_epochs] for h in histories], axis=0)
    avg_acc = np.mean([h['accuracy'][:num_epochs] for h in histories], axis=0)
    avg_val_acc = np.mean([h['val_accuracy'][:num_epochs] for h in histories], axis=0)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(avg_loss, label='Train')
    plt.plot(avg_val_loss, label='Validation')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(avg_acc, label='Train')
    plt.plot(avg_val_acc, label='Validation')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

# Plot delle metriche finali
plot_metrics([final_history])