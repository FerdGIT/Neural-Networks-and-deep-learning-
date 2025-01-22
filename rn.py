import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28*28) / 255.0  # Normalizzazione e reshaping (28x28 a 784)
x_test = x_test.reshape(-1, 28*28) / 255.0

# One-hot encoding delle etichette
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

# Funzione di forward pass
def forward_pass(X, weights):
    z1 = np.dot(X, weights["W1"]) + weights["b1"]
    a1 = np.tanh(z1)  # Funzione di attivazione tanh
    z2 = np.dot(a1, weights["W2"]) + weights["b2"]
    a2 = softmax(z2)  # Funzione softmax per la classificazione multi-classe
    return z1, a1, z2, a2


def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

# Funzione per calcolare la loss (entropia incrociata)
def compute_loss(y, a2):
    m = y.shape[0]
    log_likelihood = -np.log(a2[range(m), np.argmax(y, axis=1)])
    loss = np.sum(log_likelihood) / m
    return loss

# Funzione di backward pass
def backward_pass(X, y, z1, a1, z2, a2, weights):
    m = X.shape[0]
    
    delta2 = a2 - y  # Derivata della loss rispetto all'output
    dW2 = np.dot(a1.T, delta2) / m
    db2 = np.sum(delta2, axis=0) / m

    delta1 = np.dot(delta2, weights["W2"].T) * (1 - np.tanh(z1) ** 2)  # Derivata della funzione tanh
    dW1 = np.dot(X.T, delta1) / m
    db1 = np.sum(delta1, axis=0) / m

    grads = {
        "W1": dW1, "b1": db1,
        "W2": dW2, "b2": db2
    }
    return grads

# Funzione RProp per l'aggiornamento dei pesi
def rprop_update(weights, grads, prev_grads, prev_steps, eta_pos, eta_neg):
    for key in weights:
        grad_sign = np.sign(grads[key])
        prev_grad_sign = np.sign(prev_grads[key])

        # Aggiornamento passo
        update_step = prev_steps[key] * np.where(grad_sign == prev_grad_sign, eta_pos, eta_neg)
        prev_steps[key] = update_step
        weights[key] -= np.sign(grads[key]) * update_step  # Applicazione del gradiente aggiornato

        prev_grads[key] = grads[key]
    
    return weights, prev_grads, prev_steps

# Funzione per addestrare il modello con Early Stopping
def train_model_rprop_with_early_stopping(x_train, y_train, x_val, y_val, hidden_dim, epochs, eta_pos, eta_neg, patience=10):
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]

    weights = initialize_weights(input_dim, hidden_dim, output_dim)

    prev_grads = {key: np.zeros_like(val) for key, val in weights.items()}
    prev_steps = {key: np.ones_like(val) * 0.1 for key, val in weights.items()}

    history = {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []}

    # Variabili per l'Early Stopping
    best_loss = np.inf
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        # Forward pass
        z1, a1, z2, a2 = forward_pass(x_train, weights)

        # Loss
        train_loss = compute_loss(y_train, a2)

        # Backward pass
        grads = backward_pass(x_train, y_train, z1, a1, z2, a2, weights)

        # RProp Update
        weights, prev_grads, prev_steps = rprop_update(weights, grads, prev_grads, prev_steps, eta_pos, eta_neg)

        # Accuratezza sul training set
        predictions_train = np.argmax(a2, axis=1)
        targets_train = np.argmax(y_train, axis=1)
        train_accuracy = np.mean(predictions_train == targets_train)

        # Validazione
        z1_val, a1_val, z2_val, a2_val = forward_pass(x_val, weights)
        val_loss = compute_loss(y_val, a2_val)
        predictions_val = np.argmax(a2_val, axis=1)
        targets_val = np.argmax(y_val, axis=1)
        val_accuracy = np.mean(predictions_val == targets_val)

        # Memorizza le metriche
        history["accuracy"].append(train_accuracy)
        history["loss"].append(train_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} - Train Acc: {train_accuracy:.4f}, Train Loss: {train_loss:.4f} - Val Acc: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}")

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = weights.copy()
            patience_counter = 0  # Reset del contatore
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Restituisce i pesi migliori
    return best_weights, history

# Funzione di cross-validation con Early Stopping
def cross_validation_with_early_stopping(x_train, y_train, num_folds, hidden_dim, eta_pos, eta_neg, patience):
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_results = []
    histories = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train), 1):
        print(f"\nFold {fold}/{num_folds}")

        # Suddivisione del dataset
        x_train_fold, y_train_fold = x_train[train_idx], y_train[train_idx]
        x_val_fold, y_val_fold = x_train[val_idx], y_train[val_idx]

        # Addestramento del modello con Early Stopping
        weights, history = train_model_rprop_with_early_stopping(
            x_train_fold, y_train_fold, x_val_fold, y_val_fold,
            hidden_dim=hidden_dim, epochs=200, eta_pos=eta_pos, eta_neg=eta_neg, patience=patience
        )

        # Validazione
        z1, a1, z2, a2 = forward_pass(x_val_fold, weights)
        val_loss = compute_loss(y_val_fold, a2)
        predictions = np.argmax(a2, axis=1)
        targets = np.argmax(y_val_fold, axis=1)
        val_accuracy = np.mean(predictions == targets)

        fold_results.append({
            "fold": fold,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss
        })
        histories.append(history)

        print(f"Fold {fold}: Validation Accuracy = {val_accuracy:.4f}, Validation Loss = {val_loss:.4f}")

    return fold_results, histories

# Funzione per tracciare i grafici di accuratezza e perdita
def plot_metrics(histories):
    # Media delle metriche su tutte le fold
    num_epochs = min(len(history['accuracy']) for history in histories)  # Minimo numero di epoche tra le fold
    avg_accuracy = np.mean([history['accuracy'][:num_epochs] for history in histories], axis=0)
    avg_loss = np.mean([history['loss'][:num_epochs] for history in histories], axis=0)
    avg_val_accuracy = np.mean([history['val_accuracy'][:num_epochs] for history in histories], axis=0)
    avg_val_loss = np.mean([history['val_loss'][:num_epochs] for history in histories], axis=0)

    # Plot della Loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(avg_loss, label='Train Loss')
    plt.plot(avg_val_loss, label='Validation Loss')
    plt.title("Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot dell'Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(avg_accuracy, label='Train Accuracy')
    plt.plot(avg_val_accuracy, label='Validation Accuracy')
    plt.title("Accuracy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Parametri di addestramento
num_folds = 10
hidden_dim = 128
eta_pos = 1.1
eta_neg = 0.4
patience = 5
 
# Esegui la cross-validation con Early Stopping
fold_results, histories = cross_validation_with_early_stopping(x_train, y_train, num_folds, hidden_dim, eta_pos, eta_neg, patience)

# Traccia i grafici delle metriche
plot_metrics(histories)
