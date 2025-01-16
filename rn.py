import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Caricamento e preparazione del dataset MNIST
(x_train, y_train), (_, _) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)

# Funzione per creare il modello
def create_model(num_hidden_units):
    model = Sequential([
        Dense(num_hidden_units, input_shape=(28 * 28,), activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

# Funzione per l'aggiornamento RProp
def rprop_update(weights, grads, prev_grads, prev_step, eta_pos=1.2, eta_neg=0.5, delta_max=50.0, delta_min=1e-6):
    sign_change = np.sign(grads) != np.sign(prev_grads)
    prev_step[sign_change] *= eta_neg
    prev_step[~sign_change] *= eta_pos
    prev_step = np.clip(prev_step, delta_min, delta_max)
    weights -= prev_step * np.sign(grads)
    return weights, grads, prev_step

# Funzione di allenamento
def train_model_rprop(x_train, y_train, num_hidden_units, epochs, eta_pos, eta_neg):
    model = create_model(num_hidden_units)
    weights, biases = model.layers[0].get_weights()

    prev_grads_w = np.zeros_like(weights)
    prev_grads_b = np.zeros_like(biases)
    prev_step_w = np.ones_like(weights) * 0.1
    prev_step_b = np.ones_like(biases) * 0.1

    history = {'accuracy': [], 'loss': []}

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = model(x_train, training=True)
            loss = tf.keras.losses.categorical_crossentropy(y_train, y_pred)

        grads = tape.gradient(loss, model.trainable_variables)
        grads_w, grads_b = grads[0].numpy(), grads[1].numpy()

        weights, prev_grads_w, prev_step_w = rprop_update(weights, grads_w, prev_grads_w, prev_step_w, eta_pos, eta_neg)
        biases, prev_grads_b, prev_step_b = rprop_update(biases, grads_b, prev_grads_b, prev_step_b, eta_pos, eta_neg)

        model.layers[0].set_weights([weights, biases])

        train_loss = np.mean(loss)
        train_accuracy = model.evaluate(x_train, y_train, verbose=0)[1]

        history['loss'].append(train_loss)
        history['accuracy'].append(train_accuracy)

        print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {train_accuracy:.4f}, Loss: {train_loss:.4f}")

    return model, history

# Funzione di cross-validation
def cross_validation(x_train, y_train, num_folds, num_hidden_units, eta_pos, eta_neg):
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_results = []
    histories = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train), 1):
        print(f"\nFold {fold}/{num_folds}")

        # Suddivisione del dataset in train e validation per questo fold
        x_train_fold, y_train_fold = x_train[train_idx], y_train[train_idx]
        x_val_fold, y_val_fold = x_train[val_idx], y_train[val_idx]

        # Addestramento del modello con RProp
        model, history = train_model_rprop(x_train_fold, y_train_fold, num_hidden_units, epochs=50, eta_pos=eta_pos, eta_neg=eta_neg)

        # Valutazione delle performance su train e validation
        train_loss, train_accuracy = model.evaluate(x_train_fold, y_train_fold, verbose=0)
        val_loss, val_accuracy = model.evaluate(x_val_fold, y_val_fold, verbose=0)

        # Salvataggio dei risultati
        fold_results.append({
            "fold": fold,
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss
        })

        histories.append(history)

        # Stampa dei risultati di questo fold
        print(f"Fold {fold}: Train Accuracy = {train_accuracy:.4f}, Train Loss = {train_loss:.4f}")
        print(f"           Validation Accuracy = {val_accuracy:.4f}, Validation Loss = {val_loss:.4f}")

    return fold_results, histories

# Parametri
num_folds = 10
num_hidden_units = 128
eta_pos = 1.1
eta_neg = 0.4

# Cross-validation
fold_results, histories = cross_validation(x_train, y_train, num_folds, num_hidden_units, eta_pos, eta_neg)

# Analisi dei risultati
accuracies = [res["val_accuracy"] for res in fold_results]
losses = [res["val_loss"] for res in fold_results]

# Grafici
fig, ax = plt.subplots(2, 1, figsize=(12, 12))

# Accuratezza per fold
for fold, history in enumerate(histories, 1):
    ax[0].plot(history['accuracy'], label=f'Fold {fold}')
ax[0].set_title("Accuratezza per Fold")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Accuratezza")
ax[0].legend()

# Perdita per fold
for fold, history in enumerate(histories, 1):
    ax[1].plot(history['loss'], label=f'Fold {fold}')
ax[1].set_title("Perdita per Fold")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Perdita")
ax[1].legend()

plt.tight_layout()
plt.show()
