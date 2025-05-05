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
        sign_change = grad_sign * prev_grad_sign

        increase = sign_change > 0
        decrease = sign_change < 0

        # aggiorna la dimensione dei passi
        prev_steps[key][increase] *= eta_pos
        prev_steps[key][decrease] *= eta_neg
        # (clipping opzionale per evitare passi troppo grandi/piccoli)
        prev_steps[key] = np.clip(prev_steps[key], 1e-6, 50)

        # pesi aggiornati solo dove il segno NON è cambiato
        update_mask = sign_change >= 0
        weights[key][update_mask] -= grad_sign[update_mask] * prev_steps[key][update_mask]

        # dove c'è oscillazione: step ridotto, ma NESSUN aggiornamento del peso
        # inoltre: reset del gradiente dove c'è stato cambiamento di segno
        grads[key][decrease] = 0

        # memorizza i gradienti aggiornati
        prev_grads[key] = grads[key]
    return weights, prev_grads, prev_steps

def train_model_rprop_with_early_stopping(x_train, y_train, x_val, y_val, hidden_dim, epochs, eta_pos, eta_neg, patience=10, verbose=True):
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

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch + 1}/{epochs} - Train Acc: {train_accuracy:.4f}, Loss: {train_loss:.4f} | Val Acc: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = weights.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    return best_weights, history


def grid_search_cross_validation_with_test(x_train, y_train, param_grid, num_folds=10, patience=5, epochs=100):
    best_params = None
    best_score = -np.inf
    results = []
    
    # Per tenere traccia del miglior modello e delle sue performance sul test set
    best_test_accuracy = -np.inf
    best_test_loss = np.inf
    best_test_fold_metrics = None
    
    print("\n" + "="*80)
    print(f"INIZIANDO GRID SEARCH CON SPLIT 8/1/1 E {num_folds}-FOLD CROSS-VALIDATION")
    print("="*80)
    
    # Genera tutte le combinazioni di parametri
    param_combinations = [
        {'hidden_dim': hd, 'eta_pos': ep, 'eta_neg': en}
        for hd in param_grid['hidden_dim']
        for ep in param_grid['eta_pos']
        for en in param_grid['eta_neg']
    ]
    
    print(f"Numero totale di combinazioni di parametri da testare: {len(param_combinations)}")
    
    for param_idx, params in enumerate(param_combinations, 1):
        print("\n" + "-"*80)
        print(f"COMBINAZIONE {param_idx}/{len(param_combinations)}: {params}")
        print("-"*80)
        
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        fold_val_accuracies = []
        fold_test_accuracies = []
        fold_test_losses = []
        fold_details = []
        
        for fold, (train_val_idx, test_idx) in enumerate(kfold.split(x_train), 1):
            print(f"\nFold {fold}/{num_folds}")
            print("-"*40)
            
            # Divisione in train_val e test (90% e 10%)
            x_train_val, y_train_val = x_train[train_val_idx], y_train[train_val_idx]
            x_test_fold, y_test_fold = x_train[test_idx], y_train[test_idx]
            
            # Divisione di train_val in train e validation (8:1 rispetto al dataset originale)
            train_idx, val_idx = train_test_split(
                np.arange(len(x_train_val)), 
                test_size=1/9,  # 1/9 del 90% ≈ 10% del dataset originale
                random_state=42+fold
            )
            
            x_tr, y_tr = x_train_val[train_idx], y_train_val[train_idx]
            x_vl, y_vl = x_train_val[val_idx], y_train_val[val_idx]
            
            # Stampa delle dimensioni degli split
            print(f"Dimensione Training:   {x_tr.shape[0]} esempi ({100*x_tr.shape[0]/len(x_train):.1f}% del dataset)")
            print(f"Dimensione Validation: {x_vl.shape[0]} esempi ({100*x_vl.shape[0]/len(x_train):.1f}% del dataset)")
            print(f"Dimensione Test:       {x_test_fold.shape[0]} esempi ({100*x_test_fold.shape[0]/len(x_train):.1f}% del dataset)")
            
            # Training del modello con early stopping
            model, history = train_model_rprop_with_early_stopping(
                x_tr, y_tr, x_vl, y_vl,
                params['hidden_dim'], epochs, params['eta_pos'], params['eta_neg'], patience,
                verbose=False  # Riduciamo l'output durante il training per maggiore chiarezza
            )
            
            # Valutazione sul validation set
            _, _, _, a2_val = forward_pass(x_vl, model)
            val_accuracy = np.mean(np.argmax(a2_val, axis=1) == np.argmax(y_vl, axis=1))
            val_loss = compute_loss(y_vl, a2_val)
            fold_val_accuracies.append(val_accuracy)
            
            # Valutazione sul test set interno
            _, _, _, a2_test = forward_pass(x_test_fold, model)
            test_accuracy = np.mean(np.argmax(a2_test, axis=1) == np.argmax(y_test_fold, axis=1))
            test_loss = compute_loss(y_test_fold, a2_test)
            fold_test_accuracies.append(test_accuracy)
            fold_test_losses.append(test_loss)
            
            # Dettagli del fold
            fold_detail = {
                'epochs_trained': len(history['accuracy']),
                'final_train_accuracy': history['accuracy'][-1],
                'final_train_loss': history['loss'][-1],
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                'model': model  # Salviamo il modello addestrato
            }
            fold_details.append(fold_detail)
            
            # Verifica se questo è il miglior modello sul test set finora
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_test_loss = test_loss
                best_test_fold_metrics = fold_detail
                best_test_params = params.copy()
                best_test_fold = fold
                
            print(f"Epoche completate: {len(history['accuracy'])}")
            print(f"Training  - Accuracy: {history['accuracy'][-1]:.4f}, Loss: {history['loss'][-1]:.4f}")
            print(f"Validation - Accuracy: {val_accuracy:.4f}, Loss: {val_loss:.4f}")
            print(f"Test      - Accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}")
        
        mean_val_accuracy = np.mean(fold_val_accuracies)
        std_val_accuracy = np.std(fold_val_accuracies)
        mean_test_accuracy = np.mean(fold_test_accuracies)
        std_test_accuracy = np.std(fold_test_accuracies)
        mean_test_loss = np.mean(fold_test_losses)
        
        print("\n" + "-"*60)
        print(f"RISULTATO COMPLESSIVO PER PARAMETRI: {params}")
        print(f"Media Validation Accuracy: {mean_val_accuracy:.4f} ± {std_val_accuracy:.4f}")
        print(f"Media Test Accuracy: {mean_test_accuracy:.4f} ± {std_test_accuracy:.4f}")
        print(f"Media Test Loss: {mean_test_loss:.4f}")
        print("-"*60)
        
        if mean_val_accuracy > best_score:
            best_score = mean_val_accuracy
            best_params = params
            print("► NUOVI MIGLIORI PARAMETRI TROVATI ◄")
        
        results.append({
            'params': params,
            'mean_val_accuracy': mean_val_accuracy,
            'std_val_accuracy': std_val_accuracy,
            'mean_test_accuracy': mean_test_accuracy,
            'std_test_accuracy': std_test_accuracy,
            'mean_test_loss': mean_test_loss,
            'fold_val_accuracies': fold_val_accuracies,
            'fold_test_accuracies': fold_test_accuracies,
            'fold_test_losses': fold_test_losses,
            'fold_details': fold_details
        })
    
    # Riassunto finale dei risultati della grid search
    print("\n" + "="*80)
    print("RISULTATI FINALI DELLA GRID SEARCH")
    print("="*80)
    print(f"Miglior combinazione di parametri (basata sulla media validation): {best_params}")
    print(f"Miglior accuracy di validation: {best_score:.4f}")
    
    # Informazioni sul miglior modello individuale (basato sul test set)
    print("\n" + "="*80)
    print("MIGLIOR MODELLO SINGOLO (SULLA PERFORMANCE DEL TEST SET)")
    print("="*80)
    print(f"Parametri: {best_test_params}")
    print(f"Fold: {best_test_fold}")
    print(f"Test Accuracy: {best_test_accuracy:.4f}")
    print(f"Test Loss: {best_test_loss:.4f}")
    print(f"Epoche di training: {best_test_fold_metrics['epochs_trained']}")
    
    # Creiamo una tabella dei risultati ordinata per performance
    print("\nRiepilogo di tutte le combinazioni testate (ordinate per performance):")
    print("-"*100)
    print(f"{'#':<3} {'Hidden':<8} {'Eta+':<6} {'Eta-':<6} {'Val Accuracy':<20} {'Test Accuracy':<20} {'Test Loss':<10}")
    print("-"*100)
    
    sorted_results = sorted(results, key=lambda x: x['mean_val_accuracy'], reverse=True)
    for i, res in enumerate(sorted_results, 1):
        p = res['params']
        print(f"{i:<3} {p['hidden_dim']:<8} {p['eta_pos']:<6} {p['eta_neg']:<6} "
              f"{res['mean_val_accuracy']:.4f} ± {res['std_val_accuracy']:.4f}    "
              f"{res['mean_test_accuracy']:.4f} ± {res['std_test_accuracy']:.4f}    "
              f"{res['mean_test_loss']:.4f}")
    
    # Restituiamo anche le metriche del miglior modello singolo
    return best_params, results, {
        'best_test_params': best_test_params,
        'best_test_accuracy': best_test_accuracy,
        'best_test_loss': best_test_loss,
        'best_test_fold': best_test_fold,
        'best_test_metrics': best_test_fold_metrics
    }


def plot_metrics(history, title="Metriche di Training"):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoca')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def run_experiment():
    # Parametri per la grid search
    param_grid = {
        'hidden_dim': [64, 128],
        'eta_pos': [1.1, 1.2],
        'eta_neg': [0.4, 0.5]
    }
    
    # Esecuzione della grid search con split 8/1/1
    best_params, grid_results, best_model_info = grid_search_cross_validation_with_test(
        x_train, y_train,
        param_grid=param_grid,
        num_folds=5,  # Ridotto a 5 per velocizzare, puoi aumentare a 10
        patience=5,
        epochs=100
    )
    
    print("\n" + "="*80)
    print("CONCLUSIONI FINALI")
    print("="*80)
    print(f"I migliori parametri trovati (media validation): {best_params}")
    print(f"La miglior performance singola sul test set:")
    print(f"  - Parametri: {best_model_info['best_test_params']}")
    print(f"  - Accuracy: {best_model_info['best_test_accuracy']:.4f}")
    print(f"  - Loss: {best_model_info['best_test_loss']:.4f}")
    
    # Opzionalmente, mostriamo le metriche del miglior modello
    if best_model_info['best_test_metrics'] is not None:
        history = {
            'accuracy': best_model_info['best_test_metrics'].get('history_accuracy', []),
            'loss': best_model_info['best_test_metrics'].get('history_loss', []),
            'val_accuracy': best_model_info['best_test_metrics'].get('history_val_accuracy', []),
            'val_loss': best_model_info['best_test_metrics'].get('history_val_loss', [])
        }
        if all(len(v) > 0 for v in history.values()):
            plot_metrics(history, title=f"Metriche del Miglior Modello (Fold {best_model_info['best_test_fold']})")
    
    return best_params, grid_results, best_model_info


# Esecuzione dell'esperimento completo
if __name__ == "__main__":
    best_params, grid_results, best_model_info = run_experiment()