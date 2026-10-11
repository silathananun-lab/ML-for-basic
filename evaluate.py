import matplotlib.pyplot as plt
import os

def evaluate_model(model, X_test, y_test, history, config_name):
    """Evaluates model performance and saves training curves."""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"[{config_name}] Test Accuracy: {accuracy:.4f} | Test Loss: {loss:.4f}")
    
    # Save training curves
    os.makedirs('outputs', exist_ok=True)
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{config_name} - Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{config_name} - Loss')
    plt.legend()
    
    plt.savefig(f'outputs/{config_name}_performance.png')
    plt.close()
    
    return accuracy