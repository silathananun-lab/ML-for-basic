from data_loader import load_face_data
from split_data import split_dataset
from preprocessing import standardize_features
from vgg_model import build_dcnn
from evaluate import evaluate_model

def main():
    # 1. Load dataset
    print("Loading data...")
    X, y = load_face_data('age_gender.csv')
    
    # 2. Split data
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # 3. Standardize features
    X_train, X_test = standardize_features(X_train, X_test)
    
    # 4. Compare configurations and epochs
    configurations = [
        {"name": "Config_1_Blocks_10Epochs", "blocks": 2, "neurons": 64, "epochs": 10},
        {"name": "Config_2_Blocks_20Epochs", "blocks": 2, "neurons": 128, "epochs": 20},
        {"name": "Config_3_Blocks_10Epochs", "blocks": 3, "neurons": 128, "epochs": 10},
    ]
    
    results = {}
    for cfg in configurations:
        print(f"\nTraining {cfg['name']}...")
        model = build_dcnn(num_conv_blocks=cfg['blocks'], dense_neurons=cfg['neurons'])
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=cfg['epochs'],
            batch_size=64,
            verbose=1
        )
        
        acc = evaluate_model(model, X_test, y_test, history, cfg['name'])
        results[cfg['name']] = acc
        
    print("\n--- Final Comparison Summary ---")
    for name, acc in results.items():
        print(f"{name}: Accuracy = {acc:.4f}")

if __name__ == '__main__':
    main()