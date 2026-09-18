import os
import glob
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from audio_utils import extract_features

DATASET_PATH = r"C:\Users\vignan\Downloads\vocalization_data\vocalization_data\Chicken_Audio_Dataset"
MODEL_SAVE_PATH = "model.pkl"

def load_data():
    X = []
    y = []
    
    classes = ['Healthy', 'Noise', 'Unhealthy']
    
    for label_idx, label in enumerate(classes):
        folder_path = os.path.join(DATASET_PATH, label)
        print(f"Loading files from {folder_path}...")
        
        # Load all wav files in the directory
        for file_path in glob.glob(os.path.join(folder_path, "*.wav")):
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(label)
                
    return np.array(X), np.array(y)

def train_model():
    print("Loading data...")
    X, y = load_data()
    
    if len(X) == 0:
        print("No data loaded. Check the dataset path.")
        return
        
    print(f"Loaded {len(X)} samples.")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    # Initialize Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(rf_model, f)
        
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()