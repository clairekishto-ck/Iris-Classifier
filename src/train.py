

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib

def main():
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nModel Evaluation Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Save the confusion matrix as a PNG image 
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Iris Classification Confusion Matrix')
    plt.tight_layout()
    
    matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(matrix_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to: {matrix_path}")
    
    # Save the trained model using joblib 
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved trained model artifact to: {model_path}")

if __name__ == "__main__":
    main()
