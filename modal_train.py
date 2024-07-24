from modal import Image, Secret, App
import os

def docker_image():
    return (
        Image.debian_slim()
        .pip_install("scikit-learn", "wandb", "pandas", "numpy", "matplotlib", "seaborn")
    )

app = App("wandb-sklearn-training")

@app.function(
    image=docker_image(),
    secrets=[Secret.from_name("my-wandb-secret")]
)
async def train_model():
    import wandb
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.inspection import permutation_importance
    import joblib

    # Explicitly set the W&B API key
    wandb.login(key=os.environ["WANDB_API_KEY"])

    # Initialize W&B
    run = wandb.init(project="sklearn-modal-demo", job_type="train")

    # Generate some dummy data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    feature_names = [f"feature_{i}" for i in range(20)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Log dataset size
    wandb.log({"train_size": len(X_train), "test_size": len(X_test)})

    # Train a simple Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log metrics to W&B
    wandb.log({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    # Log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()

    # Log feature importances
    importances = model.feature_importances_
    feature_imp = pd.DataFrame(sorted(zip(importances, feature_names)), columns=['Value','Feature'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('Feature Importances')
    wandb.log({"feature_importances": wandb.Image(plt)})
    plt.close()

    # Permutation Importance
    perm_importance = permutation_importance(model, X_test, y_test)
    perm_imp = pd.DataFrame(sorted(zip(perm_importance.importances_mean, feature_names)), columns=['Value','Feature'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Value", y="Feature", data=perm_imp.sort_values(by="Value", ascending=False))
    plt.title('Permutation Importances')
    wandb.log({"permutation_importances": wandb.Image(plt)})
    plt.close()
    
    # Log the model as an artifact
    model_artifact = wandb.Artifact('random_forest_model', type='model')
    model_path = 'random_forest_model.pkl'
    joblib.dump(model, model_path)
    model_artifact.add_file(model_path)
    run.log_artifact(model_artifact)

    wandb.finish()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

# This allows the function to be imported by the FastAPI app
if __name__ == "__main__":
    app.serve()
