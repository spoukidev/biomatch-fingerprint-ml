from flask import Flask, render_template, jsonify, request
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Pour éviter les erreurs d'interface graphique
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import time
import io
import base64

app = Flask(__name__)

LIMITS = {
    'n_samples': (100, 10000),
    'n_classes': (2, 20),
    'noise_level': (1, 3),
    'rf_trees': (10, 1000),
}
ALLOWED_SVM_KERNELS = {'linear', 'poly', 'rbf', 'sigmoid'}


def parse_bounded_int(data, key, default):
    value = data.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f'{key} must be an integer')
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{key} must be an integer') from None
    low, high = LIMITS[key]
    if not low <= value <= high:
        raise ValueError(f'{key} must be between {low} and {high}')
    return value


def load_fingerprint_data(n_samples=1000, n_classes=5, noise_level=1):
    # noise_level: 1 (Faible), 2 (Moyen), 3 (Élevé)
    n_features = 50
    n_redundant = int(10 * noise_level)
    n_informative = n_features - n_redundant
    
    # Sécurité pour éviter les erreurs de génération
    if n_classes > 2 ** n_informative:
        n_classes = min(n_classes, 2 ** n_informative)
        
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=42
    )
    return X, y

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/run-simulation', methods=['POST'])
def run_simulation():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object'}), 400

    try:
        n_samples = parse_bounded_int(data, 'n_samples', 1000)
        n_classes = parse_bounded_int(data, 'n_classes', 5)
        noise_level = parse_bounded_int(data, 'noise_level', 1)
        rf_trees = parse_bounded_int(data, 'rf_trees', 100)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    svm_kernel = data.get('svm_kernel', 'rbf')
    if not isinstance(svm_kernel, str) or svm_kernel not in ALLOWED_SVM_KERNELS:
        return jsonify({'error': 'svm_kernel must be one of: linear, poly, rbf, sigmoid'}), 400
    
    # 1. Chargement des données
    X, y = load_fingerprint_data(n_samples=n_samples, n_classes=n_classes, noise_level=noise_level)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Modèles
    models = {
        "SVM": SVC(kernel=svm_kernel, C=1.0, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=rf_trees, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    results = {}
    
    for key, model in models.items():
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        # Average 'weighted' calcule la métrique pour chaque label, 
        # puis trouve leur moyenne pondérée par le support (nombre d'instances vraies pour chaque label).
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        # Matrice de confusion en Base64
        cm = confusion_matrix(y_test, y_pred)
        # Style sombre, appliqué localement : plt.style.use() modifiait l'état
        # global après la création de la figure, donc la première matrice
        # (SVM) de la première requête n'était pas stylée.
        with plt.style.context('dark_background'):
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_title('Matrice de Confusion', color='white')
            ax.set_xlabel('Prédictions', color='white')
            ax.set_ylabel('Réelles', color='white')
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', transparent=True)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        results[key] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1": round(f1 * 100, 2),
            "time": round(train_time, 4),
            "cm_image": img_base64
        }
        
    return jsonify(results)

if __name__ == '__main__':
    # Keep the debugger off unless it is explicitly requested. The Flask
    # debugger is unsafe to expose on a reachable interface.
    debug = os.environ.get('FLASK_DEBUG', '').lower() in {'1', 'true', 'yes'}
    app.run(debug=debug, port=5000)
