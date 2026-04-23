#!/bin/bash

# Nom de l'environnement virtuel
ENV_NAME="qiskit_env"

# Créer l'environnement virtuel
python3 -m venv $ENV_NAME

# Activer l'environnement virtuel
source $ENV_NAME/bin/activate

# Mettre à jour pip
pip install --upgrade pip

# Installer JupyterLab et Qiskit
pip install jupyterlab qiskit

# Installer les bibliothèques supplémentaires
pip install qiskit-ibm-runtime sympy matplotlib pylatexenc qiskit-aer 

# --- Configurer IPython pour activer %matplotlib inline par défaut ---
# Créer le profil par défaut si nécessaire
ipython profile create

# Répertoire startup
STARTUP_DIR=$(ipython locate profile default)/startup
mkdir -p "$STARTUP_DIR"

# --- Lancer JupyterLab automatiquement ---
echo "Lancement de JupyterLab"
jupyter lab 

