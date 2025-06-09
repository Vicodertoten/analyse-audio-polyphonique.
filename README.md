# Analyse audio polyphonique : une boîte à outils pour la recherche

Ce projet contient le code source du pipeline d'analyse audio automatisée développé dans le cadre du mémoire de Master "Analyse de la mise en forme sonore du podcast ‘Tacapté’" (V. Ryelandt, IHECS, 2025).

L'objectif de ce pipeline n'est pas de remplacer l'analyse qualitative humaine (*close listening*), qui reste au cœur de la méthodologie du mémoire, mais de servir d'outil d'exploration prospective pour :
1.  Évaluer les capacités et les limites des modèles actuels (Whisper, Pyannote, YAMNet) sur un objet sonore complexe.
2.  Documenter une méthode d'analyse polyphonique reproductible, en extrayant et croisant plusieurs couches d'information (parole, musique, bruitages, silences).
3.  Fournir une base de données et de code transparente pour de futures recherches.

## Fonctionnalités principales

Le pipeline (`main.py`) automatise les tâches suivantes :
* **Prétraitement audio :** Conversion automatique des fichiers en format WAV mono 16kHz pour standardiser l'analyse.
* **Extraction de caractéristiques :** Calcul du volume (RMS), du taux de passage par zéro (ZCR) et des MFCCs par seconde, avec un seuil de détection de silence adaptatif.
* **Transcription :** Transcription intégrale de l'audio via le modèle `medium` de Whisper.
* **Diarisation :** Identification des locuteurs avec `pyannote/speaker-diarization-3.1`, avec une distinction entre les segments de parole unique et les segments de parole superposée.
* **Classification sonore :** Identification multi-label d'événements sonores (Musique, Bruitages, etc.) via le modèle YAMNet.
* **Synthèse polyphonique :** Création d'une timeline unifiée qui, pour chaque pas de temps, indique quels locuteurs parlent, quels sons sont présents et si le moment est silencieux.
* **Calcul de statistiques :** Génération de statistiques sur le temps de parole, le débit, et la dynamique d'interaction (*turn-taking*).
* **Export complet :** Sauvegarde de toutes les données et statistiques dans un fichier Excel multi-feuilles et génération de visualisations graphiques (timelines, histogrammes).

## Installation

### 1. Prérequis système
* **Python (3.8 ou supérieur)**
* **FFmpeg :** Cet outil est **essentiel** pour la conversion audio. Il doit être installé sur votre système et accessible depuis votre terminal. Instructions sur [ffmpeg.org](https://ffmpeg.org/download.html).

### 2. Cloner le dépôt
Clonez ce dépôt sur votre machine locale :
```bash
git clone [https://github.com/votre-nom-utilisateur/memoire-analyse-sonore-2025.git](https://github.com/votre-nom-utilisateur/memoire-analyse-sonore-2025.git)
cd memoire-analyse-sonore-2025
```

### 3. Modèle YAMNet local
Ce projet utilise une implémentation locale du modèle YAMNet. Assurez-vous que le répertoire `yamnet/` contenant les fichiers du modèle est présent à la racine du projet, tel que fourni.

### 4. Créer un environnement virtuel et installer les dépendances
Il est fortement recommandé d'utiliser un environnement virtuel.
```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Sur Windows:
venv\Scripts\activate
# Sur macOS/Linux:
source venv/bin/activate

# Installer les bibliothèques Python requises
pip install -r requirements.txt
```

### 5. Configurer les variables d'environnement
La diarisation via Pyannote nécessite une authentification auprès de Hugging Face.
1.  Créez un fichier nommé `.env` à la racine du projet.
2.  Ajoutez-y votre token d'accès Hugging Face :
    ```
    HUGGINGFACE_TOKEN="hf_VotreTokenIci"
    ```
Vous devez accepter les conditions d'utilisation des modèles `pyannote/speaker-diarization-3.1` et `pyannote/segmentation-3.0` sur le Hugging Face Hub.

## Utilisation

Le script principal s'exécute en ligne de commande.

**Option 1 : Passer le chemin du fichier en argument**
```bash
python main.py "chemin/vers/votre/fichier.mp3"
```

**Option 2 : Mode interactif**
Si aucun argument n'est fourni, le script vous demandera d'entrer le chemin du fichier.
```bash
python main.py
# 🟢 Entrez le chemin du fichier audio : ...
```

## Structure du projet

```
.
├── modules/                # Modules Python contenant la logique de l'analyse
│   ├── audio_loader.py
│   ├── transcription.py
│   ├── diarization_pyannote.py
│   ├── yamnet_analysis.py
│   ├── analysis_utils.py
│   └── exporter.py
├── yamnet/                   # Fichiers du modèle YAMNet local
│   └── ...
├── main.py                 # Script principal pour lancer le pipeline
├── requirements.txt        # Liste des dépendances Python
├── .env                    # Fichier pour les variables d'environnement (à créer)
└── README.md               # Ce fichier
```

Une fois l'analyse terminée, un répertoire `résultats_analyse/` sera créé avec un sous-dossier portant le nom de votre fichier audio, contenant le rapport Excel et tous les graphiques.

## Licence
Ce projet est mis à disposition sous la licence MIT. Voir le fichier `LICENSE` pour plus de détails.
