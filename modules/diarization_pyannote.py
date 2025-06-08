# modules/diarization_pyannote.py
import os
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# Charger le token depuis un fichier .env (ou remplace par ton token directement ici)
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

def run_diarization(audio_path, output_rttm_path="output.rttm"):
    """
    Exécute la diarisation sur un fichier audio et enregistre le résultat au format RTTM.
    :param audio_path: Chemin vers le fichier audio à analyser.
    :param output_rttm_path: Chemin vers le fichier RTTM de sortie.
    """
    if not HUGGINGFACE_TOKEN:
        # Il est important de gérer cette erreur, lever une exception est souvent le mieux.
        print("ERREUR CRITIQUE: HUGGINGFACE_TOKEN est manquant. Vérifie ton fichier .env.")
        raise ValueError("HUGGINGFACE_TOKEN est manquant. Vérifie ton fichier .env.")

    try:
        # Initialiser le pipeline officiel de pyannote
        print("Initialisation du pipeline Pyannote (pyannote/speaker-diarization-3.1)...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", # Ce nom pourrait être externalisé dans un config.py
            use_auth_token=HUGGINGFACE_TOKEN
        )
        print("Pipeline Pyannote initialisé.")
    except Exception as e:
        print(f"ERREUR: Échec de l'initialisation du pipeline Pyannote: {e}")
        return None # Indiquer un échec

    print(f"Analyse de diarisation pour : {audio_path}")
    try:
        diarization = pipeline(audio_path)
    except Exception as e:
        print(f"ERREUR: Échec du traitement par le pipeline Pyannote pour {audio_path}: {e}")
        return None # Indiquer un échec

    # --- Assurer que le répertoire de sortie pour RTTM existe ---
    rttm_directory = os.path.dirname(output_rttm_path)
    if rttm_directory: # S'assurer que ce n'est pas une chaîne vide
        try:
            os.makedirs(rttm_directory, exist_ok=True)
            print(f"INFO: Répertoire de sortie pour RTTM assuré/créé : {rttm_directory}")
        except OSError as e:
            print(f"ERREUR: Impossible de créer le répertoire {rttm_directory} pour le fichier RTTM : {e}")
            return None # Indiquer un échec si le répertoire ne peut être créé

    # Sauvegarder les résultats
    try:
        print(f"Sauvegarde des résultats de diarisation dans {output_rttm_path}")
        with open(output_rttm_path, "w") as f:
            diarization.write_rttm(f)
        print(f"Résultat RTTM sauvegardé : {output_rttm_path}")
    except FileNotFoundError:
        # Ce bloc est moins susceptible d'être atteint avec la création explicite du dossier ci-dessus,
        # mais on le garde pour le débogage au cas où.
        print(f"ERREUR FileNotFoundError inattendue lors de l'écriture du RTTM. Chemin: {output_rttm_path}. "
              f"Le répertoire {rttm_directory} aurait dû être créé.")
        return None
    except Exception as e:
        print(f"ERREUR lors de l'écriture du fichier RTTM {output_rttm_path}: {e}")
        return None # Indiquer un échec

    return diarization