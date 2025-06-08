# modules/audio_loader.py
import pandas as pd
import librosa
import numpy as np
import os
import subprocess
import sys
from modules.analysis_utils import format_time_ms

def convert_to_wav_16k_mono(input_path):
    base, _ = os.path.splitext(input_path)
    output_path = base + "_converted.wav"
    if not os.path.exists(output_path): # Vérifier si le fichier converti existe déjà
        cmd = [
            "ffmpeg", "-y", # -y pour écraser sans demander
            "-i", input_path,
            "-ac", "1",        # mono
            "-ar", "16000",    # 16kHz
            output_path
        ]
        print("🔄 Conversion du fichier audio en WAV mono 16kHz...")
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ Fichier converti : {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur lors de la conversion avec ffmpeg : {e}")
            # Gérer l'erreur, peut-être sortir ou retourner None
            return None # ou sys.exit(1)
        except FileNotFoundError:
            print("❌ Erreur: ffmpeg n'est pas installé ou n'est pas dans le PATH.")
            return None # ou sys.exit(1)
    else:
        print(f"☑️ Fichier converti existant trouvé : {output_path}")
    return output_path

def select_audio_file():
    if len(sys.argv) < 2:
        filename_input = input("🟢 Entrez le chemin du fichier audio : ").strip().strip("'").strip('"')
    else:
        filename_input = sys.argv[1]

    if not os.path.isfile(filename_input):
        print(f"❌ Fichier introuvable : {filename_input}")
        sys.exit(1)

    # Le fichier traité sera le WAV, même si l'entrée est un MP3
    filepath_for_processing = filename_input
    if filename_input.lower().endswith(".mp3"):
        filepath_for_processing = convert_to_wav_16k_mono(filename_input)
        if filepath_for_processing is None: # Échec de la conversion
            print("❌ Impossible de traiter le fichier audio après échec de conversion.")
            sys.exit(1)
            
    print(f"✅ Fichier sélectionné pour traitement : {filepath_for_processing}")
    
    # Charger les données audio à partir du fichier traité (WAV)
    try:
        audio_samples, sample_rate = librosa.load(filepath_for_processing, sr=None) # sr=None pour charger avec le SR natif (devrait être 16k)
        if sample_rate != 16000:
            print(f"⚠️ Attention: Le taux d'échantillonnage est de {sample_rate}Hz et non 16000Hz comme attendu après conversion.")
            # Vous pourriez forcer le rééchantillonnage ici si nécessaire, mais la conversion devrait s'en charger.
            # audio_samples = librosa.resample(audio_samples, orig_sr=sample_rate, target_sr=16000)
            # sample_rate = 16000

    except Exception as e:
        print(f"❌ Erreur lors du chargement du fichier audio avec Librosa : {e}")
        sys.exit(1)
        
    return filepath_for_processing, audio_samples, sample_rate

def get_audio_duration(filepath: str) -> float:
    """Retourne la durée d'un fichier audio en secondes."""
    try:
        return librosa.get_duration(path=filepath)
    except Exception as e:
        print(f"❌ Impossible d'obtenir la durée pour {filepath}: {e}")
        return 0.0

def extract_features(
    audio_samples: np.ndarray, 
    sample_rate: int, 
    fixed_silence_threshold: float = None, # Seuil fixe optionnel
    adaptive_percentile: int = 10,        # Percentile pour estimer le plancher de son actif
    adaptive_factor: float = 0.8          # Facteur multiplicatif pour le seuil adaptatif
    ):
    """
    Extrait les caractéristiques audio par fenêtres de 1 seconde.
    Permet un seuil de silence fixe ou adaptatif.

    Args:
        audio_samples (np.ndarray): Les échantillons audio.
        sample_rate (int): Le taux d'échantillonnage.
        fixed_silence_threshold (float, optional): Si fourni, utilise ce seuil RMS fixe.
                                                   Sinon, un seuil adaptatif est calculé.
        adaptive_percentile (int): Percentile (0-100) des RMS non nuls à utiliser comme
                                   niveau de référence pour le seuil adaptatif.
        adaptive_factor (float): Facteur à appliquer au niveau de référence pour obtenir
                                 le seuil adaptatif (ex: 0.5 signifie 50% du niveau de référence).
    Returns:
        pd.DataFrame
    """
    frame_duration_sec = 1.0
    hop_length_samples = int(frame_duration_sec * sample_rate)

    num_frames_total = int(np.floor(len(audio_samples) / hop_length_samples))
    
    raw_features_data = [] # Pour stocker les dictionnaires avant de déterminer le seuil adaptatif

    if num_frames_total == 0: # Audio trop court
        print("Avertissement: Audio trop court pour extraire des features avec une fenêtre de 1s.")
        columns = ['start', 'end', 'rms', 'zcr', 'is_silence'] + [f'mfcc_{j+1}' for j in range(13)]
        return pd.DataFrame(columns=columns)

    for i in range(num_frames_total):
        start_sample = i * hop_length_samples
        end_sample = start_sample + hop_length_samples
        frame = audio_samples[start_sample:end_sample]

        if len(frame) == 0:
            continue

        start_time_sec = start_sample / sample_rate
        end_time_sec = end_sample / sample_rate
        
        rms_value = np.sqrt(np.mean(frame**2)) # Calcul RMS simple
        zcr_value = np.mean(librosa.feature.zero_crossing_rate(y=frame, frame_length=len(frame), hop_length=len(frame)+1)[0])
        mfccs = librosa.feature.mfcc(
            y=frame,
            sr=sample_rate,
            n_mfcc=13,
            n_fft=2048,
            hop_length=hop_length_samples // 4 if hop_length_samples // 4 > 0 else 512
        ) # hop_length plus petit pour MFCC
        mfcc_mean_values = np.mean(mfccs, axis=1)
        
        row_data = {
            'start': start_time_sec,
            'end': end_time_sec,
            'rms': rms_value,
            'zcr': zcr_value,
            # 'is_silence' sera ajouté après le calcul du seuil
        }
        for j in range(13):
            row_data[f'mfcc_{j+1}'] = mfcc_mean_values[j]
        
        raw_features_data.append(row_data)

    df_raw_features = pd.DataFrame(raw_features_data)

    # Détermination du seuil de silence
    actual_silence_threshold = 0.0 # Initialisation
    if fixed_silence_threshold is not None:
        actual_silence_threshold = fixed_silence_threshold
        print(f"ℹ️ Utilisation du seuil de silence fixe : {actual_silence_threshold:.4f}")
    else:
        # Calcul du seuil adaptatif
        all_rms_values = df_raw_features["rms"][df_raw_features["rms"] > 1e-6] # Ignorer les RMS quasi nuls
        if not all_rms_values.empty:
            # Niveau de référence basé sur le percentile des RMS significatifs
            reference_rms_level = np.percentile(all_rms_values, adaptive_percentile)
            actual_silence_threshold = reference_rms_level * adaptive_factor
            print(f"ℹ️ Calcul du seuil de silence adaptatif : {actual_silence_threshold:.4f} "
                  f"(basé sur le {adaptive_percentile}e percentile = {reference_rms_level:.4f} * facteur {adaptive_factor})")
        else:
            # Fallback si l'audio est complètement silencieux ou si tous les RMS sont trop bas
            actual_silence_threshold = 0.01 # Valeur de fallback
            print(f"⚠️ Impossible de calculer le seuil adaptatif (peu de son détecté), "
                  f"utilisation du seuil de fallback : {actual_silence_threshold:.4f}")

    # Appliquer le seuil pour déterminer 'is_silence'
    df_raw_features["is_silence"] = df_raw_features["rms"] < actual_silence_threshold

    if not df_raw_features.empty:
     df_raw_features['start_ms'] = df_raw_features['start'].apply(format_time_ms)
     df_raw_features['end_ms'] = df_raw_features['end'].apply(format_time_ms)

        
    return df_raw_features