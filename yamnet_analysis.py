# fichier: modules/yamnet_analysis.py

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf # Assurez-vous que tensorflow est importé

# Importation de votre modèle YAMNet local (comme dans votre code original)
from yamnet.yamnet_model import yamnet_model
# tqdm n'est pas utilisé directement dans cette version de la fonction

def classify_sounds(filename, score_threshold=0.3):

    # --- Chargement du modèle et des noms de classes (selon votre script original) ---
    model = yamnet_model() 

    class_map_url = (
        "https://raw.githubusercontent.com/tensorflow/models"
        "/master/research/audioset/yamnet/yamnet_class_map.csv"
    )
    try:
        # Tentative de lecture du fichier CSV contenant les noms des classes
        class_names_df = pd.read_csv(class_map_url)
        if "display_name" not in class_names_df.columns:
            print(f"Avertissement: La colonne 'display_name' est introuvable dans {class_map_url}.")
            # Solution de repli si 'display_name' n'existe pas
            if "mid" in class_names_df.columns:
                 class_names = class_names_df["mid"].tolist()
            else: # En dernier recours, prendre la première colonne
                 class_names = class_names_df.iloc[:, 0].tolist()
        else:
            class_names = class_names_df["display_name"].tolist()
    except Exception as e:
        print(f"Erreur lors du chargement ou de l'analyse du CSV des classes depuis {class_map_url}: {e}")
        # Les noms de classes sont essentiels ; levez une exception si le chargement échoue.
        raise

    # --- Chargement et prétraitement de l'audio ---
    # YAMNet attend un taux d'échantillonnage de 16kHz et un signal mono en float32.
    waveform, sr = librosa.load(filename, sr=16000, mono=True)
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)

# --- Sound Classification (SECTION RÉVISÉE) ---
    scores_input_to_process = None # Variable pour stocker le composant des scores bruts
    try:
        # Basé sur votre code original : scores, embeddings, spectrogram = model(waveform)
        # Cela implique que model(waveform) renvoie trois composants.
        # Nous nous attendons à ce que s, e, sp soient des tenseurs TensorFlow.
        s, e, sp = model(waveform) 
        scores_input_to_process = s # Le premier composant est supposé être le tenseur des scores
        
    except ValueError: # Si model(waveform) ne renvoie pas 3 composants à déballer
        print("Avertissement : model(waveform) n'a pas retourné 3 composants comme attendu. "
              "Tentative d'appel au modèle en supposant qu'il retourne les scores directement "
              "ou comme premier élément d'un tuple.")
        try:
            potential_scores_output = model(waveform)
            if isinstance(potential_scores_output, tuple) and len(potential_scores_output) > 0:
                # Si c'est un tuple, prendre le premier élément
                scores_input_to_process = potential_scores_output[0] 
            else:
                # Sinon, supposer que c'est directement le tenseur des scores
                scores_input_to_process = potential_scores_output 
        except Exception as e_alt_call:
            print(f"CRITIQUE : L'appel alternatif au modèle a échoué : {e_alt_call}")
            raise # Relancer l'exception, car nous ne pouvons pas obtenir les scores
    except Exception as e_model_call:
        print(f"CRITIQUE : Erreur lors de l'appel au modèle YAMNet : {e_model_call}")
        raise # Relancer l'exception

    # Vérifier si scores_input_to_process a été correctement assigné
    if scores_input_to_process is None:
        print("CRITIQUE : Impossible de récupérer les scores depuis le modèle.")
        return pd.DataFrame(columns=["start", "end", "labels", "duration"])

    # --- Conversion du composant des scores en tableau NumPy (SECTION RÉVISÉE) ---
    try:
        if hasattr(scores_input_to_process, 'numpy'):
            # Méthode standard pour les tenseurs TensorFlow
            scores = scores_input_to_process.numpy() 
        elif isinstance(scores_input_to_process, np.ndarray):
            # Si c'est déjà un tableau NumPy
            scores = scores_input_to_process 
        else:
            # Tentative de conversion directe pour d'autres types (ex: liste de listes)
            # Spécifier dtype=np.float32 pour assurer l'homogénéité numérique attendue pour les scores YAMNet.
            print(f"Avertissement : le composant des scores est de type {type(scores_input_to_process)}, "
                  "tentative de conversion directe en np.array avec dtype=np.float32.")
            scores = np.array(scores_input_to_process, dtype=np.float32)
    except Exception as e_conversion:
        print(f"CRITIQUE : Erreur lors de la conversion du composant des scores en tableau NumPy : {e_conversion}")
        print(f"Le type de scores_input_to_process était : {type(scores_input_to_process)}")
        # C'est ici que l'erreur originale s'est probablement produite si scores_input_to_process 
        # n'était pas le tenseur/tableau des scores lui-même mais une structure plus complexe (ex: le tuple entier).
        raise

    # --- Validation du tableau `scores` (légèrement ajustée pour plus de robustesse) ---
    if not isinstance(scores, np.ndarray):
        print(f"CRITIQUE : La variable 'scores' n'est pas un tableau NumPy après les tentatives de conversion. "
              f"Type actuel : {type(scores)}")
        return pd.DataFrame(columns=["start", "end", "labels", "duration"])

    if scores.ndim != 2:
        print(f"CRITIQUE : Le tableau des scores n'est pas bi-dimensionnel. Dimensions : {scores.shape}")
        return pd.DataFrame(columns=["start", "end", "labels", "duration"])

    # Vérifier que le nombre de classes dans les scores correspond au nombre de class_names
    if scores.shape[1] != len(class_names):
        print(f"CRITIQUE : Le nombre de classes dans les scores ({scores.shape[1]}) ne correspond pas "
              f"à la longueur de class_names ({len(class_names)}). Dimensions des scores : {scores.shape}")
        return pd.DataFrame(columns=["start", "end", "labels", "duration"])
        
    # --- Process Scores for Multi-Label Output ---
    # Le reste de cette section (calcul des frame_hop_seconds, boucle sur les trames,
    # détection des labels multiples, etc.) et la section de fusion des segments
    # restent les mêmes que dans ma réponse précédente. Assurez-vous que cette partie commence APRÈS
    # la validation de `scores` ci-dessus.
    frame_hop_seconds = 0.48 
    results_per_frame = []
    # ... (la suite du code pour le traitement multi-label et la fusion des segments reste inchangée)


    # --- Traitement des scores pour une sortie multi-label ---
    # YAMNet produit des scores pour des trames avec un pas de 0.48s.
    frame_hop_seconds = 0.48 
    results_per_frame = []  
    num_frames = scores.shape[0]

    for i in range(num_frames):
        frame_scores = scores[i]
        # Indices des classes dont le score dépasse le seuil
        detected_indices = np.where(frame_scores > score_threshold)[0]
        
        # Labels correspondants (en s'assurant que l'indice est valide)
        detected_labels_for_frame = [class_names[idx] for idx in detected_indices if idx < len(class_names)]
        
        start_time = i * frame_hop_seconds
        end_time = start_time + frame_hop_seconds # Durée de cette trame élémentaire

        if detected_labels_for_frame: # Ajouter seulement si des labels sont détectés
            results_per_frame.append({
                "start": start_time,
                "end": end_time,
                "labels": detected_labels_for_frame # 'labels' est maintenant une liste
            })

    if not results_per_frame: # Aucun son détecté au-dessus du seuil
        return pd.DataFrame(columns=["start", "end", "labels", "duration"])

    df_yamnet_frames = pd.DataFrame(results_per_frame)

    # --- Fusion des trames consécutives avec des ensembles de labels identiques ---
    # Conversion de la liste de labels en un tuple trié pour permettre le regroupement
    df_yamnet_frames['label_set_tuple'] = df_yamnet_frames['labels'].apply(lambda x: tuple(sorted(x)))
    # Conversion en chaîne pour faciliter les comparaisons de groupes
    df_yamnet_frames['label_set_str'] = df_yamnet_frames['label_set_tuple'].astype(str)

    # Identification des blocs : un nouveau bloc commence si 'label_set_str' change 
    # OU s'il y a une discontinuité temporelle.
    df_yamnet_frames['time_gap'] = (df_yamnet_frames['start'] > (df_yamnet_frames['end'].shift(1) + 0.01)).fillna(False)
    df_yamnet_frames['label_changed'] = (df_yamnet_frames['label_set_str'] != df_yamnet_frames['label_set_str'].shift(1)).fillna(False)
    
    df_yamnet_frames['is_new_block'] = df_yamnet_frames['time_gap'] | df_yamnet_frames['label_changed']
    # La première ligne initie toujours un nouveau bloc.
    if not df_yamnet_frames.empty:
        df_yamnet_frames.loc[df_yamnet_frames.index[0], 'is_new_block'] = True
    df_yamnet_frames['block_id'] = df_yamnet_frames['is_new_block'].cumsum()

    merged_segments_data = []
    if not df_yamnet_frames.empty:
        grouped = df_yamnet_frames.groupby('block_id')
        for _, group_df in grouped:
            merged_segments_data.append({
                'start': group_df['start'].min(),
                'end': group_df['end'].max(),
                # Conserver la liste originale (non triée) des labels de la première trame du bloc
                'labels': group_df['labels'].iloc[0] 
            })

    df_yamnet_merged = pd.DataFrame(merged_segments_data)

    if not df_yamnet_merged.empty:
        df_yamnet_merged["duration"] = df_yamnet_merged["end"] - df_yamnet_merged["start"]
        # Filtrer les segments par une durée minimale (ex: 0.1s)
        # Ce filtrage est plus simple que celui basé sur la durée totale par label fait précédemment.
        df_yamnet_final = df_yamnet_merged[df_yamnet_merged["duration"] >= 0.1].copy() 
        if df_yamnet_final.empty: # Si tout est filtré, retourner un DataFrame vide avec les bonnes colonnes
             return pd.DataFrame(columns=["start", "end", "labels", "duration"])
    else: # Si aucun segment n'a été fusionné (cas où df_yamnet_frames était vide)
        df_yamnet_final = pd.DataFrame(columns=["start", "end", "labels", "duration"])

    return df_yamnet_final