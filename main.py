# main.py
from dotenv import load_dotenv
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Pour éviter les problèmes avec certaines installations de matplotlib/torch sur Mac

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import librosa # Ajouté pour get_duration si utilisé directement ici, sinon via audio_loader
import matplotlib.ticker as mticker # Pour l'axe temporel amélioré

# Importer les modules personnalisés
from modules.audio_loader import select_audio_file, extract_features, get_audio_duration
from modules.transcription import transcribe_audio
from modules.diarization_pyannote import run_diarization
from modules.yamnet_analysis import classify_sounds # Assurez-vous que c'est la version multi-label corrigée
from modules.exporter import export_to_excel # Assurez-vous que c'est la version mise à jour
from modules.analysis_utils import (
    assign_speakers_and_compute_stats,
    compute_turn_taking_stats, 
    process_pyannote_annotation, # Nouvelle fonction pour parole unique/superposée
    create_polyphonic_timeline,   # Nouvelle fonction pour la timeline unifiée
    format_time_ms # Assurez-vous que cette fonction est importée si vous l'utilisez dans les graphiques
)
from tqdm import tqdm # Si vous l'utilisez pour des boucles personnalisées

# Charger les variables d'environnement (ex: HUGGINGFACE_TOKEN)
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN") # Utilisé par diarization_pyannote.py

# --- Définitions des Fonctions de Sauvegarde des Graphiques ---

def save_timeline_image(df, speaker_col_name, output_dir_path, plot_suffix=""):
    """Sauvegarde une image de la timeline des locuteurs."""
    if df is None or df.empty:
        print(f"Avertissement: DataFrame pour la timeline '{plot_suffix}' est vide. Graphique non généré.")
        return

    if speaker_col_name not in df.columns:
        print(f"Avertissement: Colonne '{speaker_col_name}' non trouvée pour la timeline '{plot_suffix}'. Graphique non généré.")
        return

    # Création de la figure et des axes APRÈS les vérifications
    num_unique_speakers = df[speaker_col_name].nunique()
    fig, ax = plt.subplots(figsize=(12, max(3, num_unique_speakers * 0.5))) 
    
    speakers = sorted(df[speaker_col_name].unique()) 
    colors = plt.cm.get_cmap('tab10', len(speakers) if len(speakers) > 0 else 1).colors 

    speaker_to_y_pos = {speaker: i for i, speaker in enumerate(speakers)}

    for _, row in df.iterrows():
        speaker = row[speaker_col_name]
        y_pos = speaker_to_y_pos.get(speaker)
        if y_pos is not None: 
            ax.plot([row["start"], row["end"]], [y_pos, y_pos], 
                    color=colors[y_pos % len(colors)], lw=6)

    ax.set_yticks(list(speaker_to_y_pos.values()))
    ax.set_yticklabels(list(speaker_to_y_pos.keys()))
    ax.set_xlabel("Temps (s)") # Assurez-vous que c'est la bonne étiquette
    ax.set_title(f"Timeline – {plot_suffix or speaker_col_name}")
    plt.tight_layout()
    
    image_filename = f"timeline_{plot_suffix or speaker_col_name}.png"
    full_image_path = os.path.join(output_dir_path, image_filename)
    
    plt.savefig(full_image_path)
    print(f"📊 Timeline sauvegardée : {full_image_path}")
    plt.close(fig)


def save_db_plot(df_features, output_dir_path):
    """Sauvegarde un graphique de l'évolution du volume (RMS en dB)."""
    if df_features is None or df_features.empty or 'rms' not in df_features.columns:
        print("Avertissement: df_features est vide ou ne contient pas 'rms'. Graphique RMS non généré.")
        return

    df_plot = df_features.copy()
    
    epsilon = 1e-9 
    df_plot["rms_safe"] = df_plot["rms"].replace(0, epsilon)
    df_plot.loc[df_plot["rms_safe"] <= 0, "rms_safe"] = epsilon 
    
    df_plot["rms_db"] = 20 * np.log10(df_plot["rms_safe"])
    df_plot["rms_db"] = df_plot["rms_db"].fillna(-100).clip(lower=-100) 

    plt.figure(figsize=(12, 3))
    plt.plot(df_plot["start"], df_plot["rms_db"])
    plt.title("Évolution du volume (dB RMS)")
    plt.xlabel("Temps (s)")
    plt.ylabel("Niveau (dB)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    image_filename = "volume_rms_db.png"
    full_image_path = os.path.join(output_dir_path, image_filename)
    plt.savefig(full_image_path)
    print(f"📊 Graphique volume sauvegardé : {full_image_path}")
    plt.close()


def save_yamnet_distribution(df_yamnet_multilabel, output_dir_path):
    """
    Génère et sauvegarde un graphique de la distribution des types de sons YAMNet.
    Adaptée pour une colonne 'labels' contenant des listes de labels.
    """
    if df_yamnet_multilabel is None or df_yamnet_multilabel.empty or 'labels' not in df_yamnet_multilabel.columns:
        print("Avertissement : df_yamnet_multilabel est vide ou ne contient pas la colonne 'labels'. Graphique YAMNet ignoré.")
        return pd.DataFrame(columns=['label', 'total_duration']), pd.DataFrame(columns=["start", "end", "labels", "duration"])

    df_plot = df_yamnet_multilabel.copy()
    if 'duration' not in df_plot.columns:
         df_plot["duration"] = df_plot["end"] - df_plot["start"]

    df_exploded = df_plot.explode('labels')

    if df_exploded.empty or 'labels' not in df_exploded.columns:
        print("Avertissement : df_exploded est vide après 'explode'. Graphique YAMNet ignoré.")
        return pd.DataFrame(columns=['label', 'total_duration']), df_plot if df_plot is not None else pd.DataFrame(columns=["start", "end", "labels", "duration"])


    durations_by_label = df_exploded.groupby('labels')['duration'].sum()
    labels_to_keep = durations_by_label[durations_by_label >= 1.0].index 
    
    dist_stats = durations_by_label[durations_by_label.index.isin(labels_to_keep)].sort_values(ascending=False).head(15)
    
    if not dist_stats.empty:
        plt.figure(figsize=(12, 7))
        dist_stats.plot(kind="bar")
        plt.title("Top 15 des sons YAMNet identifiés (durée cumulée >= 1s)")
        plt.ylabel("Durée Totale Cumulée (s)")
        plt.xlabel("Type de Son (YAMNet)")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, linestyle=':', alpha=0.7, axis='y')
        plt.tight_layout()
        
        image_filename = "yamnet_distribution.png"
        full_image_path = os.path.join(output_dir_path, image_filename)
        plt.savefig(full_image_path)
        print(f"📊 Graphique distribution YAMNet sauvegardé : {full_image_path}")
        plt.close()
    else:
        print("Avertissement : Aucune catégorie sonore YAMNet n'atteint le seuil pour le graphique de distribution.")

    if not labels_to_keep.empty:
        df_yamnet_to_export = df_plot[
            df_plot['labels'].apply(lambda label_list: any(label in labels_to_keep for label in label_list))
        ].copy()
    else:
        df_yamnet_to_export = pd.DataFrame(columns=df_plot.columns) 
    
    dist_stats_df = dist_stats.reset_index()
    dist_stats_df.columns = ['label', 'total_duration']

    return dist_stats_df, df_yamnet_to_export


def save_silence_histogram(df_features, output_dir_path, silence_col_name="is_silence", time_col_name="start"):
    """Sauvegarde un histogramme de la durée des segments de silence."""
    if df_features is None or df_features.empty or silence_col_name not in df_features.columns:
        print(f"Avertissement: df_features est vide ou ne contient pas '{silence_col_name}'. Histogramme des silences non généré.")
        return pd.DataFrame(columns=["durée_silence_s"])

    silence_durations = []
    current_silence_start_time = None
    
    segment_duration = (df_features['end'] - df_features['start']).median() if not df_features.empty else 1.0

    for index, row in df_features.iterrows():
        if row[silence_col_name]:
            if current_silence_start_time is None:
                current_silence_start_time = row[time_col_name]
        else:
            if current_silence_start_time is not None:
                silence_durations.append(row[time_col_name] - current_silence_start_time)
                current_silence_start_time = None
    
    if current_silence_start_time is not None and not df_features.empty:
        silence_durations.append(df_features['end'].iloc[-1] - current_silence_start_time)

    if not silence_durations:
        print("Info: Aucun segment de silence détecté pour l'histogramme.")
        return pd.DataFrame(columns=["durée_silence_s"])

    df_silence_stats = pd.DataFrame({"durée_silence_s": silence_durations})
    
    plt.figure(figsize=(10, 4))
    df_silence_stats["durée_silence_s"].plot(kind="hist", bins=30, edgecolor='black')
    plt.title("Distribution de la durée des segments de silence")
    plt.xlabel("Durée du silence (s)")
    plt.ylabel("Fréquence")
    plt.grid(True, linestyle=':', alpha=0.7, axis='y')
    plt.tight_layout()
    
    image_filename = "silence_histogram.png"
    full_image_path = os.path.join(output_dir_path, image_filename)
    plt.savefig(full_image_path)
    print(f"📊 Histogramme silences sauvegardé : {full_image_path}")
    plt.close()
    
    return df_silence_stats

def save_polyphonic_timeline_plot(
    df_poly_timeline: pd.DataFrame,
    output_dir_path: str,
    yamnet_categories_to_plot: list = ["Music", "Speech", "Silence_YamNet", "Effects"],
    time_step: float = 0.5,
    total_audio_duration_seconds: float = None # Ajouter la durée totale pour mieux définir les ticks
    ):
    """
    Sauvegarde une visualisation de la timeline polyphonique avec un axe temporel amélioré.
    """
    if df_poly_timeline is None or df_poly_timeline.empty:
        print("Avertissement: df_poly_timeline est vide. Graphique de la timeline polyphonique non généré.")
        return

    # --- Début de la logique existante pour préparer les pistes (plot_tracks, track_order) ---
    all_speakers = sorted(list(set(
        spk for sublist in df_poly_timeline["active_speakers"].explode().dropna() if sublist is not None and sublist != 'None' for spk in ([sublist] if isinstance(sublist, str) else sublist)
    )))
    
    plot_tracks = {}
    track_order = [] 

    for spk in all_speakers:
        track_name = f"Parole: {spk}"
        plot_tracks[track_name] = []
        track_order.append(track_name)

    # Piste pour la parole superposée (si plus d'un locuteur)
    # Vous pouvez la réactiver si besoin
    # track_overlapped_speech = "Parole Superposée"
    # plot_tracks[track_overlapped_speech] = []
    # track_order.append(track_overlapped_speech)

    for category in yamnet_categories_to_plot:
        # S'assurer que la catégorie est valide (par exemple, issue d'un regroupement)
        # ou qu'elle correspond à des labels dans df_poly_timeline['yamnet_labels']
        plot_tracks[category] = []
        track_order.append(category)
    
    track_silence_rms = "Silence (RMS)"
    plot_tracks[track_silence_rms] = []
    track_order.append(track_silence_rms)

    for _, row in df_poly_timeline.iterrows():
        start_time = row["time_start"]
        end_time = row["time_end"] 

        active_spk_list = row["active_speakers"]
        if len(active_spk_list) == 1 and active_spk_list[0] != 'None': # Vérifier 'None' comme chaîne ici
            spk_track_name = f"Parole: {active_spk_list[0]}"
            if spk_track_name in plot_tracks:
                 plot_tracks[spk_track_name].append((start_time, end_time))
        elif len(active_spk_list) > 1:
            for spk_o in active_spk_list: # Marquer chaque locuteur impliqué dans l'overlap
                spk_track_name_o = f"Parole: {spk_o}"
                if spk_track_name_o in plot_tracks:
                    plot_tracks[spk_track_name_o].append((start_time, end_time))

        yamnet_labels_in_row = row["yamnet_labels"]
        for category in yamnet_categories_to_plot:
            # Si `category` est un label YAMNet direct présent dans `yamnet_labels_in_row`
            # ou si vous avez une logique de mapping pour regrouper les labels bruts en ces catégories.
            # Exemple simple (suppose que category est un label brut recherché)
            if category in yamnet_labels_in_row:
                 plot_tracks[category].append((start_time, end_time))

        if row["is_silence"]: 
            plot_tracks[track_silence_rms].append((start_time, end_time))

    for track_name in plot_tracks:
        segments = plot_tracks[track_name]
        if not segments:
            continue
        
        merged_segments = []
        if segments:
            # Les segments sont déjà triés car ils viennent de df_poly_timeline qui est trié par temps
            current_merged_start, current_merged_end = segments[0]
            for next_start, next_end in segments[1:]:
                if abs(next_start - current_merged_end) < (time_step / 2 + 1e-9) : 
                    current_merged_end = next_end 
                else:
                    merged_segments.append((current_merged_start, current_merged_end))
                    current_merged_start, current_merged_end = next_start, next_end
            merged_segments.append((current_merged_start, current_merged_end)) 
        plot_tracks[track_name] = merged_segments
    # --- Fin de la logique existante pour préparer les pistes ---

    # Création du graphique
    num_tracks = len(track_order)
    fig_height = max(4, num_tracks * 0.55) # Un peu plus d'espace par piste
    fig, ax = plt.subplots(figsize=(18, fig_height)) # Augmenter la largeur pour plus de détails temporels

    # Choix d'une colormap plus distincte si possible, ou s'assurer que les couleurs se répètent bien.
    # 'tab20' a 20 couleurs distinctes, 'tab10' en a 10.
    num_colors_needed = num_tracks
    if num_colors_needed <= 10:
        colors = plt.cm.get_cmap('tab10', num_colors_needed).colors
    elif num_colors_needed <= 20:
        colors = plt.cm.get_cmap('tab20', num_colors_needed).colors
    else: # Plus de 20 pistes, les couleurs vont se répéter avec tab20, ou utiliser une autre cmap
        colors = plt.cm.get_cmap('nipy_spectral', num_colors_needed).colors


    for i, track_name in enumerate(track_order):
        y_pos = i
        segments_to_draw = plot_tracks.get(track_name, [])
        for start, end in segments_to_draw:
            # Dessiner les segments comme des barres horizontales pour une meilleure distinction
            ax.barh(y_pos, width=(end - start), left=start, height=0.7, 
                    color=colors[i % len(colors)], align='center', edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(num_tracks))
    ax.set_yticklabels(track_order, fontsize=9) # Ajuster la taille de la police si besoin
    ax.set_xlabel("Temps (minutes:secondes)", fontsize=10)
    ax.set_title("Timeline Polyphonique Détaillée", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.7, axis='x') # Grille verticale pour le temps
    ax.invert_yaxis() # Pour que la première piste soit en haut

    # --- AMÉLIORATION DE L'AXE TEMPOREL ---
    if total_audio_duration_seconds is None and not df_poly_timeline.empty:
        total_audio_duration_seconds = df_poly_timeline['time_end'].max()
    
    if total_audio_duration_seconds:
        ax.set_xlim(0, total_audio_duration_seconds) # Définir les limites de l'axe X

        # Formatter pour afficher les secondes en "minutes:secondes"
        def format_time_ticks(x_seconds, pos):
            minutes = int(x_seconds // 60)
            seconds = int(x_seconds % 60)
            return f"{minutes:02d}:{seconds:02d}"
        
        formatter = mticker.FuncFormatter(format_time_ticks)
        ax.xaxis.set_major_formatter(formatter)

        # Déterminer dynamiquement l'intervalle des graduations principales
        # Objectif : avoir entre 5 et 15 graduations principales
        if total_audio_duration_seconds <= 60: # Moins d'1 minute
            major_tick_interval = 10 # Toutes les 10 secondes
        elif total_audio_duration_seconds <= 300: # Jusqu'à 5 minutes
            major_tick_interval = 30 # Toutes les 30 secondes
        elif total_audio_duration_seconds <= 900: # Jusqu'à 15 minutes
            major_tick_interval = 60 # Toutes les minutes
        elif total_audio_duration_seconds <= 1800: # Jusqu'à 30 minutes
            major_tick_interval = 120 # Toutes les 2 minutes
        else: # Plus de 30 minutes
            major_tick_interval = 300 # Toutes les 5 minutes
        
        ax.xaxis.set_major_locator(mticker.MultipleLocator(major_tick_interval))
        
        # Optionnel : Ajouter des graduations mineures pour plus de précision
        if major_tick_interval > 10:
             minor_tick_interval = major_tick_interval / 5 # Par exemple, 5 graduations mineures entre chaque majeure
             if minor_tick_interval >= 5: # Avoir au moins 5s entre les mineures
                ax.xaxis.set_minor_locator(mticker.MultipleLocator(minor_tick_interval))


    plt.xticks(rotation=30, ha='right', fontsize=9) # Rotation pour lisibilité
    plt.tight_layout(pad=1.5) # Ajouter un peu de padding
    
    image_filename = "timeline_polyphonique_detaillee_mmss.png" # Nouveau nom de fichier
    full_image_path = os.path.join(output_dir_path, image_filename)
    try:
        plt.savefig(full_image_path, dpi=150) # Augmenter un peu le DPI pour une meilleure qualité
        print(f"📊 Timeline polyphonique détaillée (mmss) sauvegardée : {full_image_path}")
    except Exception as e_save:
        print(f"❌ Erreur lors de la sauvegarde du graphique de la timeline polyphonique : {e_save}")
    plt.close(fig)

# --- Pipeline Principal ---
if __name__ == "__main__":
    try:
        import multiprocessing
        if multiprocessing.get_start_method(allow_none=True) != "fork":
            pass
    except Exception as e_mp:
        print(f"Note: Problème avec la configuration de multiprocessing: {e_mp}")


    print("--- Début du Pipeline d'Analyse Audio ---")

    print("🎧 Chargement du fichier audio...")
    filepath_for_processing, audio_samples, sample_rate = select_audio_file()
    if filepath_for_processing is None:
        print("❌ Arrêt du pipeline : échec du chargement du fichier audio.")
        exit()
    print(f"✅ Fichier chargé pour traitement : {filepath_for_processing}")

    base_name = os.path.splitext(os.path.basename(filepath_for_processing))[0]
    safe_base_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in base_name).rstrip()
    output_dir = os.path.join("résultats_analyse", safe_base_name) 
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Répertoire de sortie : {output_dir}")

    print("🎚️ Extraction des features audio (RMS, ZCR, MFCC, Silence)...")
    df_features = extract_features(audio_samples, sample_rate)
    print(f"✅ Features extraites ({len(df_features)} segments de 1s).")

    print("📝 Transcription de l'audio (Whisper)...")
    df_transcript = transcribe_audio(filepath_for_processing) 
    print(f"✅ Transcription terminée ({len(df_transcript)} segments).")

    print("🤖 Diarisation Pyannote en cours (HuggingFace)...")
    pyannote_annotation_object = run_diarization(filepath_for_processing, output_rttm_path=os.path.join(output_dir, f"{safe_base_name}.rttm"))
    
    df_pyannote_single_speaker, df_pyannote_overlapped = pd.DataFrame(), pd.DataFrame() # Initialisation
    if pyannote_annotation_object: # S'assurer que l'objet n'est pas None
        print("Traitement des segments parole unique et superposée...")
        df_pyannote_single_speaker, df_pyannote_overlapped = process_pyannote_annotation(pyannote_annotation_object)
        print(f"✅ Diarisation traitée : {len(df_pyannote_single_speaker)} segments uniques, {len(df_pyannote_overlapped)} segments superposés.")
        if not df_pyannote_overlapped.empty:
            print("Aperçu des segments de parole superposée:")
            print(df_pyannote_overlapped.head())
    else:
        print("❌ Échec de la diarisation Pyannote. Certaines analyses dépendantes pourraient être affectées.")
        # Créer des DataFrames vides avec les colonnes attendues pour éviter les erreurs en aval
        # Les colonnes formatées _ms sont ajoutées par process_pyannote_annotation, donc on met les colonnes de base
        df_pyannote_single_speaker = pd.DataFrame(columns=["start", "end", "speaker", "duration"])
        df_pyannote_overlapped = pd.DataFrame(columns=["start", "end", "speakers_involved", "num_speakers", "duration"])


    print("🎯 Attribution des locuteurs aux transcriptions et calcul des statistiques de parole...")
    # Préparer un df_diar_for_assign vide avec la colonne attendue si df_pyannote_single_speaker est vide
    df_diar_for_assign = pd.DataFrame(columns=['start', 'end', 'speaker_pyannote'])
    if not df_pyannote_single_speaker.empty and "speaker" in df_pyannote_single_speaker.columns:
        df_diar_for_assign = df_pyannote_single_speaker.rename(columns={"speaker": "speaker_pyannote"})

    df_transcript_enriched, general_speech_stats, df_speaker_stats = assign_speakers_and_compute_stats(
        df_transcript, 
        df_diar_for_assign
    )
    print("✅ Attribution et statistiques de parole terminées.") 

    print("🔊 Classification sonore YAMNet (multi-label)...")
    df_yamnet = classify_sounds(filepath_for_processing, score_threshold=0.3) 
    print(f"✅ YAMNet terminé ({len(df_yamnet)} segments d'événements sonores).")

    print("⏱️ Calcul de la durée totale de l'audio...")
    total_duration_sec = get_audio_duration(filepath_for_processing)
    if total_duration_sec > 0:
        print(f"✅ Durée totale : {total_duration_sec:.2f} secondes.")
    else:
        print("⚠️ Durée totale de l'audio est nulle ou non déterminée.")
        if not df_features.empty:
            total_duration_sec = df_features['end'].max()
            print(f"Utilisation de la fin des features comme durée: {total_duration_sec:.2f}s")
        else:
            total_duration_sec = 0 

    df_poly_timeline = pd.DataFrame() 
    if total_duration_sec > 0:
        print("🎼 Création de la timeline polyphonique unifiée...")
        df_poly_timeline = create_polyphonic_timeline(
            total_audio_duration_seconds=total_duration_sec,
            df_single_speaker=df_pyannote_single_speaker,
            df_overlapped_speech=df_pyannote_overlapped,
            df_yamnet_multilabel=df_yamnet,
            df_audio_features=df_features,
            time_step=0.5 
        )
        print(f"✅ Timeline polyphonique créée ({len(df_poly_timeline)} pas de temps).")
        if not df_poly_timeline.empty:
            print("Aperçu de la timeline polyphonique :")
            print(df_poly_timeline.head())
    else:
        print("❌ Timeline polyphonique non créée car la durée de l'audio est nulle.")


    print("💬 Calcul des statistiques d'interaction vocale (turn-taking)...")
    turn_taking_summary_stats, df_speaker_turn_duration_stats, df_interruption_detail_stats = compute_turn_taking_stats(
        df_pyannote_single_speaker,
        df_pyannote_overlapped,
        total_audio_duration_seconds=total_duration_sec
    )
    if turn_taking_summary_stats:
        print("Résumé du Turn-Taking :", turn_taking_summary_stats)
    if not df_speaker_turn_duration_stats.empty:
        print("Durée moyenne des tours par locuteur :")
        print(df_speaker_turn_duration_stats)
    print("✅ Statistiques d'interaction vocale calculées.")

    print("📊 Génération des graphiques...")
    if not df_pyannote_single_speaker.empty and "speaker" in df_pyannote_single_speaker.columns:
        save_timeline_image(df_pyannote_single_speaker, "speaker", output_dir, plot_suffix="parole_unique")
    else:
        print("Avertissement: df_pyannote_single_speaker est vide ou ne contient pas la colonne 'speaker'. Graphique de timeline parole_unique non généré.")
        
    save_db_plot(df_features, output_dir)
    df_yamnet_stats_export, df_yamnet_filtered_export = save_yamnet_distribution(df_yamnet, output_dir)
    df_silence_stats_export = save_silence_histogram(df_features, output_dir)
    
    if not df_poly_timeline.empty:
        categories_yamnet_a_tracer = ["Music", "Speech", "Silence_YamNet", "Effects"] # Renommer Silence_YamNet si besoin
        
        actual_time_step_for_plot = 0.5 
        if len(df_poly_timeline) >= 2:
            actual_time_step_for_plot = df_poly_timeline['time_start'].iloc[1] - df_poly_timeline['time_start'].iloc[0]
        
        save_polyphonic_timeline_plot(
            df_poly_timeline, 
            output_dir, 
            yamnet_categories_to_plot=categories_yamnet_a_tracer,
            time_step=actual_time_step_for_plot,
            total_audio_duration_seconds=total_duration_sec # Passer la durée totale
        )
    else:
        print("Avertissement: df_poly_timeline est vide, le graphique de la timeline polyphonique détaillée ne sera pas généré.")

    print("✅ Graphiques sauvegardés.")


    print("📀 Export des résultats vers Excel...")
    export_to_excel(
        output_dir=output_dir, 
        df_features=df_features,
        df_transcript=df_transcript_enriched,
        df_yamnet=df_yamnet_filtered_export,
        df_silence_stats=df_silence_stats_export,
        df_yamnet_stats=df_yamnet_stats_export,
        df_speaker_stats=df_speaker_stats,
        general_speech_stats=general_speech_stats,
        df_diar_pyannote_single=df_pyannote_single_speaker,
        df_diar_pyannote_overlapped=df_pyannote_overlapped,
        df_poly_timeline=df_poly_timeline,
        turn_taking_summary_stats=turn_taking_summary_stats, 
        df_speaker_turn_duration_stats=df_speaker_turn_duration_stats 
    )
    print("--- Fin du Pipeline d'Analyse Audio ---")