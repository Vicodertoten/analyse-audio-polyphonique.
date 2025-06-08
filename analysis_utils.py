import pandas as pd
import numpy as np
from pyannote.core import Annotation

# --- Fonction Utilitaire de Formatage du Temps ---
def format_time_ms(seconds_float):
    """Convertit les secondes (float) en une chaîne MM:SS."""
    if seconds_float is None or pd.isna(seconds_float) or seconds_float < 0:
        return "00:00" # Ou une autre valeur par défaut comme "" ou None
    minutes = int(seconds_float // 60)
    seconds = int(seconds_float % 60)
    return f"{minutes:02d}:{seconds:02d}"

# --- Fonctions d'Analyse Mises à Jour ---

def assign_speakers_and_compute_stats(df_transcript, df_diar_pyannote):
    """
    Associe chaque segment de transcription à un locuteur pyannote et calcule les statistiques.
    Ajoute des colonnes formatées 'start_ms', 'end_ms', 'duration_ms'.
    Retourne :
    - df_transcript enrichi
    - general_stats : dict des stats globales (avec durées formatées)
    - speaker_stats : DataFrame des stats par locuteur (avec durées formatées)
    """
    df_transcript = df_transcript.copy()
    df_transcript["speaker"] = "unknown"

    # S'assurer que df_diar_pyannote n'est pas None et a les colonnes nécessaires
    if df_diar_pyannote is not None and not df_diar_pyannote.empty and \
       all(col in df_diar_pyannote.columns for col in ["start", "end", "speaker_pyannote"]):
        for i, row in df_transcript.iterrows():
            overlaps = df_diar_pyannote[
                (df_diar_pyannote["start"] < row["end"]) &
                (df_diar_pyannote["end"] > row["start"])
            ]
            if not overlaps.empty:
                overlaps = overlaps.copy()
                overlaps["overlap"] = overlaps.apply(
                    lambda r: min(r["end"], row["end"]) - max(r["start"], row["start"]), axis=1
                )
                best_match = overlaps.sort_values("overlap", ascending=False).iloc[0]
                df_transcript.at[i, "speaker"] = best_match["speaker_pyannote"]
    else:
        print("Avertissement (assign_speakers): df_diar_pyannote est vide ou mal formaté. Attribution des locuteurs ignorée.")


    df_transcript["nb_mots"] = df_transcript["text"].apply(lambda t: len(t.strip().split()))
    df_transcript["duration"] = df_transcript["end"] - df_transcript["start"]
    # Gérer la division par zéro si duration est nulle
    df_transcript["mots_par_seconde"] = df_transcript.apply(
        lambda row: row["nb_mots"] / row["duration"] if row["duration"] > 0 else 0, axis=1
    )


    # Ajout des colonnes formatées pour df_transcript
    df_transcript['start_ms'] = df_transcript['start'].apply(format_time_ms)
    df_transcript['end_ms'] = df_transcript['end'].apply(format_time_ms)
    df_transcript['duration_ms'] = df_transcript['duration'].apply(format_time_ms)

    total_audio_seconds = df_transcript["end"].max() if not df_transcript.empty else 0
    total_duration_min = total_audio_seconds / 60
    
    interventions_per_minute = (df_transcript.shape[0] / total_duration_min) if total_duration_min > 0 else 0
    mean_duration_sec = df_transcript["duration"].mean() if not df_transcript.empty else 0
    
    # Calculer overall_speed en s'assurant que la somme des durées n'est pas nulle
    sum_durations = df_transcript["duration"].sum()
    overall_speed = (df_transcript["nb_mots"].sum() / sum_durations) if sum_durations > 0 else 0


    general_stats = {
        "Débit global moyen (mots/s)": round(overall_speed, 2),
        "Durée moyenne intervention (s)": round(mean_duration_sec, 2),
        "Durée moyenne intervention (min:sec)": format_time_ms(mean_duration_sec),
        "Interventions par minute": round(interventions_per_minute, 2),
        "Durée totale analysée (s)": round(total_audio_seconds, 2),
        "Durée totale analysée (min:sec)": format_time_ms(total_audio_seconds)
    }

    if not df_transcript.empty and "speaker" in df_transcript.columns:
        speaker_stats = df_transcript.groupby("speaker").agg(
            nb_interventions=("text", "count"),
            durée_totale_s=("duration", "sum"),
            durée_moyenne_s=("duration", "mean"),
            débit_moyen_mots_par_s=("mots_par_seconde", "mean") # Renommé pour clarté
        ).reset_index()

        speaker_stats['durée_totale_ms'] = speaker_stats['durée_totale_s'].apply(format_time_ms)
        speaker_stats['durée_moyenne_ms'] = speaker_stats['durée_moyenne_s'].apply(format_time_ms)
        # Arrondir les valeurs numériques pour un meilleur affichage
        speaker_stats['durée_totale_s'] = speaker_stats['durée_totale_s'].round(2)
        speaker_stats['durée_moyenne_s'] = speaker_stats['durée_moyenne_s'].round(2)
        speaker_stats['débit_moyen_mots_par_s'] = speaker_stats['débit_moyen_mots_par_s'].round(2)
    else:
        speaker_stats = pd.DataFrame(columns=[
            "speaker", "nb_interventions", "durée_totale_s", "durée_moyenne_s", 
            "débit_moyen_mots_par_s", "durée_totale_ms", "durée_moyenne_ms"
        ])


    return df_transcript, general_stats, speaker_stats

def create_polyphonic_timeline(
    total_audio_duration_seconds: float,
    df_single_speaker: pd.DataFrame,
    df_overlapped_speech: pd.DataFrame,
    df_yamnet_multilabel: pd.DataFrame,
    df_audio_features: pd.DataFrame,
    time_step: float = 0.5
):
    """
    Crée une timeline polyphonique unifiée.
    Ajoute des colonnes formatées 'time_start_ms', 'time_end_ms'.
    """
    timeline_index_starts = np.arange(0, total_audio_duration_seconds, time_step)
    poly_timeline_data = []

    default_active_speakers = []
    default_yamnet_labels = []

    for t_start_current_step in timeline_index_starts:
        t_end_current_step = t_start_current_step + time_step
        current_row_data = {
            "time_start": t_start_current_step,
            "time_end": t_end_current_step,
            "active_speakers": list(default_active_speakers),
            "yamnet_labels": list(default_yamnet_labels),
            "is_silence": None,
            "rms_level": np.nan
        }

        current_active_speakers = set()
        if df_overlapped_speech is not None and not df_overlapped_speech.empty and \
           all(col in df_overlapped_speech.columns for col in ["start", "end", "speakers_involved"]):
            active_overlaps = df_overlapped_speech[
                (df_overlapped_speech["start"] < t_end_current_step) & 
                (df_overlapped_speech["end"] > t_start_current_step)
            ]
            for _, row in active_overlaps.iterrows():
                current_active_speakers.update(row["speakers_involved"])
        
        if not current_active_speakers and df_single_speaker is not None and not df_single_speaker.empty and \
           all(col in df_single_speaker.columns for col in ["start", "end", "speaker"]):
            active_singles = df_single_speaker[
                (df_single_speaker["start"] < t_end_current_step) & 
                (df_single_speaker["end"] > t_start_current_step)
            ]
            for _, row in active_singles.iterrows():
                current_active_speakers.add(row["speaker"])
        
        current_row_data["active_speakers"] = sorted(list(current_active_speakers)) if current_active_speakers else default_active_speakers

        current_yamnet_labels = set()
        if df_yamnet_multilabel is not None and not df_yamnet_multilabel.empty and \
           all(col in df_yamnet_multilabel.columns for col in ["start", "end", "labels"]):
            active_yamnet_segments = df_yamnet_multilabel[
                (df_yamnet_multilabel["start"] < t_end_current_step) & 
                (df_yamnet_multilabel["end"] > t_start_current_step)
            ]
            for _, row in active_yamnet_segments.iterrows():
                current_yamnet_labels.update(row["labels"])
        current_row_data["yamnet_labels"] = sorted(list(current_yamnet_labels)) if current_yamnet_labels else default_yamnet_labels
        
        if df_audio_features is not None and not df_audio_features.empty and \
           all(col in df_audio_features.columns for col in ["start", "end", "is_silence", "rms"]):
            relevant_feature_row = df_audio_features[
                (df_audio_features["start"] <= t_start_current_step) & 
                (df_audio_features["end"] > t_start_current_step)
            ]
            if not relevant_feature_row.empty:
                current_row_data["is_silence"] = relevant_feature_row.iloc[0]["is_silence"]
                current_row_data["rms_level"] = relevant_feature_row.iloc[0]["rms"]

        poly_timeline_data.append(current_row_data)
    
    df_poly_timeline = pd.DataFrame(poly_timeline_data)
    
    if "active_speakers" not in df_poly_timeline.columns:
        df_poly_timeline["active_speakers"] = [list(default_active_speakers) for _ in range(len(df_poly_timeline))]
    if "yamnet_labels" not in df_poly_timeline.columns:
        df_poly_timeline["yamnet_labels"] = [list(default_yamnet_labels) for _ in range(len(df_poly_timeline))]

    # Ajout des colonnes formatées
    if not df_poly_timeline.empty:
        df_poly_timeline['time_start_ms'] = df_poly_timeline['time_start'].apply(format_time_ms)
        df_poly_timeline['time_end_ms'] = df_poly_timeline['time_end'].apply(format_time_ms)
        # Arrondir les valeurs numériques pour un meilleur affichage
        df_poly_timeline['time_start'] = df_poly_timeline['time_start'].round(3)
        df_poly_timeline['time_end'] = df_poly_timeline['time_end'].round(3)
        if 'rms_level' in df_poly_timeline.columns:
            df_poly_timeline['rms_level'] = df_poly_timeline['rms_level'].round(4)

    return df_poly_timeline

def compute_turn_taking_stats(
    df_single_speaker_segments: pd.DataFrame, 
    df_overlapped_segments: pd.DataFrame = None,
    total_audio_duration_seconds: float = None
    ):
    """
    Calcule des statistiques sur l'alternance des tours de parole.
    Ajoute des colonnes formatées pour les durées dans les DataFrames retournés
    et dans le dictionnaire de résumé.
    """
    if df_single_speaker_segments is None or df_single_speaker_segments.empty or \
       not all(col in df_single_speaker_segments.columns for col in ["start", "speaker", "duration"]):
        print("Avertissement (compute_turn_taking): df_single_speaker_segments est vide ou mal formaté. Stats de turn-taking non calculées.")
        return {}, pd.DataFrame(columns=["speaker", "avg_turn_duration_s", "avg_turn_duration_ms", 
                                         "total_speech_time_s", "total_speech_time_ms", "num_turns"]), pd.DataFrame()

    df_sorted_turns = df_single_speaker_segments.sort_values(by="start").reset_index(drop=True)
    
    speaker_specific_turn_durations = {}
    turn_changes = 0
    
    if len(df_sorted_turns) > 0:
        for i in range(len(df_sorted_turns)):
            segment = df_sorted_turns.iloc[i]
            speaker = segment["speaker"]
            
            if speaker not in speaker_specific_turn_durations:
                speaker_specific_turn_durations[speaker] = []

            turn_duration = segment["duration"]
            speaker_specific_turn_durations[speaker].append(turn_duration)

            if i > 0 and df_sorted_turns.iloc[i-1]["speaker"] != speaker:
                turn_changes += 1
    
    avg_turn_durations_list = []
    for speaker, durations in speaker_specific_turn_durations.items():
        if durations:
            avg_s = np.mean(durations)
            sum_s = np.sum(durations)
            avg_turn_durations_list.append({
                "speaker": speaker,
                "avg_turn_duration_s": round(avg_s, 2),
                "avg_turn_duration_ms": format_time_ms(avg_s),
                "total_speech_time_s": round(sum_s, 2),
                "total_speech_time_ms": format_time_ms(sum_s),
                "num_turns": len(durations)
            })
    df_avg_turn_duration_per_speaker = pd.DataFrame(avg_turn_durations_list)

    turn_taking_summary = {
        "total_unique_speaker_segments": len(df_sorted_turns),
        "total_turn_changes": turn_changes
    }
    if total_audio_duration_seconds and total_audio_duration_seconds > 0:
        turn_taking_summary["turn_changes_per_minute"] = round((turn_changes / total_audio_duration_seconds * 60), 2)
    else:
        turn_taking_summary["turn_changes_per_minute"] = None

    df_interruption_stats = pd.DataFrame() 
    if df_overlapped_segments is not None and not df_overlapped_segments.empty and \
       "duration" in df_overlapped_segments.columns:
        total_overlap_duration_s = df_overlapped_segments["duration"].sum()
        turn_taking_summary["num_overlapped_speech_segments"] = len(df_overlapped_segments)
        turn_taking_summary["total_duration_overlapped_speech_s"] = round(total_overlap_duration_s, 2)
        turn_taking_summary["total_duration_overlapped_speech_ms"] = format_time_ms(total_overlap_duration_s)
    else:
        turn_taking_summary["num_overlapped_speech_segments"] = 0
        turn_taking_summary["total_duration_overlapped_speech_s"] = 0.0
        turn_taking_summary["total_duration_overlapped_speech_ms"] = "00:00"

    return turn_taking_summary, df_avg_turn_duration_per_speaker, df_interruption_stats

def process_pyannote_annotation(annotation: Annotation, min_speakers_for_overlap: int = 2):
    """
    Traite l'objet Annotation de Pyannote.
    Ajoute des colonnes formatées 'start_ms', 'end_ms', 'duration_ms'.
    """
    single_speaker_data = []
    overlapped_speech_data = []

    if annotation is None: # Vérification ajoutée
        print("Avertissement (process_pyannote): Annotation Pyannote est None. Retour de DataFrames vides.")
        return pd.DataFrame(columns=["start", "end", "speaker", "duration", "start_ms", "end_ms", "duration_ms"]), \
               pd.DataFrame(columns=["start", "end", "speakers_involved", "num_speakers", "duration", "start_ms", "end_ms", "duration_ms"])

    timeline_parole_totale = annotation.get_timeline().support()

    for seg_global in timeline_parole_totale:
        locuteurs_actifs_dans_segment = []
        for _segment_interne, _track_id, label in annotation.crop(seg_global).itertracks(yield_label=True):
            if label not in locuteurs_actifs_dans_segment:
                locuteurs_actifs_dans_segment.append(label)
        
        num_locuteurs_actifs = len(locuteurs_actifs_dans_segment)
        duration_s = seg_global.duration

        if num_locuteurs_actifs == 1:
            single_speaker_data.append({
                "start": seg_global.start,
                "end": seg_global.end,
                "speaker": locuteurs_actifs_dans_segment[0],
                "duration": duration_s
            })
        elif num_locuteurs_actifs >= min_speakers_for_overlap:
            overlapped_speech_data.append({
                "start": seg_global.start,
                "end": seg_global.end,
                "speakers_involved": sorted(locuteurs_actifs_dans_segment),
                "num_speakers": num_locuteurs_actifs,
                "duration": duration_s
            })

    df_single_speaker_segments = pd.DataFrame(single_speaker_data)
    df_overlapped_speech_segments = pd.DataFrame(overlapped_speech_data)

    if not df_single_speaker_segments.empty:
        df_single_speaker_segments['start_ms'] = df_single_speaker_segments['start'].apply(format_time_ms)
        df_single_speaker_segments['end_ms'] = df_single_speaker_segments['end'].apply(format_time_ms)
        df_single_speaker_segments['duration_ms'] = df_single_speaker_segments['duration'].apply(format_time_ms)
        # Arrondir les valeurs numériques
        df_single_speaker_segments['start'] = df_single_speaker_segments['start'].round(3)
        df_single_speaker_segments['end'] = df_single_speaker_segments['end'].round(3)
        df_single_speaker_segments['duration'] = df_single_speaker_segments['duration'].round(3)


    if not df_overlapped_speech_segments.empty:
        df_overlapped_speech_segments['start_ms'] = df_overlapped_speech_segments['start'].apply(format_time_ms)
        df_overlapped_speech_segments['end_ms'] = df_overlapped_speech_segments['end'].apply(format_time_ms)
        df_overlapped_speech_segments['duration_ms'] = df_overlapped_speech_segments['duration'].apply(format_time_ms)
        # Arrondir les valeurs numériques
        df_overlapped_speech_segments['start'] = df_overlapped_speech_segments['start'].round(3)
        df_overlapped_speech_segments['end'] = df_overlapped_speech_segments['end'].round(3)
        df_overlapped_speech_segments['duration'] = df_overlapped_speech_segments['duration'].round(3)

    return df_single_speaker_segments, df_overlapped_speech_segments