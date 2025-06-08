# modules/exporter.py
import pandas as pd
# import xlsxwriter # Pas besoin d'importer directement
import os

def export_to_excel(
    output_dir: str, # Mis en premier pour clarté
    df_features: pd.DataFrame = None,
    df_transcript: pd.DataFrame = None,
    df_yamnet: pd.DataFrame = None, # Segments YAMNet filtrés pour le graphique principal
    df_silence_stats: pd.DataFrame = None, # Durées des silences
    df_yamnet_stats: pd.DataFrame = None, # Top labels YAMNet et leurs durées
    df_speaker_stats: pd.DataFrame = None, # Stats de parole par locuteur (débit, etc. de assign_speakers...)
    general_speech_stats: dict = None, # Débit global, etc.
    df_diar_pyannote_single: pd.DataFrame = None,
    df_diar_pyannote_overlapped: pd.DataFrame = None,
    df_poly_timeline: pd.DataFrame = None, # LA timeline unifiée
    turn_taking_summary_stats: dict = None, # Nouvelles stats globales de turn-taking
    df_speaker_turn_duration_stats: pd.DataFrame = None, # Nouvelles stats de durée de tour par locuteur
    # df_interruption_detail_stats: pd.DataFrame = None, # Si vous l'implémentez
    **kwargs # Pour flexibilité future
):
    """
    Exporte toutes les analyses et statistiques vers un fichier Excel.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_folder_name = os.path.basename(output_dir) # Pour un nom de fichier plus descriptif
    excel_path = os.path.join(output_dir, f"analyse_audio_polyphonique_{base_folder_name}.xlsx")
    
    print(f"✍️ Début de l'export vers Excel : {excel_path}")
    
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        workbook = writer.book

        def write_and_autofit(df_to_write, sheet_name_str):
            if df_to_write is not None and not df_to_write.empty:
                try:
                    df_to_write.to_excel(writer, sheet_name=sheet_name_str, index=False)
                    worksheet = writer.sheets[sheet_name_str]
                    for i, column in enumerate(df_to_write.columns):
                        # S'assurer que les données de la colonne peuvent être converties en str
                        try:
                            col_data = df_to_write[column].astype(str).values
                            # Longueur max des données + longueur du header
                            max_len = max([len(s) for s in col_data] + [len(str(column))])
                            # Limiter la largeur maximale des colonnes
                            worksheet.set_column(i, i, min(max_len + 2, 70)) 
                        except Exception as e_col_autofit:
                            print(f"  ⚠️ Erreur autofit colonne '{column}' feuille '{sheet_name_str}': {e_col_autofit}")
                    print(f"  📄 Feuille '{sheet_name_str}' écrite avec {len(df_to_write)} lignes.")
                except Exception as e_sheet_write:
                    print(f"  ❌ Erreur écriture feuille '{sheet_name_str}': {e_sheet_write}")
            else:
                print(f"  ℹ️ DataFrame pour feuille '{sheet_name_str}' est vide ou None. Feuille non écrite.")

        # --- Onglets de Données Brutes et Traitées ---
        print("  Writing primary data sheets...")
        write_and_autofit(df_transcript, "Transcription_Enrichie")
        write_and_autofit(df_diar_pyannote_single, "Diar_Parole_Unique")
        write_and_autofit(df_diar_pyannote_overlapped, "Diar_Parole_Superposée")
        write_and_autofit(df_yamnet, "YAMNet_Segments_Filtres") # Les segments utilisés pour le graph
        write_and_autofit(df_features, "Features_Audio_1s")
        if df_poly_timeline is not None and not df_poly_timeline.empty : # Très important
             write_and_autofit(df_poly_timeline, "Timeline_Polyphonique_Unifiée")

        # --- Onglets de Statistiques ---
        print("  Writing statistics sheets...")
        # Statistiques de parole (Whisper + Diarisation)
        if general_speech_stats is not None and isinstance(general_speech_stats, dict):
            write_and_autofit(pd.DataFrame([general_speech_stats]), "Stats_Parole_Globale")
        write_and_autofit(df_speaker_stats, "Stats_Parole_Par_Locuteur") # de assign_speakers...
        
        # Statistiques d'interaction vocale (Turn-Taking)
        if turn_taking_summary_stats is not None and isinstance(turn_taking_summary_stats, dict):
            write_and_autofit(pd.DataFrame([turn_taking_summary_stats]), "Stats_TurnTaking_Global")
        write_and_autofit(df_speaker_turn_duration_stats, "Stats_TurnTaking_Par_Locuteur")
        # write_and_autofit(df_interruption_detail_stats, "Stats_Interruptions_Detail") # Si implémenté

        # Statistiques YAMNet
        write_and_autofit(df_yamnet_stats, "Stats_YAMNet_Top_Labels") # du graphe YAMNet

        # Statistiques des Silences (RMS)
        write_and_autofit(df_silence_stats, "Stats_Segments_Silence_RMS")


        # TODO: Ajouter un onglet pour les Statistiques Croisées issues de df_poly_timeline
        # quand ces stats seront calculées.
        # write_and_autofit(df_crossed_polyphonic_stats, "Stats_Polyphoniques_Croisees")


        # --- Onglet de Statistiques Générales Sommaire (à revoir/affiner) ---
        print("  Writing summary statistics sheet...")
        summary_data = {}
        if df_transcript is not None and not df_transcript.empty and 'end' in df_transcript.columns:
            # S'assurer que 'end' n'est pas vide avant d'appeler max()
            valid_ends = df_transcript['end'].dropna()
            if not valid_ends.empty:
                 summary_data['Durée totale (min)'] = [round(valid_ends.max() / 60, 2)]
        
        if df_features is not None and not df_features.empty and 'is_silence' in df_features.columns:
            summary_data['Taux de silence (RMS %)'] = [round(100 * df_features["is_silence"].sum() / len(df_features),1) if len(df_features) > 0 else 0]
        
        if df_yamnet is not None and not df_yamnet.empty: # df_yamnet est le filtré ici
            summary_data['Nombre de segments YAMNet (graph)'] = [df_yamnet.shape[0]]
        
        if df_diar_pyannote_single is not None and not df_diar_pyannote_single.empty and "speaker" in df_diar_pyannote_single.columns:
            summary_data['Nombre de locuteurs (uniques diar)'] = [df_diar_pyannote_single["speaker"].nunique()]
        
        if summary_data:
            write_and_autofit(pd.DataFrame(summary_data), "Statistiques_Sommaire")

        # --- Grille d'analyse manuelle (inchangée pour l'instant) ---
        print("  Writing manual analysis grid sheet...")
        manual_analysis_sheet = workbook.add_worksheet("GrilleAnalyseManuelle")
        writer.sheets["GrilleAnalyseManuelle"] = manual_analysis_sheet # Enregistrer la référence
        headers = ['Timestamp (s)', 'Axe Polyphonique', 'Observation (Qualitatif)', 
                   'Extrait Sonore (si pertinent)', 'Commentaire / Interprétation', 
                   'Donnée Automatisée de Réf. (Feuille/Ligne Excel)']
        manual_analysis_sheet.write_row('A1', headers)
        # Autofit pour la grille manuelle
        for i, header_text in enumerate(headers):
            manual_analysis_sheet.set_column(i, i, len(header_text) + 5 if len(header_text) < 30 else 40)
        
        # Axes pré-remplis (inspirés de votre document)
        axes_polyphoniques = [
            'Voix Humaines (Interaction, Jeux de voix)', 'Musique (Rôle, Dialogue avec parole)',
            'Bruitages/Effets (Fonction, Interaction)', 'Silence (Expressivité, Structuration)',
            'Montage (Transitions, Rythme global, Superpositions)', 
            'Orchestration Générale (Harmonie, Contraste, Tension)',
            'Structure Narrative Sonore'
        ]
        for r, axe in enumerate(axes_polyphoniques, start=1): # Commence à la ligne 2 (index 1)
            manual_analysis_sheet.write(r, 1, axe) # Colonne B (index 1)
        print("  Grille d'analyse manuelle prête.")

        # --- Insertion des Images (Graphiques) ---
        print("  Inserting graph images...")
        # Assurez-vous que les noms de fichiers correspondent à ceux sauvegardés par les fonctions de plot
        image_files_to_insert = [
            ("Timeline Parole Unique", "timeline_parole_unique.png"), # Modifié pour correspondre au suffixe
            # ("Timeline Polyphonique", "timeline_polyphonique_detaillee.png"), # À ajouter quand le plot est prêt
            ("Volume Audio (RMS)", "volume_rms_db.png"),
            ("Distribution YAMNet", "yamnet_distribution.png"),
            ("Distribution Silences (RMS)", "silence_histogram.png")
        ]
        # Ajouter la nouvelle timeline polyphonique si elle est générée
        poly_timeline_plot_filename = "timeline_polyphonique_detaillee.png"
        if os.path.exists(os.path.join(output_dir, poly_timeline_plot_filename)):
            image_files_to_insert.append(("Timeline Polyphonique", poly_timeline_plot_filename))
        
        for sheet_title, image_filename_only in image_files_to_insert:
            full_image_path = os.path.join(output_dir, image_filename_only)
            if os.path.exists(full_image_path):
                try:
                    image_sheet = workbook.add_worksheet(sheet_title[:31]) # Limiter la longueur du nom de feuille
                    image_sheet.insert_image("B2", full_image_path, {'x_scale': 0.7, 'y_scale': 0.7}) # Ajuster l'échelle si besoin
                    print(f"    🖼️ Image '{image_filename_only}' insérée dans la feuille '{sheet_title}'.")
                except Exception as e_img:
                    print(f"    ⚠️ Erreur insertion image {image_filename_only}: {e_img}")
            else:
                print(f"    ℹ️ Image {image_filename_only} non trouvée, non insérée.")

    print(f"✅ Analyse complète exportée avec succès dans : {excel_path}")