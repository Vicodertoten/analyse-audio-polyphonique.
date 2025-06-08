import whisper
import pandas as pd
from tqdm import tqdm
from modules.analysis_utils import format_time_ms

def transcribe_audio(filename):
    model = whisper.load_model("medium", device="cpu")
    result = model.transcribe(filename, language="fr", fp16=False)

# Affiche la progression par segment
    segments = result["segments"]
    data = []  
    for seg in tqdm(segments, desc="🗣️ Transcription des segments", unit="segment"):
        data.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"]
        })
    
    df = pd.DataFrame(data)
    df['start_ms'] = df['start'].apply(format_time_ms) # Assurez-vous d'importer format_time_ms
    df['end_ms'] = df['end'].apply(format_time_ms)

    return df