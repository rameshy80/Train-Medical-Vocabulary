import os
import re
import pandas as pd
from docx import Document
from gtts import gTTS  # Native cloud-based engine workaround

# 1. Configuration Constants
WORD_DOC_PATH = "medsentences.docx"
AUDIO_OUTPUT_DIR = "medsentence_audio"
CSV_OUTPUT_PATH = "medsentence.csv"

# Ensure output directory exists
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

def extract_quoted_sentences(docx_path):
    """Reads a .docx file and pulls all text enclosed within quotation marks."""
    try:
        doc = Document(docx_path)
    except Exception as e:
        print(f"❌ Error opening {docx_path}: {e}")
        return []
    
    extracted = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
            
        # Standardize curly quotes
        normalized_text = (
            text.replace("“", '"')
                .replace("”", '"')
                .replace("„", '"')
                .replace("‟", '"')
        )
        
        matches = re.findall(r'"([^"]*)"', normalized_text)
        for match in matches:
            clean_text = match.strip()
            if clean_text:
                extracted.append(clean_text)
                
    return extracted

# 3. Main Data Processing Execution Loop
if __name__ == "__main__":
    print("📂 Reading medical scripts from Word document...")
    sentences = extract_quoted_sentences(WORD_DOC_PATH)
    
    if not sentences:
        print("⚠️ No quoted sentences found. Check your quotation formatting.")
        exit()
        
    print(f"📋 Found {len(sentences)} quoted lines. Commencing local voice synthesis...")
    dataset_rows = []
    
    for i, sentence in enumerate(sentences):
        audio_filename = f"{AUDIO_OUTPUT_DIR}/sample_{i}.mp3"  # gTTS saves natively as mp3
        audio_path = os.path.abspath(audio_filename)
        
        try:
            # Generate speech audio using Google's voice engine backend
            tts = gTTS(text=sentence, lang='en', tld= 'co.in', slow=False)
            tts.save(audio_path)
            
            # Map values cleanly into your tracking array
            dataset_rows.append({"audio_path": audio_path, "transcript": sentence})
            print(f" 🎙️ Synthesized sample {i+1}/{len(sentences)}")
            
        except Exception as loop_err:
            print(f"❌ Error synthesizing row {i}: {loop_err}")
            continue

    # 4. Export standard HuggingFace training CSV
    df = pd.DataFrame(dataset_rows)
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"\n✅ Pipeline complete! Created '{CSV_OUTPUT_PATH}' with all {len(df)} audio records.")
