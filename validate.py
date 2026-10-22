import os
import pandas as pd

# 1. Configuration Constants (Updated to match your multi-CSV setup)
CSV_PATHS = ["abbrev.csv", "comtrdrug.csv", "drugsentences.csv", "hospmed.csv","medequip.csv"]

def run_validation():
    print("🔍 Starting stable local dataset validation check for multiple files...")
    
    # Global tracking counters across all files
    total_records_processed = 0
    total_missing_files = 0
    total_empty_transcripts = 0
    total_zero_byte_files = 0
    files_with_errors = 0

    for csv_path in CSV_PATHS:
        print(f"\n📖 Processing file: {csv_path}")
        
        # Verify CSV Presence
        if not os.path.exists(csv_path):
            print(f"❌ Error: The file '{csv_path}' was not found. Skipping file.")
            files_with_errors += 1
            continue
            
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ Error: Failed to read CSV file. Details: {e}. Skipping file.")
            files_with_errors += 1
            continue
            
        # Standardize column headers to match what our script checks
        if "audio_path" in df.columns:
            df = df.rename(columns={"audio_path": "audio"})
            
        required_cols = ["audio", "transcript"]
        if not all(col in df.columns for col in required_cols):
            print(f"❌ Error: CSV must contain exactly {required_cols} headers. Skipping file.")
            files_with_errors += 1
            continue
            
        file_rows = len(df)
        if file_rows == 0:
            print("⚠️ Warning: This specific dataset CSV file is completely empty.")
            continue
            
        print(f"📋 Structure verified. Found {file_rows} entries. Checking disk assets...")
        total_records_processed += file_rows
        
        # Iterate through and physically check every row and file asset in current CSV
        for idx, row in df.iterrows():
            raw_path = str(row["audio"])
            transcript = str(row["transcript"]).strip()
            
            # Check for blank medical text strings
            if not transcript or pd.isna(row["transcript"]) or transcript.lower() == "nan":
                print(f"⚠️ {csv_path} [Row {idx}]: Transcript text is missing or blank.")
                total_empty_transcripts += 1
                
            # Check if the file physically exists on the disk
            if not os.path.exists(raw_path):
                print(f"❌ {csv_path} [Row {idx}]: Audio asset missing at path -> {raw_path}")
                total_missing_files += 1
                continue
                
            # Check for empty or unwritten audio files (0 bytes)
            if os.path.getsize(raw_path) == 0:
                print(f"❌ {csv_path} [Row {idx}]: Audio file is empty (0 bytes) -> {raw_path}")
                total_zero_byte_files += 1

    # 3. Final Unified Evaluation Summary Report
    print("\n📊 --- Final Aggregate Validation Summary ---")
    print(f" • Total CSV files tracked: {len(CSV_PATHS)}")
    print(f" • Total valid data records found: {total_records_processed}")
    print(f" • Unreadable/missing CSV files: {files_with_errors}")
    print(f" • Missing audio assets: {total_missing_files}")
    print(f" • Blank medical transcripts: {total_empty_transcripts}")
    print(f" • Empty/0-byte audio files: {total_zero_byte_files}")
    
    if (total_missing_files == 0 and 
        total_empty_transcripts == 0 and 
        total_zero_byte_files == 0 and 
        files_with_errors == 0 and 
        total_records_processed > 0):
        print(f"\n🚀 SUCCESS: All {total_records_processed} records across all files are perfectly mapped! Paths are valid, text is complete, and audio assets are ready for chained training.")
    else:
        print("\n⚠️ WARNING: Please resolve the validation issues above before attempting to run your training pipeline script.")

if __name__ == "__main__":
    run_validation()
