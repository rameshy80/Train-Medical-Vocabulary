import os
import torch
import pandas as pd
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from torch.utils.data import IterableDataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# 1. Base Project Configurations (Modified to accept an array/list of CSV paths)
MODEL_ID = "whisper-medical-final"
CSV_PATHS = ["medsentence.csv"] 
OUTPUT_DIR = "./whisper-medical-local"
FINAL_MODEL_DIR = "./whisper-medical-final"

print("🎨 Initializing Sharded Low-RAM Initialization Pipeline...")

# 2. Instantiate Whisper Processor
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="English", task="transcribe")

# 3. Stream Generator Layer (Modified to handle multiple CSV files sequentially)
class MedicalStreamingDataset(IterableDataset):
    def __init__(self, csv_paths: List[str], processor):
        self.csv_paths = csv_paths
        self.processor = processor

    def __iter__(self):
        for csv_path in self.csv_paths:
            print(f"📖 Streaming dataset file: {csv_path}")
            try:
                df = pd.read_csv(csv_path)
                if "audio_path" in df.columns:
                    df = df.rename(columns={"audio_path": "audio"})
                
                for _, row in df.iterrows():
                    try:
                        # Load a single file into memory right when the GPU calls for it
                        audio_array, _ = librosa.load(str(row["audio"]), sr=16000, mono=True)
                        
                        # Compute spectrogram matrices
                        input_features = self.processor.feature_extractor(
                            audio_array, sampling_rate=16000
                        ).input_features[0]
                        
                        # Tokenize text labels
                        labels = self.processor.tokenizer(str(row["transcript"])).input_ids
                        yield {"input_features": input_features, "labels": labels}
                    except Exception as e:
                        print(f"⚠️ Skipping row record in {csv_path}: {e}")
                        continue
            except Exception as file_err:
                print(f"❌ Error loading CSV file {csv_path}: {file_err}")
                continue

# Instantiate the streaming dataset with the array of paths
streaming_dataset = MedicalStreamingDataset(CSV_PATHS, processor)

# 4. Data Collator Class for Dynamic Sequence Padding
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 5. Low-Memory Model Structure Initialization
print("📥 Instantiating Whisper Large weights directly to GPU VRAM...")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    low_cpu_mem_usage=True, 
    torch_dtype=torch.bfloat16
)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.enable_input_require_grads()
if device == "cuda":
    model = model.to(device)

# 6. Hardware-Optimized Hyperparameter Settings for RTX 5080 + 32GB RAM
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=10,
    max_steps=100,
    gradient_checkpointing=True,
    bf16=True,
    bf16_full_eval=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    eval_strategy="no",
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=10,
    save_steps=100,
    report_to="none"
)

# 7. Define Trainer Instance Loop
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=streaming_dataset,
    data_collator=data_collator,
    processing_class=processor.feature_extractor,
)

# 8. Start Training
print("🚀 Starting local Whisper finetuning for medical vocabulary...")
trainer.train()

print(f"💾 Saving finalized medical weights checkpoint to '{FINAL_MODEL_DIR}'...")
trainer.save_model(FINAL_MODEL_DIR)
processor.save_pretrained(FINAL_MODEL_DIR)
print("✅ Whisper training workflow finalized successfully!")
