import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate

# Keep global configuration, definitions, and helper classes out in the open
MODEL_ID = "whisper-medical-final" 

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# 1. WRAP EXECUTABLE PIPELINES INSIDE THE PROTECTED GUARD BLOCK
if __name__ == '__main__':
    
    # Precision configuration based on GPU capabilities
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    # Load processor and 8-bit quantized model checkpoint
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="English", task="transcribe")

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        quantization_config=bnb_config, 
        device_map="auto" 
    )

    # Initialize LoRA structures
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=32, 
        lora_alpha=64, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, 
        bias="none"
    )
    model = get_peft_model(model, lora_config)

    # 4. DATASET PIPELINE (Bypassing torchcodec requirements)
    dataset = load_dataset("Hani89/Synthetic-Medical-Speech-Dataset", split="train")

    # Tell Hugging Face to fetch raw audio paths/bytes instead of auto-decoding
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))

    def prepare_dataset(batch):
        # Manually unpack the raw audio data array using soundfile/librosa mechanics
        audio_data = batch["audio"]
        
        # Read the raw path or bytes into an array structure if it isn't decoded yet
        import librosa
        speech_array, sampling_rate = librosa.load(audio_data["path"], sr=16000)
        
        # Extract log-Mel spectrogram inputs safely
        batch["input_features"] = processor.feature_extractor(speech_array, sampling_rate=16000).input_features
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    # Apply mapping using your protected multiprocessing setup
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=4)
    dataset = dataset.train_test_split(test_size=0.1)


    # Setup metrics evaluation configuration
    metric = evaluate.load("jiwer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Configuration for Trainer settings
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-large-medical-further-tuned",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  
        learning_rate=5e-5,             
        warmup_steps=100,
        max_steps=2000,
        gradient_checkpointing=True,
        bf16=use_bf16,                  
        fp16=use_fp16,
        eval_strategy="steps",          # Standard non-deprecated evaluation mapping
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=25,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,    
    )

    # Initialize Modern Seq2SeqTrainer 
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,     # Clean interface handler mapping
    )

    # Launch protected training loops
    trainer.train()

    # Save outputs safely
    model.save_pretrained("./whisper-large-medical-final")
    processor.save_pretrained("./whisper-large-medical-final")
    print("Training finished successfully without bootstrapping errors!")
