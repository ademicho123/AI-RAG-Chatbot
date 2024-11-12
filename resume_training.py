import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import json
import os
from tqdm import tqdm
from datetime import datetime

class MetricsLogger(TrainerCallback):
    """Custom callback to log BLEU and ROUGE scores during training"""
    def __init__(self, tokenizer, eval_dataset, log_dir="./logs"):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.log_dir = log_dir
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.metrics_history = {
            'steps': [],
            'loss': [],
            'bleu': [],
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f'training_metrics_{timestamp}.log')
        with open(self.log_file, 'w') as f:
            f.write("step,loss,bleu,rouge1,rouge2,rougeL\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_world_process_zero:
            step = state.global_step
            
            # Calculate metrics
            metrics = self.calculate_metrics(kwargs.get('model'))
            
            # Update metrics history
            self.metrics_history['steps'].append(step)
            self.metrics_history['loss'].append(logs.get('loss', 0))
            self.metrics_history['bleu'].append(metrics['bleu'])
            self.metrics_history['rouge1'].append(metrics['rouge1'])
            self.metrics_history['rouge2'].append(metrics['rouge2'])
            self.metrics_history['rougeL'].append(metrics['rougeL'])
            
            # Log to file
            with open(self.log_file, 'a') as f:
                f.write(f"{step},{logs.get('loss', 0):.4f},{metrics['bleu']:.4f},"
                       f"{metrics['rouge1']:.4f},{metrics['rouge2']:.4f},{metrics['rougeL']:.4f}\n")
            
            # Print current metrics
            print(f"\nStep {step}:")
            print(f"Loss: {logs.get('loss', 0):.4f}")
            print(f"BLEU: {metrics['bleu']:.4f}")
            print(f"ROUGE-1: {metrics['rouge1']:.4f}")
            print(f"ROUGE-2: {metrics['rouge2']:.4f}")
            print(f"ROUGE-L: {metrics['rougeL']:.4f}")

    def calculate_metrics(self, model, num_samples=5):
        """Calculate BLEU and ROUGE scores on a subset of the evaluation dataset"""
        eval_subset = self.eval_dataset.select(range(min(num_samples, len(self.eval_dataset))))
        
        bleu_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for item in eval_subset:
            # Generate text
            input_ids = torch.tensor([item['input_ids']]).to(model.device)
            generated = model.generate(
                input_ids,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7
            )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            reference_text = self.tokenizer.decode(item['labels'], skip_special_tokens=True)
            
            # Calculate BLEU
            bleu = sentence_bleu([reference_text.split()], generated_text.split())
            bleu_scores.append(bleu)
            
            # Calculate ROUGE
            rouge_scores = self.scorer.score(reference_text, generated_text)
            rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
            rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
            rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
        
        return {
            'bleu': np.mean(bleu_scores),
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }

    def save_final_metrics(self):
        """Save the complete metrics history to a JSON file"""
        metrics_file = os.path.join(self.log_dir, 'final_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

class ResumeTrainer:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", output_dir="./resume_generator_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        
        print(f"Loading model and tokenizer from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def prepare_data(self, data_path):
        """Prepare and validate the dataset"""
        try:
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(data_path, encoding=encoding)
                    print(f"Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not read CSV file with any attempted encoding")
            
            print("\nDataset Statistics:")
            print(f"Total samples: {len(df)}")
            print(f"Unique job titles: {df['Job Title'].nunique()}")
            
            # Convert to lists for easier processing
            job_titles = df['Job Title'].tolist()
            resumes = df['Resume'].tolist()
            
            # Create input-output pairs
            formatted_data = []
            for title, resume in zip(job_titles, resumes):
                formatted_data.append({
                    'input_text': f"Generate a resume for {title}:",
                    'output_text': str(resume)  # Ensure resume is string
                })
            
            # Split the data
            train_data, val_data = train_test_split(
                formatted_data,
                test_size=0.1,
                random_state=42
            )
            
            return train_data, val_data
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def tokenize_data(self, examples):
        """Tokenize the data properly"""
        model_inputs = {
            'input_ids': [],
            'labels': []
        }
        
        for example in examples:
            # Tokenize input text
            input_ids = self.tokenizer(
                example['input_text'],
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            )['input_ids'].squeeze()
            
            # Tokenize output text
            labels = self.tokenizer(
                example['output_text'],
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            )['input_ids'].squeeze()
            
            model_inputs['input_ids'].append(input_ids)
            model_inputs['labels'].append(labels)
        
        return model_inputs

    def train(self, train_data, val_data):
        """Train the model with metric logging"""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            # Disable wandb
            report_to="none",
            run_name=None
        )
        
        # Tokenize the datasets
        train_tokenized = self.tokenize_data(train_data)
        val_tokenized = self.tokenize_data(val_data)
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_tokenized['input_ids'],
            'labels': train_tokenized['labels']
        })
        
        val_dataset = Dataset.from_dict({
            'input_ids': val_tokenized['input_ids'],
            'labels': val_tokenized['labels']
        })
        
        # Initialize metrics logger
        metrics_logger = MetricsLogger(self.tokenizer, val_dataset)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
            callbacks=[metrics_logger]
        )
        
        print("Starting training...")
        train_result = trainer.train()
        
        # Save final metrics
        metrics_logger.save_final_metrics()
        
        # Save model and tokenizer
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        return train_result, metrics_logger

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize trainer
    trainer = ResumeTrainer()
    
    # Prepare data
    train_data, val_data = trainer.prepare_data("resume_data.csv")
    
    # Train model
    results, metrics_logger = trainer.train(train_data, val_data)
    
    print("\nTraining completed. Metrics have been logged to:", metrics_logger.log_file)
    print("Final metrics have been saved to:", os.path.join(metrics_logger.log_dir, 'final_metrics.json'))

if __name__ == "__main__":
    main()