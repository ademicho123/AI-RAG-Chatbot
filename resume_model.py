# train.py
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import json
import os
from tqdm import tqdm

class MetricsCallback:
    """Custom callback to track metrics during training"""
    def __init__(self):
        self.training_loss = []
        self.validation_loss = []
        self.learning_rates = []
        self.bleu_scores = []
        self.rouge_scores = []
        
    def on_log(self, args, state, control, logs=None):
        if logs is not None:
            if 'loss' in logs:
                self.training_loss.append(logs['loss'])
            if 'eval_loss' in logs:
                self.validation_loss.append(logs['eval_loss'])
            if 'learning_rate' in logs:
                self.learning_rates.append(logs['learning_rate'])

class ResumeTrainer:
    def __init__(self, model_name="meta-llama/Llama-2-7b", output_dir="./resume_generator_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.metrics_callback = MetricsCallback()
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize wandb for experiment tracking
        wandb.init(project="resume-generator", name="llama-finetuning")

    def prepare_data(self, data_path):
        """Prepare and validate the dataset"""
        df = pd.read_csv(data_path)
        
        # Data validation
        print("Dataset Statistics:")
        print(f"Total samples: {len(df)}")
        print(f"Unique job titles: {df['Job Title'].nunique()}")
        print("\nSample lengths distribution:")
        df['resume_length'] = df['Resume'].str.len()
        print(df['resume_length'].describe())
        
        # Format data
        def format_example(row):
            return {
                'input': f"Generate a resume for {row['Job Title']}:",
                'output': row['Resume']
            }
        
        formatted_data = df.apply(format_example, axis=1).tolist()
        
        # Split data
        train_data, val_data = train_test_split(
            formatted_data,
            test_size=0.1,
            random_state=42
        )
        
        return train_data, val_data

    def compute_metrics(self, eval_pred):
        """Compute various metrics for evaluation"""
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Calculate BLEU scores
        bleu_scores = [
            sentence_bleu([ref.split()], pred.split())
            for ref, pred in zip(decoded_labels, decoded_preds)
        ]
        
        # Calculate ROUGE scores
        rouge_scores = [
            self.scorer.score(ref, pred)
            for ref, pred in zip(decoded_labels, decoded_preds)
        ]
        
        # Calculate perplexity
        loss = torch.nn.CrossEntropyLoss()(
            torch.tensor(predictions).view(-1, self.model.config.vocab_size),
            torch.tensor(labels).view(-1)
        )
        perplexity = torch.exp(loss)
        
        metrics = {
            'bleu': np.mean(bleu_scores),
            'rouge1': np.mean([score['rouge1'].fmeasure for score in rouge_scores]),
            'rouge2': np.mean([score['rouge2'].fmeasure for score in rouge_scores]),
            'rougeL': np.mean([score['rougeL'].fmeasure for score in rouge_scores]),
            'perplexity': perplexity.item()
        }
        
        return metrics

    def train(self, train_data, val_data):
        """Train the model with comprehensive metrics tracking"""
        # Setup training arguments
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
            report_to="wandb"
        )
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': [self.tokenizer(x['input'])['input_ids'] for x in train_data],
            'labels': [self.tokenizer(x['output'])['input_ids'] for x in train_data]
        })
        
        val_dataset = Dataset.from_dict({
            'input_ids': [self.tokenizer(x['input'])['input_ids'] for x in val_data],
            'labels': [self.tokenizer(x['output'])['input_ids'] for x in val_data]
        })
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
            compute_metrics=self.compute_metrics,
            callbacks=[self.metrics_callback]
        )
        
        # Train
        print("Starting training...")
        train_result = trainer.train()
        
        # Save training metrics
        metrics = {
            'train_loss': self.metrics_callback.training_loss,
            'val_loss': self.metrics_callback.validation_loss,
            'learning_rates': self.metrics_callback.learning_rates,
            'bleu_scores': self.metrics_callback.bleu_scores,
            'rouge_scores': {
                'rouge1': train_result.metrics['eval_rouge1'],
                'rouge2': train_result.metrics['eval_rouge2'],
                'rougeL': train_result.metrics['eval_rougeL']
            }
        }
        
        # Save metrics to file
        with open(os.path.join(self.output_dir, 'training_metrics.json'), 'w') as f:
            json.dump(metrics, f)
        
        # Generate and save visualizations
        self._plot_metrics(metrics)
        
        # Save model and tokenizer
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        return metrics

    def _plot_metrics(self, metrics):
        """Generate and save visualization of training metrics"""
        # Create directory for plots
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['train_loss'], label='Training Loss')
        plt.plot(metrics['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, 'loss_curve.png'))
        plt.close()
        
        # Plot learning rate
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['learning_rates'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.savefig(os.path.join(plots_dir, 'learning_rate.png'))
        plt.close()
        
        # Plot ROUGE scores
        plt.figure(figsize=(10, 6))
        rouge_scores = metrics['rouge_scores']
        plt.bar(rouge_scores.keys(), rouge_scores.values())
        plt.title('ROUGE Scores')
        plt.ylabel('Score')
        plt.savefig(os.path.join(plots_dir, 'rouge_scores.png'))
        plt.close()

def main():
    # Initialize wandb logging
    wandb.login()
    
    # Create trainer instance
    trainer = ResumeTrainer()
    
    # Prepare data
    train_data, val_data = trainer.prepare_data("resume_data.csv")
    
    # Train model and get metrics
    metrics = trainer.train(train_data, val_data)
    
    # Print final metrics
    print("\nTraining completed. Final metrics:")
    print(f"Final training loss: {metrics['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {metrics['val_loss'][-1]:.4f}")
    print(f"ROUGE-1: {metrics['rouge_scores']['rouge1']:.4f}")
    print(f"ROUGE-2: {metrics['rouge_scores']['rouge2']:.4f}")
    print(f"ROUGE-L: {metrics['rouge_scores']['rougeL']:.4f}")
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()