# fine_tune_agiasi.py
"""Fine-tune AGIASI model on the 4-class dataset.

Usage:
    python fine_tune_agiasi.py --model-path fused_model.pt --data-dir D:\\dataset\\final --output-dir checkpoints
"""
import argparse
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoConfig

class AGIASIWrapper(torch.nn.Module):
    """Wrapper to make the fused model compatible with HuggingFace Trainer."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = AutoConfig.from_pretrained("microsoft/Phi-3.5-mini-instruct") # Assuming base architecture
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Forward pass through the custom fused model
        # This assumes the fused model has a standard forward signature or similar
        outputs = self.model(input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return {"loss": loss, "logits": logits} if loss is not None else logits

    def save_pretrained(self, save_directory):
        torch.save(self.model, f"{save_directory}/agiasi_finetuned.pt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    # Load fused model
    print(f"Loading model from {args.model_path}...")
    raw_model = torch.load(args.model_path)
    model = AGIASIWrapper(raw_model)
    
    # Load tokenizer (assuming Phi-3.5 base)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    data_files = {
        "train": f"{args.data_dir}/train.jsonl",
        "validation": f"{args.data_dir}/val.jsonl"
    }
    dataset = load_dataset("json", data_files=data_files)

    def tokenize_function(examples):
        # Format: "Label: [LABEL]\nContent: [TEXT]"
        texts = [f"Label: {l}\nContent: {t}" for l, t in zip(examples["label"], examples["text"])]
        return tokenizer(texts, padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2, # RTX 3060
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        fp16=True, # Use mixed precision
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
