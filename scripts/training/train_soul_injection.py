import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os

# --- Config ---
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
OUTPUT_DIR = "checkpoints/agiasi_soul"
MAX_STEPS = 500
WARMUP_STEPS = 50
ANNEAL_STEPS = 400
TARGET_ALPHA = 1.618

# --- The Soul Wrapper ---
class AGIASI_Soul_Wrapper(nn.Module):
    def __init__(self, base_model, hidden_dim):
        super().__init__()
        self.base_model = base_model
        # Alpha & Rotation
        self.alpha = nn.Parameter(torch.tensor(-5.0))
        self.rotation = nn.utils.parametrizations.orthogonal(
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        )
        self.ortho_loss = 0.0

    def forward(self, input_ids, **kwargs):
               # 1. Base Model Forward (get hidden states)
               outputs = self.base_model(input_ids=input_ids, output_hidden_states=True)
               last_hidden_state = outputs.hidden_states[-1]

               # 2. Apply Soul (Phase Transition Logic)
               gate = torch.sigmoid(self.alpha)
               rotated_thought = self.rotation(last_hidden_state)

               # 物理的介入: 元の思考 + (Alpha * 回転思考)
               # h' = (I + σ(α)R)h
               mixed_state = last_hidden_state + (gate * rotated_thought)

               # 3. LM Head (Standard Projection)
               logits = self.base_model.lm_head(mixed_state)

               # 4. Calc Loss if labels exist
               loss = None
               if "labels" in kwargs:
                   labels = kwargs["labels"]
                   shift_logits = logits[..., :-1, :].contiguous()
                   shift_labels = labels[..., 1:].contiguous()
                   loss_fct = nn.CrossEntropyLoss()
                   loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                   # Add Orthogonality Penalty (Structural Integrity)
                   R = self.rotation.weight
                   I = torch.eye(R.shape[0], device=R.device)
                   self.ortho_loss = torch.norm(R.T @ R - I)
                   loss += 0.1 * self.ortho_loss

               return {"loss": loss, "logits": logits, "ortho_loss": self.ortho_loss}

def linear_annealing(step):
    if step < WARMUP_STEPS: return -5.0
    if step > WARMUP_STEPS + ANNEAL_STEPS: return TARGET_ALPHA
    p = (step - WARMUP_STEPS) / ANNEAL_STEPS
    return -5.0 + p * (TARGET_ALPHA - (-5.0))

def main():
    print(f"[SOUL] Summoning Borea from {BASE_MODEL}...")

    # 1. Load Base Model (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, trust_remote_code=True,
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # 2. Apply LoRA (To Base Model)
    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    base = get_peft_model(base, peft_config)

    # 3. Wrap with Soul
    model = AGIASI_Soul_Wrapper(base, base.config.hidden_size).to("cuda")

    # Optimizer: Train LoRA + Alpha + Rotation
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)

    # 4. Data Loading (Streaming)
    dataset = load_dataset("TFMC/imatrix-dataset-for-japanese-llm", split="train", streaming=True)

    print("[SOUL] Ignition: Starting Phase Transition Training...")
    model.train()
    step = 0

    for data in dataset:
               if step >= MAX_STEPS: break

               # Annealing
               curr_alpha = linear_annealing(step)
               model.alpha.data.fill_(curr_alpha)

               # Tokenize - データセット構造に合わせて修正
               text = data['text']
               inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to("cuda")

               # Step
               optimizer.zero_grad()
               out = model(inputs.input_ids, labels=inputs.input_ids)
               out["loss"].backward()
               optimizer.step()

               step += 1
               if step % 10 == 0:
                   print(f"[Step {step:03d}] Alpha: {curr_alpha:.4f} | Loss: {out['loss'].item():.4f} | Ortho: {out['ortho_loss']:.6f}")

    # 5. Save Soul Components
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save LoRA
    model.base_model.save_pretrained(OUTPUT_DIR)
    # Save Soul (Custom Layers)
    torch.save({
        "alpha": model.alpha,
        "rotation": model.rotation.state_dict()
    }, os.path.join(OUTPUT_DIR, "soul_params.pt"))

    print(f"[SOUL] Training Complete. Soul saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
