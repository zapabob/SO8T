import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class GRAPEAlignmentScorer:
    """
    Groups and scores candidate responses based on base model alignment (logprob).
    Part of Phase 1: GRAPE -> SFT.
    """
    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def score_candidates(self, prompt: str, candidates: List[str]) -> List[float]:
        """
        Calculates alignment score (average logprob) for each candidate.
        """
        scores = []
        self.model.eval()
        
        for cand in candidates:
            # Combine prompt and candidate
            text = f"{prompt}{cand}"
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            target_ids = inputs["input_ids"].clone()
            
            # Mask the prompt part in loss calculation
            prompt_len = len(self.tokenizer(prompt)["input_ids"])
            target_ids[:, :prompt_len] = -100
            
            outputs = self.model(**inputs, labels=target_ids)
            # Alignment score = negative loss (higher is better aligned with model distribution)
            alignment = -outputs.loss.item()
            scores.append(alignment)
            
        return scores

def run_grape_sft_cycle(model, tokenizer, dataset, num_candidates=4):
    """
    Main loop for Phase 1: Selecting best candidates via GRAPE scoring then SFTing.
    """
    scorer = GRAPEAlignmentScorer(model, tokenizer)
    selected_data = []
    
    for item in dataset:
        prompt = item["instruction"]
        candidates = item["candidates"][:num_candidates]
        
        # 1. Base Alignment scoring
        scores = scorer.score_candidates(prompt, candidates)
        best_idx = np.argmax(scores)
        
        selected_data.append({
            "instruction": prompt,
            "output": candidates[best_idx],
            "alignment_score": scores[best_idx]
        })
        
    return selected_data
