import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

class QAModel(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = 'distilbert-base-cased-distilled-squad'
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def get_answer(self, question, context):
        # Tokenize the input
        inputs = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=512,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            padding="max_length"
        )
        
        # Process each chunk of the context
        best_answer = None
        best_score = -float('inf')
        
        for i in range(len(inputs["input_ids"])):
            # Get the input ids and attention mask for this chunk
            input_ids = inputs["input_ids"][i:i+1].to(self.model.device)
            attention_mask = inputs["attention_mask"][i:i+1].to(self.model.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
            
            # Get start and end logits
            start_logits = outputs.start_logits[0].cpu()
            end_logits = outputs.end_logits[0].cpu()
            
            # Find the best non-empty answer span
            start_idx = torch.argmax(start_logits)
            end_idx = torch.argmax(end_logits[start_idx:]) + start_idx
            
            if end_idx < start_idx:
                end_idx = start_idx + 1
                
            # Get answer tokens and decode
            answer_tokens = input_ids[0][start_idx:end_idx + 1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            # Calculate score
            score = float(start_logits[start_idx] + end_logits[end_idx])
            
            # Update best answer if this one is better
            if score > best_score and answer.strip() and len(answer.split()) >= 2:
                best_score = score
                best_answer = answer.strip()
        
        if not best_answer:
            return None
            
        return {
            "answer": best_answer,
            "score": best_score
        }
