import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

class WeightedGradAscentTrainer(Trainer):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        
        if labels is None:
            loss = -outputs.loss if outputs.loss is not None else None
            if loss is None:
                raise ValueError("No labels provided and model outputs do not contain loss.")
        else:
            mask = (labels != -100)
            
            if mask.sum() == 0:
                loss = torch.tensor(0.0, device=logits.device)
            else:
                log_probs = F.log_softmax(logits, dim=-1)
                nll = -torch.gather(log_probs, dim=-1, 
                                    index=labels.unsqueeze(-1)).squeeze(-1)
                if self.beta > 0:
                    probs = torch.gather(F.softmax(logits, dim=-1), dim=-1,
                                         index=labels.unsqueeze(-1)).squeeze(-1)
                    weights = probs ** self.beta
                    nll = weights * nll
                loss = - (nll * mask).sum() / mask.sum()
        
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)

        labels = None
        if hasattr(self, 'label_names') and self.label_names:
            labels = tuple(inputs.get(name) for name in self.label_names)
            if labels and all(l is not None for l in labels):
                labels = nested_detach(labels)
                if len(labels) == 1:
                    labels = labels[0]

        with torch.no_grad():
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

            if isinstance(outputs, dict):
                logits = outputs.get('logits')
            else:
                logits = outputs[1] if len(outputs) > 1 else None

        if prediction_loss_only:
            return (loss, None, None)
        
        if logits is not None:
            logits = nested_detach(logits)
        
        return (loss, logits, labels)
