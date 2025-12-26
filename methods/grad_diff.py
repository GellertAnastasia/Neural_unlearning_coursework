import torch
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

class GradDiffTrainer(Trainer):
    def __init__(self, gamma=1.0, alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = inputs["forget"]
        retain_inputs = inputs["retain"]
        forget_outputs = model(**forget_inputs)
        retain_outputs = model(**retain_inputs)

        total_loss = -self.gamma * forget_outputs.loss + self.alpha * retain_outputs.loss
        
        return total_loss
    
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