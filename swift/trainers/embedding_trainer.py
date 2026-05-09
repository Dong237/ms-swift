# Copyright (c) ModelScope Contributors. All rights reserved.
from swift.utils import get_logger
from .trainer import Trainer
from .utils import gather_for_unpadded_tensors

logger = get_logger()


class EmbeddingTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ip_ids = inputs.pop('ip_ids', None)
        if self.compute_loss_func is not None:
            labels = inputs.pop('labels', None)
            outputs = model(**inputs)
            if labels is not None:
                loss_kwargs = {'num_items_in_batch': num_items_in_batch}
                if ip_ids is not None:
                    # ip_ids may contain -1 for old-format rows; class-aware losses must ignore those anchors.
                    loss_kwargs['ip_ids'] = ip_ids
                loss = self.compute_loss_func(outputs, labels, **loss_kwargs)
            else:
                loss = outputs.loss

            if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
                loss = loss / self.args.gradient_accumulation_steps

            # Skip _compute_acc: embedding outputs are dicts with 'last_hidden_state',
            # not model outputs with .logits. Accuracy is not meaningful for retrieval tasks.

            return (loss, outputs) if return_outputs else loss
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors
        return output
