from typing import Optional, Union, Tuple

from sklearn.metrics import f1_score, classification_report
from torch.nn import BCEWithLogitsLoss
from transformers import BertPreTrainedModel, BertModel, EvalPrediction, BertTokenizerFast, Trainer, TrainingArguments
import torch
from transformers.modeling_outputs import SequenceClassifierOutput


class BertForEmo(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, 512),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, self.num_labels),
        )

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
          loss_fct = BCEWithLogitsLoss()
          loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def emo_compute_metrics(p: EvalPrediction, return_report=False, threshold=0.13):
    preds_softmax = torch.nn.functional.softmax(torch.Tensor(p.predictions),dim=1)
    preds = (preds_softmax > threshold).long()

    out = {'f1_micro': f1_score(p.label_ids, preds, average='micro'),
            'f1_macro': f1_score(p.label_ids, preds, average='macro')}
    if return_report:
      out['report'] = classification_report(p.label_ids, preds, digits=3)

    return out


# hf trainer for inference
def get_trainer_infer(checkpoint_dir, batch_size=4096):
    bert_model = BertForEmo.from_pretrained(checkpoint_dir)
    tokenizer = BertTokenizerFast.from_pretrained(checkpoint_dir)

    return Trainer(
        bert_model,
        TrainingArguments('emo-infer', per_device_eval_batch_size=batch_size),
        tokenizer=tokenizer,
    )