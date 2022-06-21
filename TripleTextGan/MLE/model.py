from typing import Dict
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import math
import torchfly
from torchfly.nn.transformers import GPT2LMHeadModel
from torchfly.training import FlyModel
from torchfly.nn.losses import SequenceCrossEntropyLoss
from torchfly.metrics import Average
from torchfly.common.download import get_pretrained_weights

# pylint: disable=no-member


class LanguageModel(FlyModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.decoder = GPT2LMHeadModel(config.model)

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        # self.criterion = SequenceCrossEntropyLoss(reduce="sentence")
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.model.pad_token_id)
        self._perplexity = Average()

        # load pretrained weights
        #model_weights = torch.load("/home/hqh/TextGail/Conditional/MLE/1.pth")
        model_weights = torch.load("../../../1.pth")
        self.decoder.load_state_dict(model_weights, strict=False)
        #model_weights = get_pretrained_weights("roberta-tokenized-gpt2")
        # print(self.decoder.load_state_dict(model_weights, strict=False))

    def forward(self, batch):
        batch_size = len(batch["target_token_ids"])
        device = batch["target_token_ids"].device

        past_mask = torch.ones(batch_size, 1).bool().to(device)
        batch["target_mask"] = batch["target_token_ids"] != self.tokenizer.pad_token_id
        batch["target_position_ids"] = batch["target_mask"].cumsum(-1)
        joint_mask = torch.cat((past_mask, batch["target_mask"]), dim=1)



        logits, _ = self.decoder(
            input_ids=batch["target_token_ids"],
            position_ids=batch['target_position_ids'],
            attention_mask=joint_mask,
            past = batch["past"]
        )

        loss = self.compute_lm_loss(batch["target_token_ids"], logits, batch["target_mask"])
        results = {"loss": loss}
        # record training statistics,我后来发现少了，自己加上去的
        self.training_metrics["loss"](loss.item())
        return results

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        ppl = self._perplexity.get_metric(reset)
        metrics = {"perplexity": ppl}
        return metrics

    def get_training_metrics(self) -> Dict[str, str]:
        loss = self.training_metrics["loss"].get_metric()
        ppl = math.exp(loss)
        metrics = {"loss": (f"{loss:.4f}", loss), "ppl": (f"{ppl:.4f}", ppl)}
        return metrics

    def get_evaluation_metrics(self, reset: bool = False) -> Dict[str, float]:
        loss = self.evaluation_metrics["loss"].get_metric()
        ppl = math.exp(loss)
        metrics = {"loss": (f"{loss:.4f}", loss), "ppl": (f"{ppl:.4f}", ppl)}
        return metrics

    def compute_lm_loss(self, input_ids, logits, mask):
        logits = logits[:, :-1].contiguous()
        target = input_ids[:, 1:].contiguous()
        #mask = mask[:, 1:].float()
        #return self.criterion(logits, target, mask)
        return self.criterion(logits.view(-1, logits.size(-1)), target.view(-1))

    def predict(self, batch):
        results = self.forward(batch)
        #我后来发现少了，自己加上去的
        self.evaluation_metrics["loss"](results["loss"].item())
        self._perplexity(results["loss"].exp().item())