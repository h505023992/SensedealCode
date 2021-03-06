from typing import Any, List, Dict, Iterator, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, RobertaForMultipleChoice,RobertaModel,RobertaForSequenceClassification
#from TorchFly import torchfly
from torchfly.nn.transformers import GPT2LMHeadModel
from torchfly.training import FlyModel
#from TorchFly.torchfly.nn.losses import SequenceCrossEntropyLoss
from torchfly.metrics import Average
#from torchfly.common.download import get_pretrained_weights
#from TorchFly.torchfly.text.decode import TransformerDecoder
import ot


# pylint: disable=no-member


class TextGAILModel(FlyModel):
    def __init__(self, config):
        super().__init__(config)
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        self._perplexity = Average()
        if self.config.training.triple:
            self.classifier = Classifier(config)
    def configure_optimizers(self, total_num_update_steps) -> [List, List]:
        D_optimizer, D_scheduler = self.discriminator.configure_optimizers(total_num_update_steps)
        G_optimizer, G_scheduler = self.generator.configure_optimizers(total_num_update_steps)
        if self.config.training.triple:
            C_optimizer,C_scheduler = self.classifier.configure_optimizers(total_num_update_steps)
            self.optimizers = D_optimizer + G_optimizer+C_optimizer
            self.schedulers = D_scheduler + G_scheduler+C_scheduler
        else:
            self.optimizers = D_optimizer + G_optimizer
            self.schedulers = D_scheduler + G_scheduler
        return self.optimizers, self.schedulers

    def predict(self, batch):
        self.generator.rl_mode = False
        results = self.generator.forward(batch,rl_mode=False)
        self.generator.rl_mode = True
        self._perplexity(results["loss"].exp().item())

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        ppl = self._perplexity.get_metric(reset)
        metrics = {"perplexity": ppl}
        return metrics
class Classifier(FlyModel):
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base',num_labels=3)
    def forward(self,batch):# batch??????n???seq_len????????????????????????n*3???logits
        return self.model(batch).logits
class Generator(FlyModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.decoder = GPT2LMHeadModel(config.model)
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        # self.criterion = SequenceCrossEntropyLoss(reduce="sentence")
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.model.pad_token_id)
        self._perplexity = Average()

        # load pretrained weights
        # model_weights = get_pretrained_weights("roberta-tokenized-gpt2")
        # print(self.encoder.load_state_dict(model_weights, strict=False))
        self.rl_mode = True


    def forward(self, batch,rl_mode=True):
        #?????????????????????pad??????
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
            past=batch["past"]
        )
        if not rl_mode:
            loss = self.compute_lm_loss(batch["target_token_ids"], logits, batch["target_mask"])

            return {"loss": loss}
        else:
            logits = logits[:, :-1].contiguous() #???????????????????????????????????????????????????????????????????????????+1??????????????????????????????target
            target_token_ids = batch["target_token_ids"][:, 1:].contiguous() # target_token_ids??????????????????????????????action???????????????????????????
            log_probs = torch.log_softmax(logits, dim=-1)
            #?????????target?????????
            log_probs = torch.gather(log_probs,# B S
                                    dim=-1,
                                    index=target_token_ids.unsqueeze(-1)).squeeze(-1)

            # mask
            log_probs = ( log_probs  * batch["target_mask"][:, 1:]).sum(-1) # / mask.sum(-1) #???
            results = {"log_probs": log_probs, "loss": 0.0}
            return results
            #????????????????????????sum(-1)
            # probs = torch.softmax(logits, dim=-1)
            # #?????????target?????????
            # probs = torch.gather(probs,# B S
            #                         dim=-1,
            #                         index=target_token_ids.unsqueeze(-1)).squeeze(-1)
            # GAMMA = torch.tensor(0.99)
            # for i in range(probs.shape[-1],0,-1):
            #     probs[:,i-1] = probs[:,i-1]*torch.log(GAMMA)
            #     GAMMA*=GAMMA
            # # mask
            # log_probs = (probs * batch["target_mask"][:, 1:]).sum(-1) # / mask.sum(-1) #???
            # results = {"log_probs": log_probs, "loss": 0.0}
            # return results

    def predict(self, batch):
        results = self.forward(batch)
        self._perplexity(results["loss"].exp().item())

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        ppl = self._perplexity.get_metric(reset)
        metrics = {"perplexity": ppl}
        return metrics

    def compute_lm_loss(self, input_ids, logits, mask):
        logits = logits[:, :-1].contiguous()
        target = input_ids[:, 1:].contiguous()
        #mask = mask[:, 1:].float()
        #return self.criterion(logits, target, mask)
        return self.criterion(logits.view(-1, logits.size(-1)), target.view(-1))


class Discriminator(FlyModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.l1 = nn.Linear(771, 256)#?????????????????????onehot?????????????????????
        self.active_l1 = F.relu
        self.l2 = nn.Linear(256, 1)
        self.active_l2 = torch.sigmoid
        self.l_ot=nn.Linear(256,256)
        # for param in self.model.parameters():
        #     param.requires_grad = True

    def cal(self,input,label):
        out = self.model(input)[1]
        x = torch.cat([out,label],dim=-1)
        return self.active_l2(self.l2(self.active_l1(self.l1(x))))

    def cal2(self, input, label):
        out = self.model(input)[1]
        x = torch.cat([out, label], dim=-1)
        return self.l_ot(self.active_l1(self.l1(x)))
    def cal_ot(self,batch):
        ref_feature = self.cal2(batch["reference"],batch['label'])
        candidate_feature = self.cal2(batch["candidate"],batch['label'])
        C = 1 - torch.cosine_similarity(ref_feature.unsqueeze(0), candidate_feature.unsqueeze(1),
                                        dim=2)  # 1?????????????????????candi???0?????????????????????ref
        device = C.device
        with torch.no_grad():
            G = C  # ...???????????????????????????
            candidate_prob = torch.tensor([1. / len(candidate_feature)] * len(candidate_feature)).to(device)
            ref_prob = torch.tensor([1. / len(ref_feature)] * len(ref_feature)).to(device)
            max_iter = 10
            reg = 0.1
            numItermax = 1000
            for i in range(max_iter):
                T = ot.bregman.sinkhorn_log(candidate_prob, ref_prob, G, reg, numItermax)
                G = G - reg * torch.log(T)
        loss = -(T * C).sum()
        mean_r = ((T * C).sum(dim=1)).mean()
        ratio = ((T * C).sum() / mean_r)
        return {"reward": (-(T * C).sum(dim=1) * ratio).detach().cpu().numpy(), "loss": loss,"ref_prob":[0]}
    def forward(self, batch,weights):
        if  not self.config.text_gail.ot:
            prob_ref = self.cal(batch["reference"],batch['label'])
            prob_can = self.cal(batch["candidate"],batch['label'])
            prob_can_loss = torch.log(prob_can).mean()
            if weights is not None:
                ref_loss = (torch.log(prob_ref)*weights).mean()
                loss = -(2*ref_loss - prob_can_loss)
            else:
                ref_loss = torch.log(prob_ref).mean()
                loss = -(ref_loss-prob_can_loss)#?????????ref?????????????????????can????????????????????????
            #loss.backward() #???????????????????????????
            # loss.requires_grad_(True)

            return {"reward": prob_can.detach().cpu().numpy(), "loss": loss,"ref_prob":prob_ref}
        else:
            return self.cal_ot(batch)
    def get_reward(self, candidate,label):
        prob_can = self.cal(candidate,label)
        return prob_can.detach()
    def classify_adv(self,expert,classify_label):
        prob_classify = self.cal(expert, classify_label)
        loss = torch.log(prob_classify).mean() #?????????????????????????????????????????????0???????????????????????????????????????
        #classification?????????????????????????????????1????????????????????????????????????????????????
#        real_loss.backward()
        return loss,prob_classify.detach()
