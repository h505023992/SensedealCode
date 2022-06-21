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
    def forward(self,batch):# batch只有n个seq_len的句子，输出就是n*3的logits
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
        # # batch["source_mask"] = batch["source_token_ids"] != self.tokenizer.pad_token_id
        # batch["target_mask"] = batch["target_token_ids"] != self.tokenizer.pad_token_id
        # # batch["source_position_ids"] = batch["source_mask"].cumsum(-1) - 1
        # batch["target_position_ids"] = batch["target_mask"].cumsum(-1) - 1
        batch_size = len(batch["target_token_ids"])
        device = batch["target_token_ids"].device

        past_mask = torch.ones(batch_size, 1).bool().to(device)
        batch["target_mask"] = batch["target_token_ids"] != self.tokenizer.pad_token_id
        batch["target_position_ids"] = batch["target_mask"].cumsum(-1)
        joint_mask = torch.cat((past_mask, batch["target_mask"]), dim=1)

        # (12,2, batch_size, num_head, sql_len+1, head_features)
        # past = torch.randn(size=(12, 2, batch_size, 12, 1, 61)).to(device)  # 64-3
        # temp = batch["source_token_ids"].unsqueeze(1).unsqueeze(2).unsqueeze(0).unsqueeze(0)
        # classification = torch.tile(temp, (12, 2, 1, 12, 1, 1))
        # past = torch.cat((classification, past), dim=-1)

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
            logits = logits[:, :-1].contiguous() #最后一个结束词的概率不要
            target_token_ids = batch["target_token_ids"][:, 1:].contiguous() # target_token_ids在生成器训练的时候是action，因为要求新老概率
            log_probs = torch.log_softmax(logits, dim=-1)
            # #计算熵正则
            # masks = batch["target_mask"][:, 2:].unsqueeze(-1)  # B*(S-1) 多加最后一维便于广播机制
            # log_probs_mask = log_probs * masks # B S-1 V
            # probs_mask = log_probs_mask.exp() # B  S-1 V
            # entropy_loss = -(log_probs_mask * probs_mask).sum(-1).sum(-1).mean()  # 梯度下降让负的信息熵越小，正的信息熵越大
            #只计算target的概率
            log_probs = torch.gather(log_probs,# B S
                                    dim=-1,
                                    index=target_token_ids.unsqueeze(-1)).squeeze(-1)
            # mask
            log_probs = (log_probs * batch["target_mask"][:, 1:]).sum(-1) # / mask.sum(-1) #与



            results = {"log_probs": log_probs, "loss": 0.0}
            return results

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
        self.l1 = nn.Linear(771, 256)#将特征和标签的onehot向量都输入进去
        self.active_l1 = F.relu
        self.l2 = nn.Linear(256, 1)
        self.active_l2 = torch.sigmoid

        # for param in self.model.parameters():
        #     param.requires_grad = True

    def cal(self,input,label):
        out = self.model(input)[1]
        x = torch.cat([out,label],dim=-1)
        return self.active_l2(self.l2(self.active_l1(self.l1(x))))

    def forward(self, batch):
        prob_ref = self.cal(batch["reference"],batch['label'])
        prob_can = self.cal(batch["candidate"],batch['label'])
        ref_loss = torch.log(prob_ref).mean()
        prob_can_loss = torch.log(prob_can).mean()
        loss = -(ref_loss-prob_can_loss)#最大化ref的概率，最小化can的概率，然后取负
        #loss.backward() #报错说明问题出在这
        # loss.requires_grad_(True)
        return {"reward": prob_can.detach().cpu().numpy(), "loss": loss,"ref_prob":prob_ref}

    def get_reward(self, candidate,label):
        prob_can = self.cal(candidate,label)
        return prob_can
    def classify_adv(self,expert,classify_label):
        prob_classify = self.cal(expert, classify_label)
        loss = torch.log(prob_classify).mean() #鉴别器要尽可能对这个东西判别为0，所以对这个值进行梯度下降
        #classification尽可能对这个东西判别为1，所以对这个值的负值进行梯度下降
#        real_loss.backward()
        return loss,prob_classify.detach()
