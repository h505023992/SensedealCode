from typing import Callable, Iterator
from omegaconf import DictConfig
import torch
import  torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import deque, namedtuple
from operator import itemgetter
import logging
import math
from torchfly.training import TrainerLoop
from torchfly.training.callbacks import Callback, CallbackHandler, Events
from torchfly.training.callbacks import Checkpoint
from torchfly.common import move_to_device
from torchfly.text.rl import TextRLReplayBuffer, TextRLLogHandler
from fast_bleu import BLEU, SelfBLEU
logger = logging.getLogger(__name__)

# pylint:disable=no-member


class TextGAILTrainerLoop(TrainerLoop):
    """
    On Policy Text RL Trainer
    """
    def __init__(self, config, tokenizer,reward_func, decoder_helper, collate_fn=None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.reward_func = reward_func
        self.decoder_helper = decoder_helper
        self.collate_fn = collate_fn
        self.tokenizer = tokenizer
        self.pad_token_id = self.decoder_helper._tokenizer.pad_token_id

        self.MLE_alpha = self.config.text_gail.MLE_alpha
        self.MLE_size = self.config.text_gail.MLE_size

        self.ppo_epoch = self.config.text_gail.ppo_epoch
        self.ppo_buffer_size = self.config.text_gail.ppo_buffer_size
        self.sample_batch_size = self.config.text_gail.sample_batch_size
        self.ppo_mini_batch_size = self.config.text_gail.ppo_mini_batch_size
        self.mix_human_demo_ratio = self.config.text_gail.mix_human_demo_init_ratio
        self.mix_human_demo_ratio_warmup_steps = self.config.text_gail.mix_human_demo_ratio_warmup_steps
        self.ppo_epsilon = self.config.text_gail.ppo_epsilon
        self.recompute_log_probs = self.config.text_gail.recompute_log_probs
        self.discriminator_pretrain_steps = self.config.text_gail.discriminator_pretrain_steps
        self.constant_human_demo_reward = self.config.text_gail.constant_human_demo_reward
        self.done_discriminator_pretrain = False

        self.replay_buffer = TextRLReplayBuffer(max_buffer_size=self.ppo_buffer_size)
        self.classification_log={'CE_real':"","adv":"",'CE_gen':""}
        self.G_lr = self.config.text_gail.G_lr

        self.gradient_accumulation_steps = 1
        self.tmp_vars = {}

        self.configure_bleu_function()
        self.best_valid_bleu = 0
        self.best_F1_bleu = 0
        # Configuration Check
        if self.gradient_accumulation_steps != 1:
            raise ValueError("Please set gradient accumulation steps to 1!")

        if self.training_in_epoch:
            raise ValueError(
                "Does not support epoch training! Please set config.training.total_num.num_steps bigger than 0."
            )

        if self.collate_fn is None:
            # Maybe it is defined in the train_dataloader
            self.collate_fn = self.train_dataloader.dataset.collate_fn

    def configure_optimizers(self):
        # The model will update multiple times in each update step
        # So we need to adjust the scheduler
        update_steps_multiplier = self.config.text_gail.ppo_buffer_size // self.config.text_gail.ppo_mini_batch_size
        return self.model.configure_optimizers(self.total_num_update_steps * update_steps_multiplier)

    def configure_callbacks(self):
        # Callback
        # by default set up LogHandler and Checkpointer
        self.checkpoint_callback = Checkpoint(self.config)
        self.add_callback(self.checkpoint_callback)

        if self.rank == 0:
            self.log_handler = TextRLLogHandler(self.config)
            self.add_callback(self.log_handler)

    def train_epoch(self):
        self.D_optimizer = self.optimizers[0]
        self.G_optimizer = self.optimizers[1]

        self.D_scheduler = self.schedulers[0]
        self.G_scheduler = self.schedulers[1]

        self.G_optimizer.param_groups[0]["lr"]=self.G_lr
        if self.config.training.triple:
            self.C_optimizer = self.optimizers[2]
            self.C_scheduler = self.schedulers[2]

        prop_decay = self.mix_human_demo_ratio / self.total_num_update_steps
        # prop_decay = 0
        # #确保一开始鉴别器的更新：
        # self.init_mix_human_demo_ratio = self.mix_human_demo_ratio
        # self.mix_human_demo_ratio = 0
        # total counts
        self.local_step_count = 0
        self.generator_step_count = 0
        self.discriminator_step_count = 0

        iter_train_dataloader = iter(self.train_dataloader)

        while self.global_step_count < self.total_num_update_steps:

            MLE_weight = max(((self.MLE_alpha * (
                    self.total_num_update_steps - 1.1*self.global_step_count)) / self.total_num_update_steps),0)
            self.callback_handler.fire_event(Events.BATCH_BEGIN)
            MLE_log = "MLE权重："+str(MLE_weight)
            # Collect samples
            buffer_count = 0
            while buffer_count < self.ppo_buffer_size:
                try:
                    batch = next(iter_train_dataloader)
                except StopIteration:
                    iter_train_dataloader = iter(self.train_dataloader)
                    self.epochs_trained += 1
                    logger.info(f"{self.epochs_trained} has finished!")
                    batch = next(iter_train_dataloader)
                batch_collated = self.collate_fn(batch)
                self.collect_samples(batch_collated)
                buffer_count += len(batch) #一开始是lazy collate，返回的是bs个字典
            # Discriminator Warmup
            if not self.done_discriminator_pretrain:
                for mini_batch in self.replay_buffer.iterate_sample(self.ppo_mini_batch_size):
                    states, experts, pasts, actions, action_log_probs, rewards, normalized_rewards, expert_rewards = zip(*mini_batch)
                    self.tmp_vars["log_dict"] = self.train_discriminator_step(torch.stack(states), actions, experts)
                    self.train_classifier_CE(experts, states, real=True)
                    #self.train_classifier_D(experts,pretrian_D=True)
                    # self.D_optimizer.zero_grad()#这一步正常训练的时候要注释掉
                    if (self.discriminator_step_count + 1) % self.gradient_accumulation_steps == 0:
                        # self.callback_handler.fire_event(Events.STEP_BEGIN)
                        self.D_optimizer.step()
                        self.D_scheduler.step()
                        self.D_optimizer.zero_grad()
                        # self.callback_handler.fire_event(Events.STEP_END)

                self.discriminator_step_count += 1

                if self.discriminator_step_count >= self.discriminator_pretrain_steps:
                    # self.mix_human_demo_ratio = self.init_mix_human_demo_ratio
                    self.done_discriminator_pretrain = True
                    print("Pretrain for D is completed!")
                    #break
                #我写的优化版本

                # self.callback_handler.fire_event(Events.BACKWARD_BEGIN)
                # self.loss_backward(reward_loss_prob['loss'])
                # self.callback_handler.fire_event(Events.BACKWARD_END)
                # self.D_optimizer.step()
                # self.D_scheduler.step()
                # self.D_optimizer.zero_grad()
                # # return the results
                # self.tmp_vars["log_dict"] = {"discriminator/loss": reward_loss_prob['loss'].item(),
                #             "Generated_Prob:": reward_loss_prob['reward'].mean().item(),
                #             "Real_Prob:": reward_loss_prob['ref_prob'].mean().item()}
                # self.discriminator_step_count += 1
                # if self.discriminator_step_count >= self.discriminator_pretrain_steps:
                #     # self.mix_human_demo_ratio = self.init_mix_human_demo_ratio
                #     self.done_discriminator_pretrain = True
                #     print("Pretrain for D is completed!")
                #     break
            else:
                # CLASS
                if self.config.training.triple:
                    for mini_batch in self.replay_buffer.iterate_sample(self.ppo_mini_batch_size):
                        "Classification CE：cross-entropy"
                        states,experts,pasts, actions, action_log_probs, rewards, normalized_rewards,expert_rewards = zip(*mini_batch)
                        self.train_classifier_CE(actions,states)
                        self.train_classifier_CE(experts,states,real=True) # expert_states是tensor,cuda:0
                        "Classification adv"
                        self.train_classifier_D(experts)


                "Generator adv Training"
                for _ in range(self.ppo_epoch):
                    # Train the Generator
                    for mini_batch in self.replay_buffer.iterate_sample(self.ppo_mini_batch_size):

                        # (state, action, action_log_prob, reward, normalized_reward)
                        states,experts,pasts, actions, action_log_probs, rewards, normalized_rewards,expert_rewards = zip(*mini_batch)

                        # ppo_batch = self.collate_fn(states)
                        #
                        ppo_batch = {}
                        ppo_batch["target_token_ids"] = pad_sequence(
                            actions, batch_first=True, padding_value=self.pad_token_id
                        )
                        # ppo_batch["target_token_ids"] = torch.stack(actions)
                        ppo_batch["past"] = torch.FloatTensor(torch.stack(pasts).transpose(0,2))
                        ppo_batch["normalized_rewards"] = torch.FloatTensor(np.array(normalized_rewards))
                        ppo_batch["old_log_probs"] = torch.FloatTensor(action_log_probs)
                        #ppo_batch["old_log_probs"] = torch.stack(action_log_probs)
                        ppo_batch = move_to_device(ppo_batch, self.device)

                        self.tmp_vars["log_dict"] = self.train_generator_step(ppo_batch)

                            # self.callback_handler.fire_event(Events.STEP_BEGIN)
                        self.G_optimizer.step()
                        self.G_scheduler.step()
                        self.G_optimizer.zero_grad()
                            # self.callback_handler.fire_event(Events.STEP_END)
                    self.generator_step_count += 1
                "Generator MLE adjusting"
                MLE_log2=''
                #self.MLE_alpha = ((self.MLE_alpha*self.global_step_count)/self.total_num_update_steps)
                if MLE_weight>0:
                    for i in range(0,len(batch_collated['target_token_ids']),self.MLE_size):
                        mini_batch = {}
                        mini_batch ['past']= batch_collated ['past'][:,:,i:i + self.MLE_size]
                        mini_batch['target_token_ids'] = batch_collated['target_token_ids'][i:i + self.MLE_size]
                        mini_batch = move_to_device(mini_batch,self.device)
                        output = self.model.generator(mini_batch, rl_mode=False)
                        self.callback_handler.fire_event(Events.BACKWARD_BEGIN)
                        self.loss_backward(output['loss']*MLE_weight)
                        self.callback_handler.fire_event(Events.BACKWARD_END)
                        self.G_optimizer.step()
                        self.G_scheduler.step()
                        self.G_optimizer.zero_grad()
                    MLE_log2 ="| " \
                              "MLE损失：" + str(output['loss'].item())
                "Discriminator Training"
                for mini_batch in self.replay_buffer.iterate_sample(self.ppo_mini_batch_size):
                    states,experts,pasts, actions, action_log_probs, rewards, normalized_rewards,expert_rewards = zip(*mini_batch)
                    log_dict = self.train_discriminator_step(torch.stack(states), actions, experts,expert_rewards)
                    self.tmp_vars["log_dict"].update(log_dict)
                    #self.D_optimizer.zero_grad()#这一步正常训练的时候要注释掉
                        # self.callback_handler.fire_event(Events.STEP_BEGIN)
                    self.D_optimizer.step()
                    self.D_scheduler.step()
                    self.D_optimizer.zero_grad()
                        # self.callback_handler.fire_event(Events.STEP_END)

                self.discriminator_step_count += 1
                if MLE_weight > 0 :
                    logger.info(MLE_log + MLE_log2)
                # self.callback_handler.fire_event(Events.BACKWARD_BEGIN)
                # self.loss_backward(reward_loss_prob['loss'])
                # self.callback_handler.fire_event(Events.BACKWARD_END)
                # self.D_optimizer.step()
                # self.D_scheduler.step()
                # self.D_optimizer.zero_grad()
                # # return the results
                # self.tmp_vars["log_dict"] = {"discriminator/loss": reward_loss_prob['loss'].item(),
                #                              "Generated_Prob:": reward_loss_prob['reward'].mean().item(),
                #                              "Real_Prob:": reward_loss_prob['ref_prob'].mean().item()}
                # self.discriminator_step_count += 1
            # update human mix_human_ratio
            self.mix_human_demo_ratio -= prop_decay 
            # self.tmp_vars["log_dict"]["mix_human_demo_ratio"] = self.mix_human_demo_ratio

            self.callback_handler.fire_event(Events.BATCH_END)
            self.replay_buffer.clear()


            logger.info(self.classification_log['CE_real'] +self.classification_log['CE_gen']+ self.classification_log['adv'])
            # Only rank 0 can run the validation dataset
            if self.rank == 0:
                if (self.global_step_count + 1) % self.config.training.validation.steps_interval == 0 and self.done_discriminator_pretrain:
                    if not self.validation_dataloader is None:
                        self.model.eval()
                        # BEGIN
                        self.callback_handler.fire_event(Events.VALIDATE_BEGIN)
                        self.tmp_vars["validate_metrics"] = self.validate()
                        self.callback_handler.fire_event(Events.VALIDATE_END)

                        # self.callback_handler.fire_event(Events.TEST_BEGIN)
                        # self.tmp_vars["test_metrics"] = self.test()
                        # self.callback_handler.fire_event(Events.TEST_END)

                        self.model.train()
            
            self.global_step_count += 1
            self.local_step_count += 1
    def train_classifier_CE(self,sequence,label_onehot,real = False):
        label_onehot = torch.stack(label_onehot)  # n*3,tensor
        label = torch.argmax(label_onehot,dim=-1).to(self.device) # n tensor
        sequence = pad_sequence(sequence, batch_first=True,
                               padding_value=self.pad_token_id).to(self.device)  # n*len,tensor
        sequence_logits = self.model.classifier(sequence)
        #sequence_log_prob = F.log_softmax(sequence_logits)
        sequence_cross_entropy_loss = F.cross_entropy(sequence_logits, label)

        if real:
            self.callback_handler.fire_event(Events.BACKWARD_BEGIN)
            sequence_cross_entropy_loss.backward(retain_graph=True)#这一步保存C对专家标签输出的值产生的计算图
            self.callback_handler.fire_event(Events.BACKWARD_END)
            self.C_optimizer.step()
            self.C_scheduler.step()
            self.C_optimizer.zero_grad()
            acc_real = (torch.argmax(sequence_logits,dim=-1) == label).sum()
            self.classification_log['CE_real']= " | 分类器对真实数据的分类正确率"+str((acc_real/len(label)).item())
            #return F.softmax(sequence_logits,dim=-1)#对于真实句子，返回one-hot类别向量，拼接起来输送给D鉴别
        else:
            self.callback_handler.fire_event(Events.BACKWARD_BEGIN)
            self.loss_backward(sequence_cross_entropy_loss)
            self.callback_handler.fire_event(Events.BACKWARD_END)
            self.C_optimizer.step()
            self.C_scheduler.step()
            self.C_optimizer.zero_grad()
            acc_real = (torch.argmax(sequence_logits,dim=-1) == label).sum()
            self.classification_log['CE_gen'] = " | 分类器对生成数据的分类正确率" + str((acc_real / len(label)).item())
    def train_classifier_D(self,sequence,pretrian_D=False):

        #sequence:cpu n*len [tensor,tensor...tensor] list, generated sequences list,consist of tensors
        sequence = pad_sequence(sequence, batch_first=True,
                                padding_value=self.pad_token_id).to(self.device)  # n*len,tensor
        sequence_logits = self.model.classifier(sequence)
        classify_label = F.softmax(sequence_logits, dim=-1)
        if pretrian_D:
            real_loss, _ = self.model.discriminator.classify_adv(sequence, classify_label.detach())
            # 鉴别器参数更新：
            self.callback_handler.fire_event(Events.BACKWARD_BEGIN)
            self.loss_backward(real_loss)
            self.callback_handler.fire_event(Events.BACKWARD_END)
            self.D_optimizer.step()
            self.D_scheduler.step()
            self.D_optimizer.zero_grad()  # 这个不能少貌似？
        else:
            loss,_ = self.model.discriminator.classify_adv(sequence,classify_label.detach())
        # 鉴别器参数更新：
            real_loss = loss
            self.callback_handler.fire_event(Events.BACKWARD_BEGIN)
            self.loss_backward(real_loss)#0.1
            self.callback_handler.fire_event(Events.BACKWARD_END)
            self.D_optimizer.step()
            self.D_scheduler.step()
            self.D_optimizer.zero_grad()# 这个不能少貌似？
            #self.C_optimizer.zero_grad() #要把D反传时对传到C里面梯度清掉
            #不需要了，我传给他的是detach掉的sequence，反传时C不会产生梯度

            # 分类器参数更新
            loss, prob_classify = self.model.discriminator.classify_adv(sequence, classify_label)
            fake_loss = -loss
            self.callback_handler.fire_event(Events.BACKWARD_BEGIN)
            self.loss_backward(fake_loss)#5
            self.callback_handler.fire_event(Events.BACKWARD_END)
            self.C_optimizer.step()
            self.C_scheduler.step()
            self.C_optimizer.zero_grad()
            self.D_optimizer.zero_grad() #要把D反传时对传到D里面梯度清掉
            self.classification_log ['adv']=' | 判别器对分类器的输出判断的信号'+str(prob_classify.mean().item())

    def train_discriminator_step(self, states, actions, experts,rewards=None):
        if rewards is not None and not self.config.text_gail.ot:
            r = torch.FloatTensor(np.array(rewards))
            r_min = torch.min(r)
            r_mid = r.mean()
            N_pre = self.discriminator_pretrain_steps
            N = self.total_num_update_steps
            j = self.global_step_count
            k_j = r_mid - math.pow((j - N_pre) / N, 1) * (r_mid - r_min)
            w = 1 / (1 + torch.exp(k_j - r))
            results = self.reward_func.get_loss(states, actions, experts, weights=w)
        else:
            results = self.reward_func.get_loss(states, actions, experts,weights=None)

        self.callback_handler.fire_event(Events.BACKWARD_BEGIN)
        loss = results["loss"] / self.gradient_accumulation_steps
        self.loss_backward(loss)
        self.callback_handler.fire_event(Events.BACKWARD_END)
        # return the results
        if not self.config.text_gail.ot:
            log_dict = {"discriminator/loss": loss.item() * self.gradient_accumulation_steps,"Generated_Prob:":results["reward"].mean().item(),"Real_Prob:":results["ref_prob"].mean().item()}
        else:
            log_dict = {"discriminator/loss": loss.item() * self.gradient_accumulation_steps,"ot": results["reward"].mean().item()}
        return log_dict

    def train_generator_step(self, batch):
        results = self.model.generator(batch)

        # old log probilities
        log_probs = results["log_probs"]#.sum(-1)#n*len
        old_log_probs = batch["old_log_probs"]#.sum(-1)

        # advantages
        batch_rewards = batch["normalized_rewards"]
        advantages = batch_rewards

        # self-imitation
        # advantages = advantages.clamp(min=0)

        # Policy Loss
        ## shape: (batch)
        ratio = (log_probs - old_log_probs.detach()).exp()#我自己加了个detach
        ## shape: (batch)
        #
        # advantages = torch.tile(advantages , (1, log_probs.shape[-1]))#batch_size seq_len
        # advantages = advantages/advantages.shape[-1]
        # r =0.8
        # GAMMA = 1
        # for i in range(advantages.shape[-1],0,-1):
        #     advantages[:,i-1] = advantages[:,i-1]*GAMMA
        #     GAMMA = GAMMA*r
        #
        policy_loss1 = -advantages * ratio
        ## shape: (batch)
        policy_loss2 = -advantages * ratio.clamp(1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon)

        policy_loss = torch.max(policy_loss1, policy_loss2).mean()
        #entropy_loss
        #entropy_loss = self.config.training.entropy_lamda * results["entropy_loss"]
        loss = policy_loss # + results["loss"] #+ entropy_loss

        # Backward Loss
        self.callback_handler.fire_event(Events.BACKWARD_BEGIN)
        loss = loss / self.gradient_accumulation_steps
        self.loss_backward(loss)
        self.callback_handler.fire_event(Events.BACKWARD_END)

        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > self.ppo_epsilon).float().mean()
            approx_kl = (log_probs - old_log_probs).pow(2).mean()

        log_dict = {}
        log_dict["generator/loss"] = loss.item() * self.gradient_accumulation_steps
        log_dict["generator/policy_loss"] = policy_loss.item()
        log_dict["generator/clip_frac"] = clip_frac.item()
        log_dict["generator/approx_kl"] = approx_kl.item()

        return log_dict

    def collect_samples(self, sample_batch_collated):
        """Generate samples, collect rewards, and update replay buffer"""

        # Update Buffer for Generations
        # actual_sample_size = min(len(batch) - num_human_demos, self.sample_batch_size)
        # actual_sample_size = self.sample_batch_size
        # if actual_sample_size > 0:
        #     # for i in range(num_human_demos, len(batch), actual_sample_size):
        #     for i in range(0, len(batch), actual_sample_size):
        #
        #
        #         sample_batch = batch[i:i + actual_sample_size]
        # sample_batch_collated = self.collate_fn(sample_batch)
        #sample_batch_collated = move_to_device(sample_batch_collated, self.device)
        with torch.no_grad():
            results = self.decoder_generate(sample_batch_collated)

            # TODO: Consider num_return_sequences
            results["tokens"] = [item[0] for item in results["tokens"]] #results是一个三维矩阵 这是batch 1 sequence_len 的
            # logger.info("试一试有没有信息学")
            # 得到句子之后，采用截断的方法重新计算log概
            # 这个我给取消了
            if self.recompute_log_probs:
                experts = sample_batch_collated["target_token_ids"]
                sample_batch_collated["target_token_ids"] = pad_sequence(
                    results["tokens"], batch_first=True, padding_value=self.pad_token_id
                )
                sample_batch_collated_device = move_to_device(sample_batch_collated, self.device)
                log_probs = self.model.generator(sample_batch_collated_device)["log_probs"]
                #results["log_probs"] = log_probs.tolist()
                results["log_probs"] = [item for item in log_probs]#list of tensors
                # we switch back the original target_token_ids
            else:
                results["log_probs"] = [item[0] for item in results["log_probs"]]
        if not self.config.text_gail.ot:
            expert_rewards = self.reward_func.get_reward(sample_batch_collated ["source_token_ids"],experts,gen_reward=False)
            rewards = self.reward_func.get_reward(sample_batch_collated ["source_token_ids"],results["tokens"])
        else:
            expert_rewards = self.reward_func.get_reward(sample_batch_collated["source_token_ids"], experts,
                                                         gen_reward=False)
            rewards=self.reward_func.get_loss(sample_batch_collated ["source_token_ids"],results["tokens"],experts)['reward']
        #print(len(sample_batch_collated["target_token_ids"][0]),len(results["log_probs"][0]))
        self.replay_buffer.update_batch(
            states=sample_batch_collated ["source_token_ids"],pasts=sample_batch_collated["past"].transpose(0,2),experts=experts,
            actions= sample_batch_collated["target_token_ids"]#results["tokens"]
            #sample_batch_collated["target_token_ids"]#
            , action_log_probs=results["log_probs"], rewards=rewards,expert_rewards=expert_rewards
        )

        # reward_loss_prob = self.reward_func.get_loss(sample_batch_collated["source_token_ids"], results["tokens"],
        #                                         experts=sample_batch_collated["target_token_ids"])
        #
        # self.replay_buffer.update_batch(
        #     states=sample_batch_collated["source_token_ids"], pasts=sample_batch_collated["past"].transpose(0, 2),
        #     experts=sample_batch_collated["target_token_ids"],
        #     actions=results["tokens"], action_log_probs=results["log_probs"], rewards=reward_loss_prob["reward"]
        # )
        # return reward_loss_prob
    def decoder_generate(self, batch):
        # force it not to use beam search
        #input_ids = batch["target_token_ids"][:, :1] #只传入BOS
        #model_inputs[input_ids]和model_inputs[past]
        #model_inputs = model_inputs

        #device = batch["target_token_ids"].device

        model_inputs = {}
        model_inputs['input_ids'] = batch["target_token_ids"][:, :1].to(self.device)
        model_inputs['past'] = batch["past"].to(self.device)

        results = self.decoder_helper.generate(model_inputs=model_inputs)
        return results

    def configure_bleu_function(self):
        ref=[]
        for batch in self.validation_dataloader:
            token = [item["target_token_ids"][1:-1] for item in batch]#去掉起始和结尾
            ref.extend(token)
        #print(ref[0:10])
        self.valid_bleu_function = BLEU(ref, {'4gram': (0.25, 0.25, 0.25, 0.25)})
        ref = []
        for batch in self.test_dataloader:
            token = [item["target_token_ids"][1:-1] for item in batch]
            ref.extend(token)
        self.test_bleu_function = BLEU(ref, {'4gram': (0.25, 0.25, 0.25, 0.25)})

    def validate(self):
        # Validation
        self.model.eval()
        # No gradient is needed for validation
        with torch.no_grad():
            for batch in self.validation_dataloader:
                batch = self.collate_fn(batch)
                batch = move_to_device(batch, self.device)
                if self.distributed_training:
                    self.model.module.predict(batch)
                else:
                    self.model.predict(batch)
        #END
        #计算bleu和self-bleu
            styles = ['fact', 'romantic', 'funny']
            s = {}
            s["fact"] = torch.tensor([1, 0, 0, 0])
            s["romantic"] = torch.tensor([0, 1, 0, 0])
            s["funny"] = torch.tensor([0, 0, 1, 0])
            num = 256
            generated = []
            temperature = 0.5
            for style in styles:
                #32*16=512
                for i in range(1):
                    model_inputs = {}
                    model_inputs["input_ids"] = torch.zeros((num, 1), dtype=int).to(self.device)
                    temp = s[style].unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    model_inputs['past'] = torch.tile(temp, (12, 2, num, 12, 1, 16)).to(self.device)
                    results = self.decoder_helper.generate(model_inputs=model_inputs, temperature=temperature)
                    tmp = [item[0][1:-1].cpu() for item in results["tokens"]]
                    generated.extend(tmp)
            #print(generated[0:10])
            valid_bleu_score = np.mean(self.valid_bleu_function.get_score(generated)['4gram'])
            test_bleu_score = np.mean(self.test_bleu_function.get_score(generated)['4gram'])
            self_bleu = SelfBLEU(generated, {'4gram': (0.25, 0.25, 0.25, 0.25)})
            self_bleu_score = np.mean(self_bleu.get_score()['4gram'])
            #print(bleu_score)
            F1_bleu = (2*valid_bleu_score*(1-self_bleu_score))/(valid_bleu_score+(1-self_bleu_score))
            #(2*valid_bleu_score*(1-self_bleu_score))/(valid_bleu_score+(1-self_bleu_score)) #valid_bleu_score+(1-self_bleu_score)#

            #     self.best_step=self.global_step_count
            if self.best_F1_bleu<F1_bleu:
                self.best_F1_bleu = F1_bleu
                self.best_F1_step = self.global_step_count
                self.best_valid_bleu = valid_bleu_score
                self.best_self_bleu = self_bleu_score
                torch.save(self.get_model_state(), self.config.training.checkpointing.directory+"/best_f1_bleu_model.pth")
                logger.info("------------------------------save model!----------------------------")
            logger.info("With temperature = " + str(temperature) +
                        " | F1 bleu_score : " + str(F1_bleu)+
                        " | valid bleu_score : " + str(valid_bleu_score)+ " | self-bleu:" + str(self_bleu_score) +
                        " | test bleu_score : " + str(test_bleu_score) )
            #logger.info("best bleu step" + str(self.best_step) +"| best bleu: "+ str(self.best_valid_bleu))
            logger.info("best(Saved) F1-bleu step: " + str(self.best_F1_step) + "| best F1 bleu: " + str(self.best_F1_bleu)+
                        "| bleu: " + str(self.best_valid_bleu) + "| self-bleu: " + str(self.best_self_bleu)
                        )
            #
            # get metrics


    def test(self):
        # test
        self.model.eval()
        # No gradient is needed for test
        with torch.no_grad():
            for batch in self.test_dataloader:
                batch = self.collate_fn(batch)
                # batch_size = len(batch["target_token_ids"])
                # past = torch.randn(size=(12, 2, batch_size, 12, 1, 61))  # .to(self.device)  # 61=64-3
                # temp = batch["source_token_ids"].unsqueeze(1).unsqueeze(2).unsqueeze(0).unsqueeze(0)
                # classification = torch.tile(temp, (12, 2, 1, 12, 1, 1))
                # batch["past"] = torch.cat((classification, past), dim=-1)
                # send to cuda device
                batch = move_to_device(batch, self.device)

                if self.distributed_training:
                    self.model.module.predict(batch)
                else:
                    self.model.predict(batch)
        #END
        # get metrics
        if self.distributed_training:
            metrics = self.model.module.get_metrics(reset=True)
        else:
            metrics = self.model.get_metrics(reset=True)
        return metrics