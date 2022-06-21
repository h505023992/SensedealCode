import hydra
import hydra.experimental
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchfly.flylogger import FlyLogger
from transformers import RobertaTokenizer
from omegaconf import DictConfig
#from TorchFly import  torchfly
from torchfly.text.decode import TransformerDecoder
from torchfly.common import set_random_seed
#from TorchFly.torchfly.text.rl import TextRLRewardFunc

from configure_dataloader import DataLoaderHandler
from torchfly.flyconfig import FlyConfig
from model import TextGAILModel
from textgail_discriminator import TextGAILDiscriminator
from textgail_trainerloop import TextGAILTrainerLoop
import logging

#@hydra.main(config_path="config/config.yaml", strict=False)
def main():
    config = FlyConfig.load()
    #config.task.weights_path = "/home/hqh/TextGail/Conditional/MLE/1.pth"

    config.task.weights_path = "../../MLE/outputs/Checkpoints/iter_1692_model_state.pth"
    print(config)
    fly_logger = FlyLogger(config)
    set_random_seed(config.training.random_seed)
    dataloader_handler = DataLoaderHandler(config)


    model = TextGAILModel(config)
    model_weights = torch.load(config.task.weights_path,map_location = "cuda:"+str(config.training.local_rank))
    model.generator.load_state_dict(model_weights, strict=False)

    #class载入
    class_pth = "/home/hqh/Triple-Gan/classify.pth"
    # model.classifier.model = torch.load(class_pth, map_location="cuda:" + str(config.training.local_rank))
    #model = model.cuda()

    # Register your transformer for decoding
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    decoder_helper = TransformerDecoder(config.decode)
    decoder_helper.register_generator(model.generator.decoder)
    decoder_helper.register_tokenizer(tokenizer)

    reward_func = TextGAILDiscriminator(config, tokenizer, model.discriminator)

    trainer = TextGAILTrainerLoop(config=config,
                                tokenizer = tokenizer,
                                reward_func=reward_func, 
                                decoder_helper=decoder_helper,
                                model=model, 
                                train_dataloader_fn=dataloader_handler.train_dataloader,
                                valid_dataloader_fn=dataloader_handler.valid_dataloader,
                                test_dataloader_fn=dataloader_handler.test_dataloader)
                            
    trainer.train()

if __name__ == "__main__":
    main()