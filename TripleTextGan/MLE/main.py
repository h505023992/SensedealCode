import hydra
import hydra.experimental
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torchfly.flylogger import FlyLogger
from torchfly.training import TrainerLoop
#from torchfly.common.download import get_pretrained_weights
from torchfly.common import set_random_seed
from torchfly.flyconfig import FlyConfig
from model import LanguageModel
from configure_dataloader import DataLoaderHandler
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
logger = logging.getLogger(__name__)

# @hydra.main(config_path="config", config_name="config.yaml")
# def my_app(cfg : DictConfig):
#     print(OmegaConf.to_yaml(cfg))
#     return OmegaConf.to_yaml(cfg)

def main():
    #config=my_app()
    config = FlyConfig.load()
    fly_logger = FlyLogger(config)
    #print(config)
    set_random_seed(config.training.random_seed)
    dataloader_handler = DataLoaderHandler(config)
    model = LanguageModel(config)
    trainer = TrainerLoop(config, model, dataloader_handler.train_dataloader, dataloader_handler.valid_dataloader)
    trainer.train()


if __name__ == "__main__":
    main()