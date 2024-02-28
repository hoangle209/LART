import sys
sys.path.insert(1, ".")

import torch
from lightning import LightningModule 

import hydra
from omegaconf import DictConfig, OmegaConf

from lart.models.lart_lite import LART_LitModule
from lart import utils
log = utils.get_pylogger(__name__)

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

@utils.task_wrapper
def load_model(cfg):
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    return model, None

@hydra.main(version_base="1.2", config_path="../configs", config_name="lart_lite.yaml")
def load_(cfg: DictConfig):
    model, _ = load_model(cfg)
    return model

if __name__ == "__main__":
    input = {
        "2D_joints": torch.rand(1, 125, 5, 28),
        "apperance": torch.rand(1, 125, 5, 2048),
        "has_detection": torch.ones(1, 125, 5, 1),
        "mask_detection": torch.zeros(1, 125, 5, 1)
    }

    # cfg = OmegaConf.load("configs/lart_lite_wo_hydra.yaml")
    model = load_()


