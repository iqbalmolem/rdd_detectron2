import fvcore

import detectron2
from detectron2.config import get_cfg
from detectron2.config import LazyCall as L
from detectron2 import model_zoo

from tools.register_dataset import register_dataset
from tools.lazyconfig_train import do_test, do_train

def prepare_config():

    cfg = model_zoo.get_config("COCO-Detection/fcos_R_50_FPN_1x.py")

    cfg.lr_multiplier = L(detectron2.solver.WarmupParamScheduler)(scheduler=L(fvcore.common.param_scheduler.CosineParamScheduler)(start_value = 0.1, end_value=0.0001),
                                                                warmup_factor=0.05,
                                                                warmup_length=0.1)

    cfg.dataloader.evaluator = L(detectron2.evaluation.COCOEvaluator)(dataset_name='rddval', output_dir=cfg.train.output_dir)
    cfg.dataloader.test.dataset.names = 'rddtest'
    cfg.dataloader.train.dataset.names = 'rddtrain'
    cfg.dataloader.train.num_workers = 1
    cfg.dataloader.train.total_batch_size = 4
    cfg.train.max_iter= 100

    cfg.train.device='cuda'

def main():

    register_dataset()
    prepare_config()

    do_train()
