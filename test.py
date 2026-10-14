"""
# Created: 2023-08-09 10:28
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.

# Description: Output the evaluation results, go for local evaluation or online evaluation
"""

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig
import hydra, wandb, os, sys
from hydra.core.hydra_config import HydraConfig
from src.dataset import HDF5Dataset
from src.trainer import ModelWrapper
from src.utils import InlineTee
import open3d as o3d
import numpy as np

def precheck_cfg_valid(cfg):
    if os.path.exists(cfg.dataset_path + f"/{cfg.data_mode}") is False:
        raise ValueError(f"Dataset {cfg.dataset_path}/{cfg.data_mode} does not exist. Please check the path.")
    if cfg.supervised_flag not in [True, False]:
        raise ValueError(f"Supervised flag {cfg.supervised_flag} is not valid. Please set it to True or False.")
    if cfg.leaderboard_version not in [1, 2]:
        raise ValueError(f"Leaderboard version {cfg.leaderboard_version} is not valid. Please set it to 1 or 2.")
    return cfg

@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir

    if 'iter_only' in cfg.model and cfg.model.iter_only:
        from src.runner import launch_runner
        launch_runner(cfg, cfg.data_mode, output_dir)
        print(f"---LOG[eval]: Finished optimization-based evaluation. Logging saved to {output_dir}/output.log")
        return
    
    if not os.path.exists(cfg.checkpoint):
        print(f"Checkpoint {cfg.checkpoint} does not exist. Need checkpoints for evaluation.")
        sys.exit(1)
        
    torch_load_ckpt = torch.load(cfg.checkpoint)
    checkpoint_params = DictConfig(torch_load_ckpt["hyper_parameters"])
    cfg.output = checkpoint_params.cfg.output + f"-e{torch_load_ckpt['epoch']}-{cfg.data_mode}-v{cfg.leaderboard_version}"
    # replace output_dir ${old_output_dir} with ${output_dir}
    output_dir = output_dir.replace(HydraConfig.get().runtime.output_dir.split('/')[-2], checkpoint_params.cfg.output.split('/')[-1])
    cfg.model.update(checkpoint_params.cfg.model)
    cfg.num_frames = cfg.model.target.get('num_frames', checkpoint_params.cfg.get('num_frames', cfg.get('num_frames', 2)))
    
    mymodel = ModelWrapper.load_from_checkpoint(cfg.checkpoint, cfg=cfg, eval=True)
    os.makedirs(output_dir, exist_ok=True)
    sys.stdout = InlineTee(f"{output_dir}/output.log")
    print(f"---LOG[eval]: Loaded model from {cfg.checkpoint}. The backbone network is {checkpoint_params.cfg.model.name}.")
    print(f"---LOG[eval]: Evaluation data: {cfg.dataset_path}/{cfg.data_mode} set.\n")

    if cfg.wandb_mode != "disabled":
        logger = WandbLogger(save_dir=output_dir,
                            entity="kth-rpl",
                            project=f"opensf-eval", 
                            name=f"{cfg.output}",
                            offline=(cfg.wandb_mode == "offline"))
        logger.watch(mymodel, log_graph=False)
    else:
        # check local tensorboard logging: tensorboard --logdir logs/jobs/{log folder}
        logger = TensorBoardLogger(save_dir=output_dir, name="logs")
    
    trainer = pl.Trainer(logger=logger, devices=1)
    # NOTE(Qingwen): search & check: def eval_only_step_(self, batch, res_dict)

    # class PCDDataset(torch.utils.data.Dataset):
    #     def __init__(self):
    #         pass

    #     def __getitem__(self, idx):

    #         return item
    #     def __len__(self):
    #         return 1
    pcd1 = o3d.io.read_point_cloud("/home/kin/data/datasets/scene_1/pointcloud/1747643605.199571.pcd")
    points1 = torch.from_numpy(np.asarray(pcd1.points)).float()
    pcd2 = o3d.io.read_point_cloud("/home/kin/data/datasets/scene_1/pointcloud/1747643605.299568.pcd")
    points2 = torch.from_numpy(np.asarray(pcd2.points)).float()
    
    item = {
        "pc0": points1.unsqueeze(0).cuda(),
        "pc1": points2.unsqueeze(0).cuda(),
        "gm0": torch.zeros(points1.shape[0], dtype=torch.bool).unsqueeze(0).cuda(),
        "gm1": torch.zeros(points2.shape[0], dtype=torch.bool).unsqueeze(0).cuda(),
        "pose0": torch.eye(4).unsqueeze(0).cuda(),
        "pose1": torch.eye(4).unsqueeze(0).cuda(),
    }
    mymodel.cuda()
    out = mymodel.run_model_wo_ground_data(item)
    print(out[0].keys())
    print(out[1].keys())
    print(out[0]['pc0'].shape)
    print(out[0]['pc1'].shape)
    print(out[0]['pc0'][out[1]['pc0_valid_point_idxes']].shape)
    print(out[1]['flow'].shape)

    batch, res_dict = out
    eval_mask = batch['eval_mask'].squeeze()
    pc0 = batch['origin_pc0']
    transform_pc0 = pc0
    pose_flow = transform_pc0 - pc0

    final_flow = pose_flow.clone()
    if 'pc0_valid_point_idxes' in res_dict:
        valid_from_pc2res = res_dict['pc0_valid_point_idxes']

        # flow in the original pc0 coordinate
        pred_flow = pose_flow[~batch['gm0']].clone()
        pred_flow[valid_from_pc2res] = res_dict['flow'] + pose_flow[~batch['gm0']][valid_from_pc2res]
        final_flow[~batch['gm0']] = pred_flow

        print(final_flow.max())

if __name__ == "__main__":
    main()