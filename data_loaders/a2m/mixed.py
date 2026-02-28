import os
import sys
from utils.misc import to_torch
from .dataset import Dataset
import glob
import numpy as np
import torch
from tools.io_tools import load_random_hint_from_file

class Mixed(Dataset):
    dataname = "mixed"
    def __init__(self, datapath="dataset/processed_datasets", split="train", controlnet=False, **kargs):
        self.datapath = datapath
        self.controlnet = controlnet
        kargs["pose_rep"] = "rotvec"
        kargs["max_len"] = 160
        kargs["num_frames"] = -1
        super().__init__(**kargs)
        data_subset_collections = [["babel/walk", "babel/turn", "humanml3d/jog_run", "humanml3d/walk_turn"], ["babel/sit", "humanml3d/sit"], ["babel/lie", "babel/lie"]]
        total_num_actions = len(data_subset_collections)
        self.num_actions = total_num_actions
        self._data_file_paths = []
        self._actions = []
        for action_id, data_subset_collection in zip(range(self.num_actions), data_subset_collections): 
            for data_subset in data_subset_collection:
                new_file_paths = sorted(glob.glob(os.path.join(datapath, data_subset, "*.npz")))
                self._data_file_paths.extend(new_file_paths)
                self._actions.extend([action_id] * len(new_file_paths))
        reorder_idx = np.random.permutation(len(self._data_file_paths))
        self._data_file_paths = [self._data_file_paths[i] for i in reorder_idx]
        self._actions = [self._actions[i] for i in reorder_idx]
        self._train = list(range(len(self._data_file_paths)))
        self._num_frames_in_video = [dict(np.load(self._data_file_paths[0], allow_pickle=True))["transl"].shape[0]] * len(self._data_file_paths)

        keep_actions = np.arange(0, total_num_actions)
        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}
        self._action_classes = mixed_coarse_action_enumerator

        return

    def _load(self, ind, frame_idx):
        file_path = self._data_file_paths[ind]
        data_dict = dict(np.load(file_path, allow_pickle=True))
        transl = data_dict["transl"][frame_idx]
        pose = data_dict["pose"][frame_idx,:66]
        nframes = transl.shape[0]
        pose_rep = self.pose_rep
        transl = to_torch(transl)
        pose = to_torch(pose)
        if pose_rep == "rotvec":
            pass
        elif pose_rep == "rot6d":
            pose = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose.view(nframes, -1, 3))).view(nframes, -1)
        else:
            raise NotImplementedError
        ret = torch.concat([pose, transl], dim=1).unsqueeze(-1)
        ret = ret.permute(1, 2, 0).contiguous()
        return ret.float()

    def get_controlnet_hint(self, ind, frame_idx):
        file_path = self._data_file_paths[ind]
        hint = load_random_hint_from_file(file_path)
        return hint
        

mixed_coarse_action_enumerator = {
    0: "walk",
    1: "sit",
    2: "lie",
}
if __name__ == "__main__":
    dataset = Mixed(datapath="/home/gongjingyu/gcode/RGBD/code/guided-motion-diffusion/dataset/processed_datasets", split="train", pose_rep="rotvec", num_frames=-1, max_len=160)
    print(dataset[0])
