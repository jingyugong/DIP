import os
import socket
import torch


class __GMDM_Project_Config():
    def __init__(self) -> None:
        self.get_project_dir_according_to_host()
        self.get_host_device_info()
        self.init_prox()
        self.init_random()
        self.init_replica()
        self.init_shapenet_real()
        self.init_control()
        self.register_to_global_space()


    def get_project_dir_according_to_host(self):
        self.hostname = socket.gethostname()
        self.map_hostname_to_dir = {
            "lagrange" : "/home/gongjingyu/gcode/RGBD/code/guided-motion-diffusion",
            "linus" : "/home/gongjingyu/gcode/RGBD/code/guided-motion-diffusion",
            "txd-T45" : "/data/zc/guided-motion-diffusion",
        }
        self.map_hostname_to_prefix = {
            "lagrange" : "dataset/dimos_data",
            "linus" : "dataset/dimos_data",
            "txd-T45" : "dataset",
        }
        self.project_dir = self.map_hostname_to_dir.get(self.hostname, None)
    

    def init_prox(self):
        "can load .yaml config here in the furture"
        self.prox_data_dir = os.path.join(self.project_dir, "dataset", "PROX_data")
        self.prox_room_list = ['BasementSittingBooth', 'MPH16', 'N0SittingBooth', 
                               'N3Office', 'MPH112', 'MPH1Library', 'N0Sofa', 
                               'N3OpenArea', 'MPH11', 'MPH8', 'N3Library', 'Werkraum']


    def init_random(self):
        self.random_scene_test_dir = os.path.join(self.project_dir, "dataset", "dimos_data", "scenes", "random_scene_test")
        self.test_room_list = ['test_room']


    def init_shapenet_real(self):
        self.shapenet_real_dir = os.path.join(self.project_dir, self.map_hostname_to_prefix.get(self.hostname, ""), "shapenet_real")
        self.shapenet_obj_list = {
            'Armchairs': ['9faefdf6814aaa975510d59f3ab1ed64',
                'cacb9133bc0ef01f7628281ecb18112',
                'ea918174759a2079e83221ad0d21775',],
            'L-Sofas': ['5cea034b028af000c2843529921f9ad7',],
            'Sofas': ['1dd6e32097b09cd6da5dde4c9576b854',
                '71fd7103997614db490ad276cd2af3a4',
                '277231dcb7261ae4a9fe1734a6086750',],
            'StraightChairs':['2ed17abd0ff67d4f71a782a4379556c7',
                '68dc37f167347d73ea46bea76c64cc3d',
                'd93760fda8d73aaece101336817a135f']
        }


    def init_replica(self):
        self.replica_dir = os.path.join(self.project_dir, self.map_hostname_to_prefix.get(self.hostname, ""), "replica")
        self.replica_room_list = ['office_0', 'office_1', 'office_2', 'office_3', 'office_4', 'room_0', 'room_1', 'hotel_0']


    def init_control(self):
        self.control_stand_poset_before_sit = False


    def get_host_device_info(self):
        if torch.cuda.is_available():
            self.host_device = "cuda"
            self.host_device_count = torch.cuda.device_count()
        else:
            self.host_device = "cpu"
            self.host_device_count = os.cpu_count()


    def register_to_global_space(self):
        for attr, value in vars(self).items():
            globals()[attr] = value
        

__config = __GMDM_Project_Config()
del __config
