import numpy as np
np.set_printoptions(threshold=1)
a = np.load("/home/lihan/ASE/ase/data/motions/dribble/amp_humanoid_run.npy",allow_pickle=True).item()
# print(a['rotation']['arr'].shape)
# print(a['root_translation']['arr'].shape)
# print(a['global_velocity']['arr'].shape)
# print(a['global_angular_velocity']['arr'].shape)
print(a.keys())