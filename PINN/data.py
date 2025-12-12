
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from analytical_solution.solution import ConcentrationSolution


class DataGenerator:
    def __init__(self, r1, r2, c1, c2, t_start, t_finish, D):
        self.r1=r1
        self.r2=r2
        self.c1=c1
        self.c2=c2
        self.t_start = t_start
        self.t_finish = t_finish
        self.D=D

        self.C = ConcentrationSolution(r1=r1, r2=r2, c1=c1, c2=c2, D=D)

    def get_problem_grid(self, r1, r2, t_start, t_finish, N=10) -> np.ndarray:
        
        R = np.linspace(r1, r2, N)
        T = np.linspace(t_start, t_finish, N)
        # descartes product
        R_T = np.transpose([np.tile(R, len(T)), np.repeat(T, len(R))])

        return R_T
    
    def generate_physics_grid(self, r_min, r_max, t_min, t_max, num_points=50):
        r_values = torch.linspace(r_min, r_max, num_points)
        t_values = torch.linspace(t_min, t_max, num_points)

        X, Y = torch.meshgrid(r_values, t_values, indexing='ij')  # 'ij' indexing for Cartesian order

        # Reshape and concatenate to form Cartesian product
        cartesian_product = torch.stack([X.flatten(), Y.flatten()], dim=1)
        return cartesian_product

    def get_data(self, N = 10, sample=0.2):

        R = np.linspace(self.r1, self.r2, N)
        T = np.linspace(self.t_start, self.t_finish, N)
        # descartes product
        R_T = np.transpose([np.tile(R, len(T)), np.repeat(T, len(R))])

        C_values = []
        for r, t in R_T:
            C_values.append(self.C(r, t))
        C_values = np.array(C_values)


        size = int(len(C_values)*sample)
        train_indexis = np.random.choice(range(len(C_values)), size=size, replace=False)
        test_indexis = np.array([ind for ind in range(len(C_values)) if not ind in train_indexis])

        R_scaled = self.min_max_scale(R_T[:, 0])[0].reshape(-1, 1)
        T_scaled = self.min_max_scale(R_T[:, 1])[0].reshape(-1, 1)
        R_T_scaled = np.concatenate([R_scaled, T_scaled], axis=1)
        C_values_scaled = self.min_max_scale(C_values)[0].reshape(-1, 1)

        R_T_train_scaled = R_T_scaled[train_indexis, :]
        C_train_scaled = C_values_scaled[train_indexis]

        R_T_test_scaled = R_T_scaled[test_indexis, :]
        C_test_scaled = C_values_scaled[test_indexis]

        return R_T_train_scaled, C_train_scaled, R_T_test_scaled, C_test_scaled, R_T_scaled, C_values_scaled

    def min_max_scale(self, data, feature_range=(0, 1)):
        min_val = np.min(data)
        max_val = np.max(data)
        scale = (feature_range[1] - feature_range[0]) / (max_val - min_val)
        scaled_data = feature_range[0] + (data - min_val) * scale
        return scaled_data, min_val, max_val

    def inverse_min_max_scale(scaled_data, min_val, max_val, feature_range=(0, 1)):
        scale = (max_val - min_val) / (feature_range[1] - feature_range[0])
        original_data = min_val + (scaled_data - feature_range[0]) * scale
        return original_data


    def just_some_stuff():

        R_T_train, C_train, R_T_test, C_test, R_T, C_values = get_data(r1, r2, t_start, t_finish, N = 20, sample=0.01)

        R_train_scaled = min_max_scale(R_T_train[:, 0])[0].reshape(-1, 1)
        T_train_scaled = min_max_scale(R_T_train[:, 1])[0].reshape(-1, 1)
        R_T_train_scaled = np.concatenate([R_train_scaled, T_train_scaled], axis=1)

        R_test_scaled =  min_max_scale(R_T_test[:, 0])[0].reshape(-1, 1)
        T_test_scaled = min_max_scale(R_T_test[:, 1])[0].reshape(-1, 1)
        R_T_test_scaled = np.concatenate([R_test_scaled, T_test_scaled], axis=1)

        R_scaled =  min_max_scale(R_T[:, 0])[0].reshape(-1, 1)
        T_scaled = min_max_scale(R_T[:, 1])[0].reshape(-1, 1)
        R_T_scaled = np.concatenate([R_scaled, T_scaled], axis=1)

        R_T_train_tn = torch.tensor(R_T_train_scaled, dtype=torch.float32)
        R_T_test_tn = torch.tensor(R_T_test_scaled, dtype=torch.float32)

        R_T_tn = torch.tensor(R_T_scaled, dtype=torch.float32)

        R_train_scaled.shape, T_train_scaled.shape, R_T_train_scaled.shape, C_train.shape

        R_test_scaled.shape, T_test_scaled.shape, R_T_test_scaled.shape, C_test.shape

        R_scaled.shape, T_scaled.shape, R_T_scaled.shape, C_values.shape



