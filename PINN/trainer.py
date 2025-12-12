import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy

from PINN.model import SimpleNN
from PINN.data import DataGenerator
from PINN.loss_functions.physics import physics_loss_fn
from PINN.loss_functions.boundary import calc_boundary_loss


class Trainer:
    def __init__(self, args) -> None:
        self.args = args

    def train_model(self, args, D, num_epochs=6000,
                    alpha_boundary=2, alpha_physics=0.1, alpha_data=1, learning_rate=0.001):
        
        data_generator = DataGenerator(r1=args.r1, r2=args.r2, c1=args.c1, c2=args.c2,
                                    t_start=args.t_start, t_finish=args.t_finish,
                                    D=args.D)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = SimpleNN().to(device)
        

        R_T_train_scaled, C_train, R_T_test_scaled, C_test, R_T_scaled, C_values_scaled = data_generator.get_data()

        R_T_train_tn = torch.tensor(R_T_train_scaled, dtype=torch.float32)
        R_T_test_tn = torch.tensor(R_T_test_scaled, dtype=torch.float32)

        train_dataset = TensorDataset(R_T_train_tn, torch.tensor(C_train.reshape(-1, 1), dtype=torch.float32))
        test_dataset = TensorDataset(R_T_test_tn, torch.tensor(C_test.reshape(-1, 1), dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.L1Loss()

        test_best_error = float("inf")
        # Generate points over the entire interval for physics loss
        physics_grid = data_generator.generate_physics_grid(0.1, 0.9, 0.1, 0.9, num_points=300)

        best_model = copy.deepcopy(model)
        for epoch in range(num_epochs):
            model.train()

            running_loss = 0.0
            running_loss_phys = 0
            running_loss_data = 0
            running_loss_boundary = 0

            for X_batch, y_batch in train_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()

                # Forward pass and MSE loss computation
                outputs = model(X_batch)

                # Data loss
                mse_loss = criterion(outputs, y_batch)

                # Boundary loss
                boundary_loss = calc_boundary_loss(model, device, r1=0, r2=1,
                                                t_start=0, t_finish=1, N=10)
                # Physics-based loss
                physics_loss = physics_loss_fn(model, device, D, physics_grid)

                # Total loss
                loss = alpha_data * mse_loss
                loss += alpha_boundary * boundary_loss
                loss += alpha_physics * physics_loss

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * X_batch.size(0)
                running_loss_data += alpha_data * mse_loss.item() * X_batch.size(0)
                running_loss_phys += alpha_physics * physics_loss.item()
                running_loss_boundary += alpha_boundary * boundary_loss

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_loss_phys = running_loss_phys
            epoch_loss_data = alpha_data * (running_loss_data / len(train_loader.dataset))
            epoch_loss_boundary_loss = running_loss_boundary

            if (epoch + 1) % 500 == 0:

                test_error = self.eval_model(model, test_loader, device, criterion)
                print(f"Test error: {test_error}")
                print(f"Best model error: {test_best_error}, ")
                if test_best_error > test_error:
                    best_model = copy.deepcopy(model)
                    test_best_error = test_error
                    
                print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
                print(f'Epoch {epoch + 1}/{num_epochs}, Data Loss: {epoch_loss_data:.4f}')
                print(f'Epoch {epoch + 1}/{num_epochs}, Boundary Loss: {epoch_loss_boundary_loss:.4f}')
                print(f'Epoch {epoch + 1}/{num_epochs}, Physics Loss: {epoch_loss_phys:.4f}')

        print(f"Saving model to {args.result_model_path}")
        os.makedirs(os.path.dirname(args.result_model_path), exist_ok=True)
        torch.save(best_model.state_dict(), args.result_model_path)


    def eval_model(self, model, test_loader, device, criterion):
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(test_loader.dataset)
        return epoch_loss
