import torch


def physics_loss_fn(model, device, D, physics_grid):
    # Ensure r and t have the right shape and require gradients
    r = physics_grid[:, 0].clone().detach().requires_grad_(True).reshape(-1, 1).to(device)
    t = physics_grid[:, 1].clone().detach().requires_grad_(True).reshape(-1, 1).to(device)

    # Pass r and t through the model
    C = model(torch.cat((r, t), dim=1))

    # Compute partial derivatives
    dC_dt = torch.autograd.grad(C, t, grad_outputs=torch.ones_like(C), create_graph=True)[0]
    dC_dr = torch.autograd.grad(C, r, grad_outputs=torch.ones_like(C), create_graph=True)[0]
    d2C_dr2 = torch.autograd.grad(dC_dr, r, grad_outputs=torch.ones_like(C), create_graph=True)[0]

    # Compute the differential equation residual
    lhs = dC_dt
    epsilon = 1e-6
    rhs = D * (d2C_dr2 + (1/(r + epsilon)) * dC_dr)
    residual = lhs - rhs

    # Physics loss: mean squared residual of the differential equation
    physics_loss = torch.mean(residual ** 2)

    return physics_loss
