import torch


def calc_boundary_loss(model, device, r1=0, r2=1, t_start=0, t_finish=1, N=10,):
  """
    should be c1 - 4, but C scaled for NN training
    C(r1, t) = 1
    C(r2, t) = 0
    C(r, t_start) = 0
  """
  t = torch.linspace(t_start, t_finish, N).reshape(-1, 1)
  r1_t = torch.cat((torch.full((N, 1), r1), t), dim=1).to(device)
  r2_t = torch.cat((torch.full((N, 1), r2), t), dim=1).to(device)
  r = torch.linspace(r1, r2, N).reshape(-1, 1)
  r_t0 = torch.cat((r, torch.full((N, 1), 0)), dim=1).to(device)
  C_r1_t = model(r1_t)
  C_r2_t = model(r2_t)
  C_r_t0 = model(r_t0)
  loss_value = torch.mean((C_r1_t-1)**2+(C_r2_t-0)**2+(C_r_t0-0)**2)
  return loss_value