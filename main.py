# TODO: calc and error
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torch
import numpy as np


from PINN.model import SimpleNN
from PINN.data import DataGenerator
from analytical_solution.solution import ConcentrationSolution
from visualizers.visualizer import Visualizer


def main(args):
    model = SimpleNN()
    model.load_state_dict(torch.load(args.weights_path, weights_only=True))
    C = ConcentrationSolution(args.r1, args.r2, args.c1, args.c2, args.D)
    visualizer = Visualizer()
    data_generator = DataGenerator(r1=args.r1, r2=args.r2, c1=args.c1, c2=args.c2,
                                   t_start=args.t_start, t_finish=args.t_finish, D=args.D)

    _, _, _, _, R_T_scaled, _ = data_generator.get_data()
    R_T_points = data_generator.get_problem_grid(args.r1, args.r2, args.t_start, args.t_finish)

    R_T_scaled_tn = torch.Tensor(R_T_scaled)

    C_values = [C(r, t) for r, t in R_T_points]
    C_predicted = model(R_T_scaled_tn).detach().numpy().squeeze()*4

    # Interactive 3D plot
    fig, ax = visualizer.plot_3d_concentration(R_T_points[:, 0], R_T_points[:, 1], C_predicted, color="blue")
    fig, ax = visualizer.plot_3d_concentration(R_T_points[:, 0], R_T_points[:, 1], C_values, ax=ax, fig=fig, color="red")

    if args.visualization_path:
        model_name = os.path.splitext(os.path.basename(args.weights_path))[0]
        # I don't like the idea of +.png, maybe there is someting better?
        visualization_folder_path = os.path.join(args.visualization_path, model_name)
        visualization_path = os.path.join(visualization_folder_path, "3d.png")
        os.makedirs(visualization_folder_path, exist_ok=True)
        plt.savefig(visualization_path)

    if args.show_visualizations:
        plt.show()


    # Example of multiple subplots at a time
    fig, ax = visualizer.plot_2d_concentration_rows_colums(R_T_points[:, 0], R_T_points[:, 1], C_predicted)
    fig, ax = visualizer.plot_2d_concentration_rows_colums(R_T_points[:, 0], R_T_points[:, 1], np.array(C_values),
                                                           fig=fig, axs=ax, color="red")

    if args.visualization_path:
        model_name = os.path.splitext(os.path.basename(args.weights_path))[0]
        # I don't like the idea of +.png, maybe there is someting better?
        visualization_folder_path = os.path.join(args.visualization_path, model_name)
        visualization_path = os.path.join(visualization_folder_path, "2d.png")
        os.makedirs(visualization_folder_path, exist_ok=True)
        plt.savefig(visualization_path)
    
    if args.show_visualizations:
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--weights_path", help="", default="weights/trained_models/first_try.pth", )
    parser.add_argument("--visualization_path",
                        help="Place for storing visualizations, can be set to None to store None",
                        default="visualizations")
    parser.add_argument("--show_visualizations",
                        action="store_true",
                        help="Flag to show visualizations, literally plot.show()",
                        default=False)
    
    parser.add_argument("--r1", help="Inner radius of the pipe in mm", default=598)
    parser.add_argument("--r2", help="Outter radius of the pipe in mm", default=610)
    parser.add_argument("--t_start", help="", type=int, default=0)
    parser.add_argument("--t_finish", help="", type=int, default=25*1e3)
    parser.add_argument("--D", help="Diffusion coefficient", type=float, default=3.2*1e-3)
    parser.add_argument("--c0", type=float, help="Initial hydrogen concentration", default=0)
    parser.add_argument("--c1", help="Hydrogen concentration inside the pipe", default=4)
    parser.add_argument("--c2", help="Hydrogen concentration outside of the pipe", default=0)

    args = parser.parse_args()

    main(args)
