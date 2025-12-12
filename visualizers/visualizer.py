import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


class Visualizer:
    def __init__(self):
        pass

    def plot_3d_concentration(self, R, T, C_values, title="3D Hydrogen concentration", fig=None, ax=None, color="blue"):
        """
            Method returns figure and axis of 3D scatter plot for hydrogen concentration. 
        """
    
        if (fig is None) or (ax is None):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        ax.scatter(R, T, C_values, color=color)

        ax.set_xlabel('R (mm)')
        ax.set_ylabel('T (secs)')
        ax.set_zlabel('C (ppm)')

        ax.set_title(title)

        return fig, ax

    def plot_2d_concentration(self, R, T, C_values, N=10, color="blue") -> list:
        """
            Method return list of plots, one for each radius
        """

        ax_list = [] 

        R = R.reshape(N, N)
        T = T.reshape(N, N)
        C_values = C_values.reshape(N, N)

        for ind, _ in enumerate(C_values):
                r = R[0][ind]
                 
                fig, ax = plt.subplots()
                ax.plot(T, C_values[:, ind])
                ax.set_title(f"R = {r}")
                ax.set_xlabel('Time (secs)')
                ax.set_ylabel('Concentration (ppm)')
                ax.ticklabel_format(useOffset=False, style='plain')
                
                ax_list.append(ax)
        

        return ax_list           
    
    def plot_2d_concentration_rows_colums(self, R, T, C_values, num_rows=2, num_cols=5, N=10, color="blue", fig=None, axs=None) -> list:
        # Reshape the input arrays

        R = R.reshape(N, N)
        T = T.reshape(N, N)
        C_values = C_values.reshape(N, N)
        
        # Determine the number of rows and columns
        num_rows = 2
        num_cols = 5

        if (axs is None) or (fig is None):
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 6), sharex='all', sharey='all')
            axs = axs.flatten()

        # Generate plots for each radius
        for ind, ax in enumerate(axs):
            if ind < len(R):
                r = R[0][ind]
                ax.plot(T, C_values[:, ind], color=color)
                ax.set_title(f"R = {int(r)}")
                ax.set_xlabel('Time (secs)')
                ax.set_ylabel('Concentration (ppm)')
                ax.ticklabel_format(useOffset=False, style='plain')
            else:
                ax.axis('off')  # Turn off any unused subplots

        # Adjust layout to prevent overlap
        plt.tight_layout()

        return fig, axs   

    def plot_2d_points(self, R, T) -> tuple:

        fig, ax = plt.subplots()
        ax.scatter(R, T)
        ax.set_xlabel("R - thickness of the pipe (mm)")
        ax.set_ylabel("t - time (sec)")
        ax.set_title("2D Scatter Plot of Thickness vs. Time")
        ax.grid(True)  # optional
        
        return fig, ax

