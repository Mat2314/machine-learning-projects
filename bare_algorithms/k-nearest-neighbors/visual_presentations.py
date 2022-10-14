import matplotlib.pyplot as plt
from calculation_functions import get_neighbors
import random
from matplotlib.animation import FuncAnimation


class VisualPresentations:
    
    def _draw_line_between_points(self, point1: tuple, point2: tuple):
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        plt.plot(x_values, y_values, color="red", marker='o')

    def show_nearest_neighbors(self, new_point: tuple = None, find_neighbors_amount: int = 5):
        """
        Method prepares a dummy dataset full of random points of 2 classes - 0 and 1.
        
        """
        # Prepare dummy dataset and plot it
        # X | Y | Class
        
        dataset = [[x, random.randint(-10, 0), 0] for x in range(-10, 0)] + \
            [[x, random.randint(0, 10), 1] for x in range(0, 10)]
        
        # Prepare dots to be plotted on a chart
        for data in dataset:
            color = "blue" if data[2] == 0 else "green"
            plt.scatter(data[0], data[1], s=200, color=color, alpha=0.5)
        
        # If new point was not given use the fixed one
        if not new_point:
            new_point = [0,0]
        
        # Find closest neighbors
        neighbors = get_neighbors(dataset, new_point, find_neighbors_amount)
        
        for neighbor in neighbors:
            # Draw lines from the new point to nearest neighbors
            self._draw_line_between_points(new_point, neighbor)
        
        plt.show()

if __name__ == "__main__":
    vp = VisualPresentations()
    vp.show_nearest_neighbors(new_point=(8,-2), find_neighbors_amount=11)