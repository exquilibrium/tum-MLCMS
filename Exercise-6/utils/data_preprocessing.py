import numpy as np
import matplotlib.pyplot as plt
import os


class Trajectories:
    def __init__(self):
        """
        Class to store and analyze pedestrian trajectories.

        Attributes:
            pedestrians (dict): Dictionary to store trajectories for each pedestrian.
        """
        self.pedestrians = {}

    def add_frame(self, id_frame, id, x, y):
        """
        Add a frame to the trajectory of a specific pedestrian.

        Args:
            id_frame (int): Frame ID.
            id (int): Pedestrian ID.
            x (float): X-coordinate.
            y (float): Y-coordinate.

        Returns:
            None
        """
        id_str = str(id)
        if id_str not in self.pedestrians:
            self.pedestrians[id_str]=[]
        self.pedestrians[id_str].append({'frame': id_frame, 'x':x, 'y':y})

    def print_trajectories(self):
        """
        Print the length of each pedestrian's trajectory.

        Returns:
            None
        """
        for key, value in self.pedestrians.items():
            print(key, "value", len(value))
    
    def calculate_speed(self, windows_size_x = 1, windows_size_y = 8, x_new = False):
        """
        Calculate X and Y velocities for each frame in the trajectories.

        Args:
            windows_size_x (int): Size of the window for X-direction velocity calculation.
            windows_size_y (int): Size of the window for Y-direction velocity calculation.

        Returns:
            None
        """
        assert windows_size_x > 0
        assert windows_size_y > 0
        for id, value in self.pedestrians.items():
            if(x_new):
                X = np.array([item['x_new'] for item in value])
            else: 
                X = np.array([item['x'] for item in value])
            Y = np.array([item['y'] for item in value])
            # Calculate velocities using the sliding window
            for i in range(windows_size_x, len(X) - windows_size_x):
                window_X = X[i - windows_size_x:i + windows_size_x+1]  # Extract the current frame and 8 frames before and after
                x_velocity = np.mean(np.diff(window_X)*16)
                value[i]['v'] = x_velocity

            for i in range(windows_size_y, len(Y) - windows_size_y):
                window_Y = Y[i - windows_size_y:i + windows_size_y+1]
                y_velocity = np.mean(np.diff(window_Y)*16)
                value[i]['u'] = y_velocity
        

    def calculate_speed_only_plot(self, file_path="", plot_number = 5):
        for id, value in self.pedestrians.items():
            if(plot_number > 0):
                X = np.array([item['x'] for item in value])
                Y = np.array([item['y'] for item in value])
                frame_id = np.array([item['frame'] for item in value])
                
                
                plt.figure(dpi=120)
                plt.title(f"X-Direction Speed of Pedestrian ID: {id} in the Dataset {os.path.basename(file_path).split('.')[0]}")
                plt.xlabel('Frame (1/16 s)')
                plt.ylabel('X-Direction Speed (cm/s)')
                plt.grid(True)
                
                
                windows_size = 1
                x_velocities = []
                y_velocities = []
                # Calculate velocities using the sliding window
                for i in range(windows_size, len(X) - windows_size):
                    window_X = X[i - windows_size:i + windows_size+1]  # Extract the current frame and 8 frames before and after
                    window_Y = Y[i - windows_size:i + windows_size+1]
                    x_velocity = np.mean(np.diff(window_X)*16)
                    y_velocity = np.mean(np.diff(window_Y)*16)    
                    x_velocities.append(x_velocity)
                    y_velocities.append(y_velocity)
                plt.plot(frame_id[windows_size: - windows_size], x_velocities, label = f'Windows Size {windows_size}')
                
                windows_size = 8
                x_velocities = []
                y_velocities = []
                # Calculate velocities using the sliding window
                for i in range(windows_size, len(X) - windows_size):
                    window_X = X[i - windows_size:i + windows_size+1]  # Extract the current frame and 8 frames before and after
                    window_Y = Y[i - windows_size:i + windows_size+1]
                    x_velocity = np.mean(np.diff(window_X)*16)
                    y_velocity = np.mean(np.diff(window_Y)*16)    
                    x_velocities.append(x_velocity)
                    y_velocities.append(y_velocity)
                plt.plot(frame_id[windows_size: - windows_size], x_velocities, label = f'Windows Size {windows_size}')    
                
                
                windows_size = 16
                x_velocities = []
                y_velocities = []
                # Calculate velocities using the sliding window
                for i in range(windows_size, len(X) - windows_size):
                    window_X = X[i - windows_size:i + windows_size+1]  # Extract the current frame and 8 frames before and after
                    window_Y = Y[i - windows_size:i + windows_size+1]
                    x_velocity = np.mean(np.diff(window_X)*16)
                    y_velocity = np.mean(np.diff(window_Y)*16)    
                    x_velocities.append(x_velocity)
                    y_velocities.append(y_velocity)     
                plt.plot(frame_id[windows_size: - windows_size], x_velocities, label = f'Windows Size {windows_size}')
                
                
                plt.legend()
                plt.show()
                plot_number -= 1
                
    def path_curve_fitting(self, degree):
        """
        Fit a polynomial curve to the X-coordinate of pedestrian trajectories.

        Args:
            degree (int): Degree of the polynomial.

        Returns:
            None
        """
        assert degree > 0
        for id, value in self.pedestrians.items():
            
            X = np.array([item['x'] for item in value])

            coefficients = np.polyfit(np.arange(len(X)), X, degree)
            poly_function = np.poly1d(coefficients)
            X_fitting = poly_function(np.arange(len(X)))
                
            for index, _ in enumerate(value):
                value[index]['x_new'] = X_fitting[index]

    def path_curve_fitting_only_plot(self, degree, file_path="", plot_number = 5):
        assert degree > 0
        for id, value in self.pedestrians.items():
            
            frame_id = np.array([item['frame'] for item in value])
            
            X = np.array([item['x'] for item in value])
            Y = np.array([item['y'] for item in value])

            coefficients = np.polyfit(np.arange(len(X)), X, degree)
            poly_function = np.poly1d(coefficients)
            X_fitting = poly_function(np.arange(len(X)))
            
            if(plot_number >0):
                plt.figure(dpi=120)
                plt.title(f"X-Direction Path of Pedestrian ID: {id} in the Dataset {os.path.basename(file_path).split('.')[0]}")
                plt.xlabel('Frame (1/16 s)')
                plt.ylabel('X Coordinate (cm)')
                plt.grid(True)
                plt.plot(frame_id, X, label = 'Original Data')
                plt.plot(frame_id, X_fitting, label = 'Fitted Curve', color= 'red')
                plt.legend()
                plt.show()

                plt.figure(dpi=120)
                plt.title(f"Y-Direction Path of Pedestrian ID: {id} in the Dataset {os.path.basename(file_path).split('.')[0]}")
                plt.xlabel('Frame (1/16 s)')
                plt.ylabel('Y Coordinate (cm)')
                plt.grid(True)
                plt.plot(frame_id, Y, label = 'Original Data')
                plt.legend()
                plt.show()
                plot_number -= 1
    
    def combine_only_plot(self, degree, file_path="", plot_number = 5):
        for id, value in self.pedestrians.items():
            if(plot_number > 0):
                X = np.array([item['x'] for item in value])
                Y = np.array([item['y'] for item in value])
                frame_id = np.array([item['frame'] for item in value])
                
                
                plt.figure(dpi=120)
                plt.title(f"X-Direction Speed of Pedestrian ID: {id} in the Dataset {os.path.basename(file_path).split('.')[0]}")
                plt.xlabel('Frame (1/16 s)')
                plt.ylabel('X-Direction Speed (cm/s)')
                
                
                windows_size = 1
                x_velocities = []
                # Calculate velocities using the sliding window
                for i in range(windows_size, len(X) - windows_size):
                    window_X = X[i - windows_size:i + windows_size+1]  # Extract the current frame and 8 frames before and after
                    x_velocity = np.mean(np.diff(window_X)*16)
                    x_velocities.append(x_velocity)
                plt.plot(frame_id[windows_size: - windows_size], x_velocities, label = f'Windows Size {windows_size}')

                coefficients = np.polyfit(np.arange(len(X)), X, degree)
                poly_function = np.poly1d(coefficients)
                X_fitting = poly_function(np.arange(len(X)))

                windows_size = 1
                x_velocities = []
                # Calculate velocities using the sliding window
                for i in range(windows_size, len(X_fitting) - windows_size):
                    window_X = X_fitting[i - windows_size:i + windows_size+1]  # Extract the current frame and 8 frames before and after
                    x_velocity = np.mean(np.diff(window_X)*16)
                    x_velocities.append(x_velocity)
                
                plt.plot(frame_id[windows_size: - windows_size], x_velocities, label = 'Fitted Curve',color = 'red')
            
                plt.grid(True)
                plt.legend()
                plt.show()
                plot_number -= 1

class Frames:
    def __init__(self, trajectories):
        """
        Class to store frames and analyze pedestrian interactions.

        Args:
            trajectories (Trajectories): Trajectories object.

        Attributes:
            frames (dict): Dictionary to store frames.
            trajectories (Trajectories): Trajectories object.
        """
        self.frames = {}
        self.trajectories = trajectories

    def add_frame(self, id_frame, id, x, y):
        """
        Add a frame to the frame data.

        Args:
            id_frame (int): Frame ID.
            id (int): Pedestrian ID.
            x (float): X-coordinate.
            y (float): Y-coordinate.

        Returns:
            None
        """
        id_frame_str = str(id_frame)
        id_str = str(id)
        if id_frame_str not in self.frames:
            self.frames[id_frame_str] = [] 
        
        v = self.trajectories.pedestrians[id_str][id_frame - self.trajectories.pedestrians[id_str][0]['frame']].get('u')
        if v is not None:
            u = self.trajectories.pedestrians[id_str][id_frame - self.trajectories.pedestrians[id_str][0]['frame']].get('v')
            self.frames[id_frame_str].append({'id': id, 'x': x, 'y': y, 'v': v, 'u': u, 'nearest_neigbors':[],
                                          'relative_position':[],'relative_speed':[],'distance':[],'nearest_neigbors_speed':[],"Sk": -1})

    def print_frames(self):
        """
        Print the number of entries in each frame.

        Returns:
            None
        """
        for key, value in self.frames.items():
            print(key, "value", len(value))
            
    def calculate_neighbors(self, k):
        """
        Calculate nearest neighbors and related information for each pedestrian.

        Args:
            k (int): Number of nearest neighbors to consider.

        Returns:
            None
        """
        for _, value in self.frames.items():
            if len(value) > k:
                points = np.array([(entry['x'], entry['y']) for entry in value])
                distances = np.sqrt(np.sum((points[:, np.newaxis] - points) ** 2, axis=-1))
                nearest_indices = np.argsort(distances, axis=1)[:, 1:k+1]
                points_speed = np.array([(entry['v'], entry['u']) for entry in value])
                
                for i in range(len(value)):
                    value[i]['nearest_neigbors'] = points[nearest_indices[i]]  
                    value[i]['distance'] = distances[i][nearest_indices[i]]  
                    value[i]['Sk'] = np.sum(value[i]['distance']) / k
                    
                    current_position = np.array([value[i]['x'],value[i]['y']])
                    value[i]['relative_position'] = value[i]['nearest_neigbors'] - current_position
                    
                    current_position_speed = np.array([value[i]['v'],value[i]['u']])
                    value[i]['nearest_neigbors_speed'] = points_speed[nearest_indices[i]]
                    value[i]['relative_speed'] = points_speed[nearest_indices[i]] - current_position_speed
