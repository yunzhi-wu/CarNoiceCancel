import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Car interior dimensions (simplified)
car_length = 4.5  # meters
car_width = 1.8  # meters
car_height = 1.5  # meters

# Positions (x, y, z) of engine and passengers
engine_pos = np.array([0.5, car_width / 2, 0.5])  # Front of the car
passenger_positions = {
    "driver": np.array([1.5, 0.5, 1.0]),
    "front_passenger": np.array([1.5, 1.3, 1.0]),
    "rear_left": np.array([3.0, 0.5, 1.0]),
    "rear_right": np.array([3.0, 1.3, 1.0])
}

# Positions of speakers (assumed to be embedded in doors at height 0.5m)
speaker_positions = {
    "front_left": np.array([3.0, 0.0, 0.5]),
    "front_right": np.array([3.0, car_width, 0.5]),
    "rear_left": np.array([1.5, 0.0, 0.5]),
    "rear_right": np.array([1.5, car_width, 0.5])
}


# Function to calculate distance between two points in 3D space
def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


# Calculate sound pressure level at each passenger position
def calculate_spl_at_position(source_pos, target_pos, signal):
    distance = calculate_distance(source_pos, target_pos)
    attenuation = 1 / (distance ** 2)  # Simplified inverse-square law
    return signal * attenuation


# Function to plot the car interior with positions
def plot_car_interior(engine_pos, passenger_positions, speaker_positions, car_length, car_width, car_height):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot car as a box
    car_box = np.array([
        [0, 0, 0],
        [car_length, 0, 0],
        [car_length, car_width, 0],
        [0, car_width, 0],
        [0, 0, car_height],
        [car_length, 0, car_height],
        [car_length, car_width, car_height],
        [0, car_width, car_height]
    ])

    edges = [
        [car_box[0], car_box[1]], [car_box[1], car_box[2]], [car_box[2], car_box[3]], [car_box[3], car_box[0]],  # Bottom edges
        [car_box[4], car_box[5]], [car_box[5], car_box[6]], [car_box[6], car_box[7]], [car_box[7], car_box[4]],  # Top edges
        [car_box[0], car_box[4]], [car_box[1], car_box[5]], [car_box[2], car_box[6]], [car_box[3], car_box[7]]   # Side edges
    ]

    for edge in edges:
        ax.plot3D(*zip(*edge), color='black')

    # Plot engine position
    ax.scatter(*engine_pos, color='red', label='Engine')

    # Plot passenger positions
    for passenger, pos in passenger_positions.items():
        ax.scatter(*pos, label=passenger)

    # Plot speaker positions
    for speaker, pos in speaker_positions.items():
        ax.scatter(*pos, marker='^', label=speaker)

    # Set plot limits and labels
    ax.set_xlim(0, car_length)
    ax.set_ylim(0, car_width)
    ax.set_zlim(0, car_height)
    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Width (m)')
    ax.set_zlabel('Height (m)')
    ax.legend()

    # Set aspect ratio to ensure rectangular box
    ax.set_box_aspect([car_length, car_width, car_height])  # Aspect ratio is 1:1:1 in data space

    return ax


# Function to plot sine waves from engine to each passenger
def plot_sine_waves(ax, engine_pos, passenger_positions, noise_freq, speed_of_sound=343.0, point_spacing=0.01):
    for passenger, pos in passenger_positions.items():
        distance = calculate_distance(engine_pos, pos)
        num_points = int(distance / point_spacing)  # Calculate number of points based on distance
        t = np.linspace(0, distance / speed_of_sound, num_points, endpoint=False)
        x_vals = np.linspace(engine_pos[0], pos[0], num_points)
        y_vals = np.linspace(engine_pos[1], pos[1], num_points)
        z_vals = np.linspace(engine_pos[2], pos[2], num_points)
        wave = 0.1 * np.sin(2 * np.pi * noise_freq * t)  # Adjusted sine wave based on distance
        ax.plot3D(x_vals, y_vals, z_vals + wave, label=f'{passenger} wave')


# Main execution
if __name__ == "__main__":
    # Generate engine noise signal
    fs = 44100  # Sampling rate
    duration = 1  # seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    noise_freq = 1500  # Frequency in Hz
    engine_noise = np.sin(2 * np.pi * noise_freq * t)

    # Speed of sound in air at 20 degrees Celsius (m/s)
    speed_of_sound = 343.0

    # Plot the car interior
    ax = plot_car_interior(engine_pos, passenger_positions, speaker_positions, car_length, car_width, car_height)

    # Plot sine waves from engine to each passenger
    plot_sine_waves(ax, engine_pos, passenger_positions, noise_freq, speed_of_sound)

    plt.show()

    # Calculate and print SPL at each passenger position
    spl_results = {}
    for passenger, pos in passenger_positions.items():
        spl_results[passenger] = calculate_spl_at_position(engine_pos, pos, engine_noise)

    # For simplicity, assume ideal anti-noise signal is generated and applied
    anti_noise_signal = -engine_noise

    # Calculate residual noise at each passenger position
    residual_noise_results = {}
    for passenger, pos in passenger_positions.items():
        spl_noise = calculate_spl_at_position(engine_pos, pos, engine_noise)
        spl_anti_noise = calculate_spl_at_position(speaker_positions["front_left"], pos, anti_noise_signal)
        residual_noise = spl_noise + spl_anti_noise
        residual_noise_results[passenger] = np.mean(residual_noise ** 2)

    # Print results
    for passenger, residual_noise in residual_noise_results.items():
        print(f'Residual Noise for {passenger}: {residual_noise:.6f} (Mean Squared Amplitude)')
