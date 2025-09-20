"""
Basic Kalman Filter Implementation
================================

This module provides a simple implementation of a 1D Kalman filter
for tracking position and velocity from noisy position measurements.

Example: Tracking a moving object with constant velocity
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class KalmanFilter1D:
    """
    1D Kalman Filter for position/velocity tracking
    
    State vector: [position, velocity]
    Measurement: position only
    """
    
    def __init__(self, 
                 dt: float = 1.0,
                 process_noise: float = 0.1,
                 measurement_noise: float = 1.0,
                 initial_position: float = 0.0,
                 initial_velocity: float = 0.0,
                 initial_uncertainty: float = 1000.0):
        """
        Initialize the Kalman filter
        
        Args:
            dt: Time step
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
            initial_position: Initial position estimate
            initial_velocity: Initial velocity estimate
            initial_uncertainty: Initial uncertainty
        """
        self.dt = dt
        
        # State vector [position, velocity]
        self.x = np.array([[initial_position], 
                          [initial_velocity]], dtype=float)
        
        # State transition matrix (constant velocity model)
        self.F = np.array([[1.0, dt],
                          [0.0, 1.0]], dtype=float)
        
        # Measurement matrix (we only measure position)
        self.H = np.array([[1.0, 0.0]], dtype=float)
        
        # Process noise covariance matrix
        self.Q = np.array([[dt**4/4, dt**3/2],
                          [dt**3/2, dt**2]], dtype=float) * process_noise
        
        # Measurement noise covariance
        self.R = np.array([[measurement_noise]], dtype=float)
        
        # Error covariance matrix
        self.P = np.eye(2) * initial_uncertainty
        
        # Identity matrix
        self.I = np.eye(2)
        
    def predict(self, control_input: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step of Kalman filter
        
        Args:
            control_input: Optional control input (acceleration)
            
        Returns:
            Predicted state and covariance
        """
        # State prediction
        self.x = self.F @ self.x
        
        # Add control input if provided
        if control_input is not None:
            B = np.array([[0.5 * self.dt**2], 
                         [self.dt]], dtype=float)
            self.x += B * control_input
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x.copy(), self.P.copy()
    
    def update(self, measurement: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step of Kalman filter
        
        Args:
            measurement: Position measurement
            
        Returns:
            Updated state and covariance
        """
        # Convert measurement to numpy array
        z = np.array([[measurement]], dtype=float)
        
        # Innovation (residual)
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = self.I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        return self.x.copy(), self.P.copy()
    
    def get_state(self) -> dict:
        """Get current state estimates"""
        return {
            'position': self.x[0, 0],
            'velocity': self.x[1, 0],
            'position_uncertainty': np.sqrt(self.P[0, 0]),
            'velocity_uncertainty': np.sqrt(self.P[1, 1])
        }

def generate_true_trajectory(num_steps: int, dt: float, initial_pos: float = 0.0, 
                           velocity: float = 1.0, acceleration: float = 0.1) -> np.ndarray:
    """Generate true trajectory with constant acceleration"""
    t = np.arange(num_steps) * dt
    positions = initial_pos + velocity * t + 0.5 * acceleration * t**2
    return positions

def generate_measurements(true_positions: np.ndarray, noise_std: float = 1.0) -> np.ndarray:
    """Generate noisy measurements from true positions"""
    noise = np.random.normal(0, noise_std, len(true_positions))
    return true_positions + noise

def demo_kalman_filter():
    """Demonstrate the Kalman filter with a simple example"""
    print("Kalman Filter Demonstration")
    print("=" * 40)
    
    # Simulation parameters
    num_steps = 50
    dt = 1.0
    true_velocity = 1.0
    true_acceleration = 0.1
    measurement_noise_std = 2.0
    
    # Generate true trajectory
    true_positions = generate_true_trajectory(num_steps, dt, 0.0, true_velocity, true_acceleration)
    true_velocities = np.gradient(true_positions, dt)
    
    # Generate noisy measurements
    measurements = generate_measurements(true_positions, measurement_noise_std)
    
    # Initialize Kalman filter
    kf = KalmanFilter1D(
        dt=dt,
        process_noise=0.1,
        measurement_noise=measurement_noise_std**2,
        initial_position=0.0,
        initial_velocity=0.0,
        initial_uncertainty=100.0
    )
    
    # Track estimates
    estimated_positions = []
    estimated_velocities = []
    position_uncertainties = []
    
    # Run filter
    print(f"{'Step':<4} {'True Pos':<8} {'Measured':<8} {'Estimated':<10} {'True Vel':<8} {'Est Vel':<8}")
    print("-" * 60)
    
    for i in range(num_steps):
        # Predict step
        kf.predict()
        
        # Update step with measurement
        kf.update(measurements[i])
        
        # Store results
        state = kf.get_state()
        estimated_positions.append(state['position'])
        estimated_velocities.append(state['velocity'])
        position_uncertainties.append(state['position_uncertainty'])
        
        # Print every 5th step
        if i % 5 == 0:
            print(f"{i:<4} {true_positions[i]:<8.2f} {measurements[i]:<8.2f} "
                  f"{state['position']:<10.2f} {true_velocities[i]:<8.2f} {state['velocity']:<8.2f}")
    
    # Calculate errors
    position_errors = np.array(estimated_positions) - true_positions
    velocity_errors = np.array(estimated_velocities) - true_velocities
    
    print(f"\nResults:")
    print(f"Position RMSE: {np.sqrt(np.mean(position_errors**2)):.3f}")
    print(f"Velocity RMSE: {np.sqrt(np.mean(velocity_errors**2)):.3f}")
    
    # Plot results
    plot_results(true_positions, measurements, estimated_positions, 
                position_uncertainties, true_velocities, estimated_velocities)

def plot_results(true_pos, measurements, est_pos, uncertainties, true_vel, est_vel):
    """Plot the filtering results"""
    time_steps = range(len(true_pos))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Position plot
    ax1.plot(time_steps, true_pos, 'g-', label='True Position', linewidth=2)
    ax1.plot(time_steps, measurements, 'r.', label='Measurements', alpha=0.6)
    ax1.plot(time_steps, est_pos, 'b-', label='Kalman Estimate', linewidth=2)
    
    # Confidence bounds
    est_pos = np.array(est_pos)
    uncertainties = np.array(uncertainties)
    ax1.fill_between(time_steps, 
                     est_pos - 2*uncertainties, 
                     est_pos + 2*uncertainties, 
                     alpha=0.2, color='blue', label='±2σ Confidence')
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Position')
    ax1.set_title('Kalman Filter: Position Tracking')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Velocity plot
    ax2.plot(time_steps, true_vel, 'g-', label='True Velocity', linewidth=2)
    ax2.plot(time_steps, est_vel, 'b-', label='Estimated Velocity', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Velocity')
    ax2.set_title('Kalman Filter: Velocity Estimation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kalman_filter_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstration
    demo_kalman_filter()
    
    print("\nTry experimenting with different parameters:")
    print("- process_noise: How much the system can change")
    print("- measurement_noise: How noisy the measurements are")
    print("- initial_uncertainty: Initial confidence in estimates")
