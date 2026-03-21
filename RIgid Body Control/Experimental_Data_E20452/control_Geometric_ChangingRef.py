import numpy as np
import csv
import time
from filterpy.kalman import KalmanFilter
from Orise_Twin_Rotor import Twin_Rotor
from time import sleep

class TwinRotorController:
    def __init__(self):
        # 1. Initialize hardware and stop motors
        self.t = Twin_Rotor()
        self.t.motors.stop()

        # 2. Physical Parameters
        self.m_a, self.r_a, self.L_a = 0.15, 0.04, 0.30
        self.m_r, self.r_r, self.L_r = 0.25, 0.22, 0.31

        self.I1 = (1/3) * self.m_a * (self.L_a**2) + 2 * self.m_r * (self.L_r**2)
        self.I2 = (1/2) * self.m_a * (self.r_a**2) + self.m_r * (self.r_r**2)
        self.I3 = self.I1
        
        # 3. Controller States
        self.eR_int = np.zeros(3)
        self.prev_yaw = 0.0
        self.prev_pitch = 0.0
        self.start_time = time.time()

        # Motor slew limiters
        self.max_slew_rate = 5000.0
        self.current_m0 = 0.0
        self.current_m1 = 0.0

        # 4. Logging setup
        self.log_file = open('rotor_log.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(['time', 'roll', 'pitch', 'yaw', 'rp_rpm', 'ry_rpm', 'err_p', 'err_y', 'mp_rpm', 'my_rpm'])

        self.kf = self.setup_ahrs_kalman()
        print("System Ready. Logging to rotor_log.csv")

    def get_reference(self, t, dt, pitch_mode='sin', yaw_mode='continuous_var'):
        """
        Define dynamic setpoints and their derivatives based on selected modes.
        
        Pitch modes: 'fixed', 'step', 'sin', 'sin2'
        Yaw modes: 'fixed', 'step', 'continuous_var'
        """

        if pitch_mode == 'fixed':
            ref_p = np.radians(-25.0)
            ref_p_dot = 0.0
            ref_p_ddot = 0.0

        elif pitch_mode == 'step':
            step_time = 10.0
            if t < step_time:
                ref_p = np.radians(-20.0)
            else:
                ref_p = np.radians(-10.0)
            ref_p_dot = 0.0
            ref_p_ddot = 0.0

        elif pitch_mode == 'sin':
            freq = 1/10
            omega = 2.0 * np.pi * freq  
            A = np.radians(10.0) 
            offset = np.radians(-25.0)  

            ref_p = A * np.sin(omega * t) + offset
            ref_p_dot = A * omega * np.cos(omega * t)
            ref_p_ddot = -A * (omega**2) * np.sin(omega * t)

        elif pitch_mode == 'sin2':
            f1, f2 = 0.1, 0.4
            w1, w2 = 2.0 * np.pi * f1, 2.0 * np.pi * f2
            A1, A2 = np.radians(10.0), np.radians(5.0)
            offset = np.radians(-25.0)

            ref_p = A1 * np.sin(w1 * t) + A2 * np.sin(w2 * t) + offset
            ref_p_dot = A1 * w1 * np.cos(w1 * t) + A2 * w2 * np.cos(w2 * t)
            ref_p_ddot = -A1 * (w1**2) * np.sin(w1 * t) - A2 * (w2**2) * np.sin(w2 * t)

        else:
            ref_p, ref_p_dot, ref_p_ddot = 0.0, 0.0, 0.0


        if yaw_mode == 'fixed':
            ref_y = np.radians(45.0)
            ref_y_dot = 0.0
            ref_y_ddot = 0.0

        elif yaw_mode == 'step':
            step_time = 10.0
            if t < step_time:
                ref_y = np.radians(45.0)
            else:
                ref_y = np.radians(90.0) 
            ref_y_dot = 0.0
            ref_y_ddot = 0.0

        elif yaw_mode == 'continuous_var':

            # base_speed = np.radians(10.0) # Average 15 deg/s
            # freq = 0.1
            # omega = 2.0 * np.pi * freq
            # # A = np.radians(10.0) # Speed varies +/- 10 deg/s
            # ref_y_dot = 0.5
            # ref_y_ddot = 0
            # ref_y = ref_y_dot * t

            # ref_y = base_speed * t - (A / omega) * np.cos(omega * t) + (A / omega) 
            # ref_y_dot = base_speed + A * np.sin(omega * t)
            # ref_y_ddot = A * omega * np.cos(omega * t)
             # Sin Yaw

            ref_y = np.radians(0.0)
            ref_y_dot = np.radians(5.0)
            ref_y_ddot = 0.0
            ref_y = ref_y_dot * t



        else:
            ref_y, ref_y_dot, ref_y_ddot = 0.0, 0.0, 0.0


        return ref_p, ref_p_dot, ref_p_ddot, ref_y, ref_y_dot, ref_y_ddot

    def setup_ahrs_kalman(self):
        kf = KalmanFilter(dim_x=6, dim_z=4)
        kf.x = np.zeros(6) 
        kf.F = np.eye(6) 
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]
        ])
        kf.P *= 10.0      
        kf.R = np.diag([0.1, 0.1, 10.0, 0.01]) 
        kf.Q = np.eye(6) * 0.01 
        return kf

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_ahrs(self, dt):
        acc = self.t.imu.acceleration
        mag = self.t.imu.magnetic
        enc1 = self.t.encoder.encoder1
        
        z_roll = np.arctan2(acc[1], acc[2])
        z_pitch = np.arctan2(acc[0], acc[2])
        z_yaw_mag = np.arctan2(mag[1], mag[0])
        z_yaw_enc = self.normalize_angle((enc1 / 406.0) * (2.0 * np.pi))
        
        self.kf.F[0, 3] = self.kf.F[1, 4] = self.kf.F[2, 5] = dt
        self.kf.predict()

        z = np.array([z_roll, z_pitch, z_yaw_mag, z_yaw_enc])
        y = z - np.dot(self.kf.H, self.kf.x)
        y[2:] = [self.normalize_angle(val) for val in y[2:]]

        S = np.dot(self.kf.H, np.dot(self.kf.P, self.kf.H.T)) + self.kf.R
        K = np.dot(self.kf.P, np.dot(self.kf.H.T, np.linalg.inv(S)))
        
        self.kf.x += np.dot(K, y)
        self.kf.P = np.dot(np.eye(6) - np.dot(K, self.kf.H), self.kf.P)
        return self.kf.x[0], self.kf.x[1], self.kf.x[2]

    def control(self, roll, pitch, yaw, dt, t_elapsed):
        if dt <= 0.0001: dt = 0.01
        
        # Dynamic Setpoints
        ref_p, ref_p_dot, ref_p_ddot, ref_y, ref_y_dot, ref_y_ddot = self.get_reference(
            t_elapsed, dt, 
            pitch_mode='sin2', 
            yaw_mode='continuous_var'
        )

        d_phi = self.normalize_angle(pitch - self.prev_pitch) / dt
        d_theta = self.normalize_angle(yaw - self.prev_yaw) / dt
        self.prev_pitch, self.prev_yaw = pitch, yaw

        # Physical Constants & Matrices
        J = np.diag([self.I1, self.I3, self.I2]) # Using your I mapping
        K_mat = np.diag([J[1,1]+J[2,2]-J[0,0], J[0,0]+J[2,2]-J[1,1], J[0,0]+J[1,1]-J[2,2]])


        R = np.array([
            [np.cos(yaw), -np.cos(pitch)*np.sin(yaw),  np.sin(pitch)*np.sin(yaw)],
            [np.sin(yaw),  np.cos(pitch)*np.cos(yaw), -np.sin(pitch)*np.cos(yaw)],
            [0,            np.sin(pitch),             np.cos(pitch)]
        ])
        Omega = np.array([d_phi, d_theta*np.sin(pitch), d_theta*np.cos(pitch)])
        omega = R @ Omega
        Pi = R @ J @ Omega

        Rr = np.array([
            [np.cos(ref_y), -np.cos(ref_p)*np.sin(ref_y),  np.sin(ref_p)*np.sin(ref_y)],
            [np.sin(ref_y),  np.cos(ref_p)*np.cos(ref_y), -np.sin(ref_p)*np.cos(ref_y)],
            [0,                np.sin(ref_p),                 np.cos(ref_p)]
        ])
        
        Omega_r = np.array([ref_p_dot, ref_y_dot*np.sin(ref_p), ref_y_dot*np.cos(ref_p)])
        Pi_dot_r = J @ np.array([ref_p_ddot, ref_y_ddot*np.sin(ref_p), ref_y_ddot*np.cos(ref_p)])
        pi_r = Rr @ J @ Omega_r


        # Errors
        Re = Rr @ R.T
        Re_body = R.T @ Rr
        # pi_e = (Re.T @ (Rr @ J @ np.zeros(3))) - Pi # Simplified for static-target derivatives
        pi_e = (Re.T @ pi_r) - Pi
        eR_hat = 0.5 * (Re_body @ K_mat - K_mat @ Re_body.T)
        eR = np.array([eR_hat[2,1], eR_hat[0,2], eR_hat[1,0]])

        Omega_r_body = Re_body @ Omega_r
        pi_e = (J @ Omega_r_body) - (J @ Omega)
        feedforward = (Re_body @ Pi_dot_r) + np.cross(Omega, J @ Omega_r_body)

        self.eR_int = np.clip(self.eR_int + eR * dt, -5.0, 5.0) 

        Kp = np.diag([10.0, 0.0, 45.0])
        Kd = np.diag([8.0, 0.0, 12.0])
        Ki = np.diag([5.0, 0.0, 3.0])
        # tau_u = (Kp @ eR) + (Kd @ pi_e) + (Ki @ self.eR_int)
        # tau_u = (Rr @ Pi_dot_r + np.cross(omega, Re.T @ pi_r)) + (Kp @ eR) + (Kd @ pi_e) + (Ki @ self.eR_int)
        # Tu = R.T @ tau_u

        Tu = feedforward + (Kp @ eR) + (Kd @ pi_e) + (Ki @ self.eR_int)

        # Allocation & Thrust mapping
        _A = np.array([[1.0, -0.0], [0.0, -1.0]]) # Simplified allocation for alpha=0, beta=90
        u, _, _, _ = np.linalg.lstsq(_A, np.array([Tu[0], Tu[2]]), rcond=None)
        
        def thrust_to_rpm(u_val):
            # The threshold where we switch from Square Root to Linear mapping
            deadband = 0.05 
            
            if abs(u_val) < deadband:
                # Linear mapping near zero: prevents infinite gain/chattering
                linear_slope = 2828.0 * np.sqrt(deadband) / deadband
                return np.sign(u_val) * linear_slope * abs(u_val)
            else:
                # Standard quadratic aerodynamic mapping
                return np.sign(u_val) * 2828.0 * np.sqrt(abs(u_val))
            
        target_m0 = thrust_to_rpm(u[0])
        target_m1 = thrust_to_rpm(u[1])

        # Slew
        def apply_ramp(curr, tar, delta_t):
            step = self.max_slew_rate * delta_t
            return curr + np.clip(tar - curr, -step, step)

        self.current_m0 = apply_ramp(self.current_m0, target_m0, dt)
        self.current_m1 = apply_ramp(self.current_m1, target_m1, dt)

        m0_speed = np.clip(self.current_m0, -2000, 2000)
        m1_speed = np.clip(-self.current_m1, -2000, 2000)

        err_p_deg = np.degrees(ref_p - pitch)
        err_y_deg = np.degrees(ref_y - yaw)

        return m0_speed, m1_speed, err_p_deg, err_y_deg, ref_p, ref_y

    def run(self):
        try:
            print("Running... Data logging to CSV active.")
            while True:
                dt = self.t.update_readings()
                t_now = time.time() - self.start_time
                r, p, y = self.get_ahrs(dt)
                
                mp, my, ep, ey, rp, ry  = self.control(r, p, y, dt, t_now)
                
                self.t.motors.set_speed_M0(-1 * my)
                # self.t.motors.set_speed_M0(0)
                self.t.motors.set_speed_M1(mp) 
                # self.t.motors.set_speed_M1(0) 

                # Log to CSV
                self.csv_writer.writerow([round(t_now, 3), round(r, 4), round(p, 4), round(y, 4), 
                                           round(rp, 2), round(ry, 2), round(ep, 2), round(ey, 2), round(mp,1), round(my, 1)])
                
                print(f"T: {t_now:.1f} | P: {np.degrees(p):.1f} | Y: {np.degrees(y):.1f} | R: {np.degrees(rp):.2f}, {np.degrees(ry):.2f} | E: {ep:.2f}, {ey:.2f}| M: {mp:.2f}, {my:.2f}", end="\r")
                sleep(0.01)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.t.motors.stop()
            self.log_file.close()

if __name__ == "__main__":
    controller = TwinRotorController()
    controller.run()