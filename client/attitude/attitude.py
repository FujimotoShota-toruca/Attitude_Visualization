import numpy as np
import matplotlib.pyplot as plt
import time
import json
from scipy.integrate import solve_ivp

# クォータニオン微分を計算する関数
def quaternion_derivative(t, state, omega_target, Kp, Kd, inertia):
    q = state[:4]
    omega = state[4:7]
    
    # 現在のクォータニオンと目標クォータニオンの誤差を計算
    q_error = quaternion_error(q, q_target)

    # 現在の角速度と目標角速度の誤差を計算
    omega_error = omega_target - omega

    # トルクを計算
    torque = Kp * np.array([q_error[1], q_error[2], q_error[3]]) + Kd * omega_error

    # 角速度の更新（慣性モーメントを考慮）
    omega_dot = np.linalg.inv(inertia).dot(torque)

    # クォータニオンの微分
    q_dot = 0.5 * np.array([
        -omega[0] * q[1] - omega[1] * q[2] - omega[2] * q[3],
        omega[0] * q[0] + omega[2] * q[2] - omega[1] * q[3],
        omega[1] * q[0] - omega[2] * q[1] + omega[0] * q[3],
        omega[2] * q[0] + omega[1] * q[1] - omega[0] * q[2]
    ])
    
    return np.concatenate((q_dot, omega_dot))

# クォータニオンの誤差を計算する関数
def quaternion_error(q_current, q_target):
    q_conjugate = np.array([q_target[0], -q_target[1], -q_target[2], -q_target[3]])
    q_error = quaternion_multiply(q_conjugate, q_current)
    return q_error

# クォータニオンの積を計算する関数
def quaternion_multiply(q1, q2):
    return np.array([
        q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
        q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
        q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
        q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    ])

# 初期条件の設定
q_initial = np.array([1, 0.5, 0.3, 0.1])  # 初期クォータニオン
q_initial = q_initial / np.linalg.norm(q_initial)
omega_initial = np.array([0.3, 0.2, 0.1])  # 初期角速度
q_target = np.array([0, 0, 0, 1])  # 目標クォータニオン
omega_target = np.array([0.0, 0.0, 0.0])   # 目標角速度

# 慣性モーメント（対角行列の例）
inertia = np.diag([1.0, 1.0, 1.0])  # 単位行列として仮定

# 制御パラメータ
Kp = 0.2  # Pゲイン
Kd = 0.1  # Dゲイン

# 時間ステップと総時間を設定
dt_simulation = 0.01  # シミュレーションの刻み幅（秒）
dt_output = 1.0       # 標準出力の間隔（秒）
total_time = 100      # 総時間（秒）
t_span = (0, total_time)
initial_state = np.concatenate((q_initial, omega_initial))

# 現在の状態
current_state = initial_state
next_output_time = dt_output

# 結果を格納するためのリスト
quaternion_history = []
omega_history = []
torque_history = []
time_history = []

# シミュレーションのループ
for t in np.arange(0, total_time, dt_simulation):
    # 次のステートを計算
    next_state = solve_ivp(quaternion_derivative, (t, t + dt_simulation), current_state, args=(omega_target, Kp, Kd, inertia), t_eval=[t + dt_simulation])
    
    # 状態を更新
    current_state = next_state.y[:, -1]

    # 時間を記録
    time_history.append(t + dt_simulation)
    
    # クォータニオンと角速度を記録
    quaternion_history.append(current_state[:4])
    omega_history.append(current_state[4:7])

    # トルクを計算して記録
    q_error = quaternion_error(current_state[:4], q_target)
    torque = Kp * np.array([q_error[1], q_error[2], q_error[3]]) + Kd * (omega_target - current_state[4:7])
    torque_history.append(torque)

    # 出力の時間になった場合、値をJSON形式で出力
    if t + dt_simulation >= next_output_time:

        _quaternion = current_state[:4].tolist()
        _angular_velocity = current_state[4:7].tolist()
        _torque = torque.tolist()

        output_data ={
            "quaternion_i2b":{
                "w":_quaternion[0],
                "x":_quaternion[1],
                "y":_quaternion[2],
                "z":_quaternion[3]
            },
            "angular_velocity":{
                "x":_angular_velocity[0],
                "y":_angular_velocity[1],
                "z":_angular_velocity[2]
            },
            "torque":{
                "x":_torque[0],
                "y":_torque[1],
                "z":_torque[2]
            }
        }
        print(json.dumps(output_data))
        next_output_time += dt_output

    # リアルタイム感を持たせる
    #time.sleep(dt_simulation)

# 時系列データをプロット
quaternion_history = np.array(quaternion_history)
omega_history = np.array(omega_history)
time_history = np.array(time_history)

plt.figure(figsize=(12, 6))

# クォータニオン成分のプロット
plt.subplot(2, 1, 1)
plt.plot(time_history, quaternion_history[:, 0], label='q0')
plt.plot(time_history, quaternion_history[:, 1], label='q1')
plt.plot(time_history, quaternion_history[:, 2], label='q2')
plt.plot(time_history, quaternion_history[:, 3], label='q3')
plt.title('Quaternion Time Series with Feedback Control')
plt.xlabel('Time [s]')
plt.ylabel('Quaternion Components')
plt.legend()
plt.grid(True)

# 角速度のプロット
plt.subplot(2, 1, 2)
plt.plot(time_history, omega_history[:, 0], label='ωx')
plt.plot(time_history, omega_history[:, 1], label='ωy')
plt.plot(time_history, omega_history[:, 2], label='ωz')
plt.title('Angular Velocity Time Series')
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity [rad/s]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
