# import numpy as np
# import matplotlib.pyplot as plt
# from filterpy.kalman import UnscentedKalmanFilter as UKF
# from filterpy.kalman import MerweScaledSigmaPoints

# def state_transition_function(x, dt):
#     # 添加过程噪声
#     process_noise = np.random.normal(0, 0.1, size=x.shape)  # 标准差为0.1的高斯噪声
#     # 三变量状态转移（位置，速度，加速度）
#     return np.array([
#         x[0] + x[1] * dt + 0.5 * x[2] * dt**2 + process_noise[0],  # 位置更新
#         x[1] + x[2] * dt + process_noise[1],                       # 速度更新
#         x[2] + process_noise[2]                                     # 加速度更新（假设加速度变化小）
#     ])

# def measurement_function(x):
#     # 添加测量噪声
#     measurement_noise = np.random.normal(0, 10)  # 标准差为100的高斯噪声
#     return np.array([x[0] + measurement_noise])  # 仅测量位置

# if __name__ == '__main__':
#     # 定义UKF参数
#     dt = 0.1
#     points = MerweScaledSigmaPoints(n=3, alpha=0.1, beta=2., kappa=0)
#     ukf = UKF(dim_x=3, dim_z=1, dt = dt, fx=state_transition_function, hx=measurement_function, points=points)

#     # 初始化状态和协方差
#     ukf.x = np.array([0., 1., 0.1])  # 初始状态: 位置0，速度1，加速度0.1
#     ukf.P = np.eye(3) * 1           # 初始协方差
#     ukf.R = np.array([[1]])          # 测量噪声协方差
#     ukf.Q = np.eye(3) * 0.1        # 过程噪声协方差

#     # 模拟一些数据
#     n_steps = 50
#     measurements = []
#     true_positions = []
#     for t in range(n_steps):
#         true_position = 0.5 * 0.1 * t**2 + t  # 使用匀加速公式生成真实位置
#         true_positions.append(true_position)
#         measurement = true_position + np.random.normal(0, 10)  # 在测量中添加噪声
#         measurements.append(measurement)

#     # 存储估计结果
#     estimated_positions = []

#     # UKF预测和更新
#     for z in measurements:
#         ukf.predict(dt=dt)
#         ukf.update(z)
#         estimated_positions.append(ukf.x[0])  # 记录估计的位置

#     # 绘制结果
#     plt.figure(figsize=(10, 6))
#     plt.plot(true_positions, label='True Position', color='g')
#     plt.scatter(range(n_steps), measurements, label='Measurements with Noise', color='r', marker='x')
#     plt.plot(estimated_positions, label='UKF Estimated Position', color='b')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Position')
#     plt.title('UKF Position Estimation with Acceleration')
#     plt.legend()
#     plt.grid()

#     # 保存图像
#     output_path = '/root/autodl-tmp/metric3d/Metric3D/postprocess/ukf_position_estimation_with_acceleration.png'
#     plt.savefig(output_path)
#     plt.close()

#     print(f"UKF results with acceleration have been saved to {output_path}.")


import numpy as np
from collections import deque
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import matplotlib.pyplot as plt

class DepthKalmanFilter:
    def __init__(self, dt=1, process_noise_std=0.01, measurement_noise_std=4.0, initial_position=0.0, initial_velocity=0.0):
        """
        初始化深度卡尔曼滤波器。
        
        参数:
        - dt: 时间步长
        - process_noise_std: 过程噪声的标准差
        - measurement_noise_std: 测量噪声的标准差
        - initial_position: 初始位置
        - initial_velocity: 初始速度
        """
        self.dt = dt

        # 状态向量 [位置, 速度]
        self.x = np.array([initial_position, initial_velocity]).reshape((2, 1))

        # 状态转移矩阵 F
        self.F = np.array([
            [1, dt],
            [0, 1]
        ])

        # 测量矩阵 H (只测量位置)
        self.H = np.array([[1, 0]])

        # 初始协方差矩阵 P
        self.P = np.eye(2) * 1000.0  # 增大初始协方差，反映对初始状态的不确定性

        # 过程噪声协方差矩阵 Q
        self.Q = np.array([
            [process_noise_std**2, 0],
            [0, process_noise_std**2]
        ])

        # 测量噪声协方差矩阵 R
        self.R = np.array([[measurement_noise_std**2]])

        # 历史记录
        self.history = deque(maxlen=50)

    def predict(self):
        """
        卡尔曼滤波器的预测步骤。
        """
        # 状态预测
        self.x = self.F @ self.x

        # 协方差预测
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        卡尔曼滤波器的更新步骤。
        
        参数:
        - z: 当前测量值
        """
        z = np.array([[z]])  # 转换为列向量

        # 计算测量残差
        y = z - self.H @ self.x

        # 计算系统不确定性
        S = self.H @ self.P @ self.H.T + self.R

        # 计算卡尔曼增益
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 更新状态向量
        self.x = self.x + K @ y

        # 更新协方差矩阵
        I = np.eye(self.F.shape[0])
        self.P = (I - K @ self.H) @ self.P

        # 保存历史
        self.history.append(self.x.copy())

    def get_current_depth(self):
        """
        获取当前滤波后的深度值。
        """
        return self.x[0, 0]

    def get_history(self):
        """
        获取历史深度值。
        """
        return [state[0, 0] for state in self.history]
 #################### UKF  Begin#######################   
def fx(x, dt):
    """
    状态转移函数。
    x: 状态向量 [位置, 速度]
    dt: 时间步长
    """
    F = np.array([
        [1, dt],
        [0, 1]
    ])
    return F @ x

def hx(x):
    """
    测量函数。
    x: 状态向量 [位置, 速度]
    返回位置
    """
    return np.array([x[0]])

class DepthUKF:
    def __init__(self, dt=1.0, process_noise_std=0.01, measurement_noise_std=4.0, initial_position=0.0, initial_velocity=0.0):
        """
        初始化深度无迹卡尔曼滤波器。
        
        参数:
        - dt: 时间步长
        - process_noise_std: 过程噪声的标准差
        - measurement_noise_std: 测量噪声的标准差
        - initial_position: 初始位置
        - initial_velocity: 初始速度
        """
        # 定义 sigma 点
        points = MerweScaledSigmaPoints(n=2, alpha=0.1, beta=2., kappa=0)
        
        # 初始化 UKF
        self.ukf = UKF(dim_x=2, dim_z=1, fx=lambda x, dt: fx(x, dt), hx=hx, dt=dt, points=points)
        
        # 初始化状态
        self.ukf.x = np.array([initial_position, initial_velocity])
        
        # 初始化协方差
        self.ukf.P = np.eye(2) * 1000.0  # 大的初始不确定性
        
        # 过程噪声协方差矩阵 Q
        self.ukf.Q = np.eye(2) * (process_noise_std**2)
        
        # 测量噪声协方差矩阵 R
        self.ukf.R = np.array([[measurement_noise_std**2]])
        
        # 历史记录
        self.history = []
    
    def predict(self):
        """
        卡尔曼滤波器的预测步骤。
        """
        self.ukf.predict()
    
    def update(self, z):
        """
        卡尔曼滤波器的更新步骤。
        
        参数:
        - z: 当前测量值
        """
        self.ukf.update(z)
        self.history.append(self.ukf.x.copy())
    
    def get_current_depth(self):
        """
        获取当前滤波后的深度值。
        """
        return self.ukf.x[0]
    
    def get_history(self):
        """
        获取历史深度值。
        """
        return [state[0] for state in self.history]

################### UKF END ###############################
# 示例调用
if __name__ == "__main__":

    # # 模拟一些有噪声的深度数据
    # measurements = [
    #     205.68837, 204.15869, 201.76071, 201.15349, 195.99513, 
    #     204.04909, 199.85182, 203.07367, 202.56609, 199.32358, 
    #     203.79509, 204.58954, 206.19106, 202.6621, 201.05463, 
    #     195.91321, 208.1742, 202.28864, 202.42557, 204.49889
    # ]

    # # 初始化卡尔曼滤波器
    # kf = DepthKalmanFilter(
    #     dt=1, 
    #     process_noise_std=0.1, 
    #     measurement_noise_std=4.0, 
    #     initial_position=measurements[0], 
    #     initial_velocity=0.0
    # )

    # # 存储滤波结果
    # estimated_depths = []

    # for z in measurements:
    #     kf.predict()
    #     kf.update(z)
    #     estimated_depths.append(kf.get_current_depth())
    #     print(kf.get_current_depth())

    # 模拟一些有噪声的深度数据，包括突变
    measurements = [
        205.68837, 204.15869, 201.76071, 201.15349, 195.99513, 
        204.04909, 199.85182, 203.07367, 202.56609, 199.32358, 
        203.79509, 204.58954, 206.19106, 202.6621, 201.05463, 
        195.91321, 208.1742, 202.28864, 202.42557, 204.49889
    ]
    
    # 初始化 UKF
    ukf_filter = DepthUKF(
        dt=1.0, 
        process_noise_std=0.1, 
        measurement_noise_std=20.0,  # 增加测量噪声以减少对突变的敏感性
        initial_position=measurements[0], 
        initial_velocity=0.0
    )
    
    # 存储滤波结果
    estimated_depths = []
    
    for z in measurements:
        estimated_depth = ukf_filter.predict_and_update(z)
        estimated_depths.append(estimated_depth)
        print(f"测量值: {z}, 滤波后深度: {estimated_depth}")








    

