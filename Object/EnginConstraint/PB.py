import numpy as np

# ==============================
# 核心函数定义
# ==============================

def payload_function(
    L: np.ndarray,
    k1: float = 1e6,
    k2: float = 1e4,
    k_m: float = 100,
    m0: float = 2000,
    C_max: float = 1.5,
    m_max: float = 5e-5,
    S_ref: float = 1e-20,
    alpha: int = 2,
    w1: float = 0.5,
    w2: float = 0.5,
    w3: float = 0
) -> np.ndarray:
    """
    计算特定臂长下的总损失值

    参数:
    - L     : 臂长数组 (单位:米)
    - k*    : 技术参数（详见问题描述）
    - C_max : 预算成本上限 (单位:欧元)
    - m_max : 运载火箭最大承载质量 (单位:kg)
    - S_ref : 参考灵敏度阈值
    - alpha : 灵敏度与臂长的依赖关系指数
    - w*    : 权重系数

    返回值:
    - 总损失值数组,形状与输入L相同
    """
    L=L/1e8
    # 计算成本 C(L)
    cost = k1*1e8 * L**2 + k2 * k_m * L
    
    # 计算卫星平台总质量 m(L)
    mass = k_m * L + m0*1e-8
    
    # 假设灵敏度模型 S(L) ∝ 1/L^alpha
    #S = 1 / (L**alpha)  # 简化的无量纲标度模型

    # 组合损失函数
    term_cost = w1 * (cost / C_max)          # 成本约束项
    term_mass = w2 * (mass / m_max)          # 质量约束项
    #term_sense = -w3 * (S / S_ref)           # 灵敏度奖励项（负号表提高灵敏度为优）

    total_loss = term_cost + term_mass #+ term_sense
    return total_loss


# ==============================
# 数值实现：一维离散化与二维转换
# ==============================

def payload_matrix(n_row,L):
    """
    生成损失函数离散化后的二维矩阵

    参数:
    - n_row     : 臂长的采样点数
    - L         : 臂长数组 (单位:米)

    返回:
    - L_1d : 一维臂长数组 (n,)
    - loss_matrix : 二维损失矩阵 (n, m)
    """
    # 生成一维臂长数组（均匀采样）
    #L_1d = np.linspace(L_min, L_max, n)
    
    # 计算对应的一维损失值
    loss_1d = payload_function(L)
    n= len(loss_1d)  # 一维数组长度
    # 将一维数组扩展为二维矩阵 (n, m)
    # 方法1：直接复制列（内存高效推荐）
    matrix = np.broadcast_to(
        loss_1d[:, np.newaxis],  # 转换为 (n,1)
        shape=(n, n_row)
    )
    
    # 方法2：等价但更显式的方式
    # loss_matrix = np.tile(loss_1d.reshape(-1,1), (1, m))
    
    return matrix