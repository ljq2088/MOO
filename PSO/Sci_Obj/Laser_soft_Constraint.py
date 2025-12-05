import numpy as np
lambda_best=1064.0
def EC_l(array, peak_position, width):
    """
    生成一个在指定数值位置具有峰值的高斯波包。

    参数:
        array (np.ndarray): 输入的1D坐标数组，表示实际数值位置（如时间、空间坐标）。
        peak_position (float): 波包峰值的实际数值位置（如5.0表示在坐标轴上的5.0处）。
        width (float): 高斯波包的标准差（宽度）。

    返回:
        np.ndarray: 高斯波包数组，形状与输入array一致。
    """
    # 根据实际坐标计算波包，不再依赖索引而是数值位置
    wave_packet = np.exp(-( (array - peak_position) ** 2 ) / (2 * width ** 2))
    return wave_packet
def LSC(paras):
    l=paras[1]
    return EC_l(l,lambda_best,100)
