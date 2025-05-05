import numpy as np
import matplotlib.pyplot as plt

import numpy as np

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



def EC_l2(array, peak_position, width, n):
    """
    Generate a matrix where each row is a wave packet.

    Parameters:
        array (np.ndarray): Input 1D array.
        peak_position (float): 波包峰值的实际数值位置（如5.0表示在坐标轴上的5.0处）。
        width (int): The width of the wave packet.
        n (int): The number of rows in the output matrix.

    Returns:
        np.ndarray: A matrix with `n` rows, each containing the wave packet.
    """
    # Generate the wave packet
    wave_packet = EC_l(array, peak_position, width)
    
    # Create a matrix by tiling the wave packet
    wave_packet_matrix = np.tile(wave_packet, (n, 1))
    return wave_packet_matrix

__all__ = ['EC_l2']