#二分法求零点
def Find_zero(f, x1, x2, e, iter):
    if f(x1) * f(x2) >= 0:
        return -1

    if f(x1) > 0:
        x1, x2 = x2, x1

    for _ in range(iter):
        mid = (x1 + x2) / 2
        val = f(mid)

        if abs(val) < e:
            return mid

        if val < 0:
            x1 = mid
        else:
            x2 = mid

    return (x1 + x2) / 2

#Secant method for zeropoint
def secant_method(f, x0, x1, tol=1e-5, max_iter=100):
    """使用割线法查找函数f(x)=0的零点。

    参数:
    f -- 目标函数
    x0, x1 -- 初始两个估计值
    tol -- 容忍误差，当函数值小于此值时停止迭代
    max_iter -- 最大迭代次数

    返回:
    零点的近似值
    """
    for i in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)
        
        # 计算割线的斜率
        if (fx1 - fx0) == 0:
            return -1  # 防止除以零
        slope = (x1 - x0) / (fx1 - fx0)

        # 更新x1和x0
        x0=x1
        x1=max(x1 - fx1 * slope,0.00001)#要求x>1
       
        # 检查是否达到容忍误差
        if abs(f(x1)) < tol:
            return x1
    return -1  # 如果没有在max_iter迭代次数内找到解，则返回None
