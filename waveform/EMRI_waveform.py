# ---- FEW 1.x / 2.x 兼容导入块 ----
import few

# 若没显式在外层设置后端，可强制允许 CPU（避免无 GPU 报错）
few.get_config_setter(reset=True).enable_backends("cuda12x", "cpu")

from few.trajectory.inspiral import EMRIInspiral
from few.waveform import (
    FastSchwarzschildEccentricFlux,
    SlowSchwarzschildEccentricFlux,
    Pn5AAKWaveform,
    GenerateEMRIWaveform,
)

from few.amplitude.romannet import RomanAmplitude
# v2 的 2D 振幅插值；起别名兼容旧代码里的 Interp2DAmplitude 名称
try:
    from few.amplitude.ampinterp2d import AmpInterpSchwarzEcc as Interp2DAmplitude
except ImportError:
    from few.amplitude.interp2dcubicspline import Interp2DAmplitude  # v1

# —— 工具函数：v2 把若干函数移到了 geodesic / mappings.pn
try:
    from few.utils.geodesic import (
        get_fundamental_frequencies,
        get_separatrix,
        get_kerr_geo_constants_of_motion,
        ELQ_to_pex,
    )
    from few.utils.utility import (
        get_mismatch,
        get_p_at_t,
        get_m2_at_t as get_mu_at_t,  # v2 改名：get_mu_at_t -> get_m2_at_t
    )
    from few.utils.mappings.pn import xI_to_Y, Y_to_xI
except ImportError:
    # FEW 1.x 老路径（仅当你降级到 v1.5.x 时会走到）
    from few.utils.utility import (
        get_mismatch,
        get_fundamental_frequencies,
        get_separatrix,
        get_mu_at_t,
        get_p_at_t,
        get_kerr_geo_constants_of_motion,
        ELQ_to_pex,
        xI_to_Y,
        Y_to_xI,
    )

from few.utils.ylm import GetYlms
# ---- 兼容导入块结束 ----





import few
from few.waveform import FastSchwarzschildEccentricFlux

# 让 FEW 先尝试 CUDA12x，不行就用 CPU
cfg = few.get_config_setter(reset=True)
cfg.enable_backends("cuda12x", "cpu")
cfg.finalize()

# ---- kwargs（注意：没有 use_gpu）----
inspiral_kwargs = {
    "DENSE_STEPPING": 0,
    "buffer_length": 1000,
}

amplitude_kwargs = {
    "buffer_length": 1000,
}

# v2 参数：include_minus_m（是否为给定的 m>=0 自动补 m<0）
Ylm_kwargs = {
    "include_minus_m": True,
}

sum_kwargs = {
    "pad_output": False,
}

# 显式指定后端（字符串即可）
backend = "cuda12x" if few.has_backend("cuda12x") else "cpu"

few_wf = FastSchwarzschildEccentricFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    force_backend=backend,
)

print("backend in use:", few_wf.backend_name)  # 预期为 'cuda12x' 或 'cpu'
