import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# ==========================================
# 1. 顶刊全局排版设置 (Science Advances / Nature Style)
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['pdf.fonttype'] = 42

plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# ==========================================
# 2. 核心参数配置
# ==========================================
# 您的真实追踪数据文件 (建议使用 ImageJ/Fiji 的 TrackMate 插件导出)
# CSV 至少需要包含两列物理或像素坐标，例如: X_um, Y_um
TRACK_CSV = "Mito_Track.csv"

# 时间和空间参数
TIME_INTERVAL_S = 14.11  # 序贯双色成像的实际帧间距
PIXEL_SIZE_UM = 0.03  # 如果 CSV 里是像素坐标，用这个换算。如果是物理坐标则设为 1.0

# 颜色定义
COLOR_DATA = '#c281b1'  # 莫兰迪粉紫 (呼应线粒体通道)
COLOR_FIT = '#333333'  # 高级深灰 (拟合线)
COLOR_BOX = '#f8f9fa'  # 公式框背景色


def calculate_msd(x, y, time_interval, max_lag_ratio=0.25):
    """
    计算均方位移 (MSD)。
    为保证高信噪比，在生物物理分析中通常只取前 25% 的滞后时间数据。
    """
    N = len(x)
    max_lag = int(N * max_lag_ratio)

    lags = np.arange(1, max_lag + 1)
    msd = np.zeros(max_lag)

    for i, lag in enumerate(lags):
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        msd[i] = np.mean(dx ** 2 + dy ** 2)

    time_lags = lags * time_interval
    return time_lags, msd


def anomalous_diffusion_2D(tau, D, alpha):
    """2D 异常扩散物理模型: MSD = 4 * D * tau^alpha"""
    return 4 * D * (tau ** alpha)


def generate_mock_track(n_frames=50, dt=14.11, D=0.002, alpha=0.6):
    """模拟一段逼真的线粒体受限扩散轨迹 (Sub-diffusion)"""
    # 简单的分数阶布朗运动近似
    time_steps = np.arange(n_frames) * dt
    # 生成步长
    steps_x = np.random.normal(0, np.sqrt(2 * D * dt ** alpha), n_frames)
    steps_y = np.random.normal(0, np.sqrt(2 * D * dt ** alpha), n_frames)

    # 累加得到轨迹
    x_um = np.cumsum(steps_x)
    y_um = np.cumsum(steps_y)
    return x_um, y_um


def main():
    # 1. 数据加载与 MSD 计算
    if os.path.exists(TRACK_CSV):
        print(f"✅ 找到 {TRACK_CSV}，正在读取真实追踪数据...")
        df = pd.read_csv(TRACK_CSV)
        # 假设前两列是 X 和 Y 坐标
        x_um = df.iloc[:, 0].values * PIXEL_SIZE_UM
        y_um = df.iloc[:, 1].values * PIXEL_SIZE_UM
    else:
        print(f"⚠️ 未找到 {TRACK_CSV}，正在使用模拟的亚细胞动力学数据...")
        x_um, y_um = generate_mock_track(n_frames=40, dt=TIME_INTERVAL_S)

    # 提取前 25% 的数据点计算 MSD
    tau, msd = calculate_msd(x_um, y_um, TIME_INTERVAL_S, max_lag_ratio=0.25)

    # 2. 物理模型拟合
    print("正在进行异常扩散方程拟合...")
    try:
        # 初始猜测: D=0.01, alpha=1.0. 边界: D>0, 0<alpha<2
        popt, _ = curve_fit(anomalous_diffusion_2D, tau, msd,
                            p0=[0.01, 1.0], bounds=([1e-6, 0.1], [1.0, 2.0]))
        D_fit, alpha_fit = popt
        msd_fit = anomalous_diffusion_2D(tau, D_fit, alpha_fit)
        fit_success = True
    except Exception as e:
        print(f"❌ 拟合失败: {e}")
        fit_success = False

    # 3. 绘制高雅的理科图表
    fig, ax = plt.subplots(figsize=(4.5, 3.8), dpi=300)

    # 绘制真实 MSD 数据点 (空心小圆圈)
    ax.plot(tau, msd, marker='o', color=COLOR_DATA, markerfacecolor='none',
            markeredgewidth=1.8, markersize=6, linestyle='none', label='Experimental MSD', zorder=3)

    # 绘制拟合虚线
    if fit_success:
        ax.plot(tau, msd_fit, color=COLOR_FIT, linestyle='--', linewidth=2.0,
                label='Model Fit', zorder=2)

        # 判断运动模式
        if alpha_fit < 0.9:
            mode = "Sub-diffusion"
        elif alpha_fit > 1.1:
            mode = "Super-diffusion"
        else:
            mode = "Normal diffusion"

        # 在图表左上角添加参数文本框
        fit_text = (f"Model: $\\langle r^2 \\rangle = 4D\\tau^\\alpha$\n"
                    f"$D = {D_fit:.4f}$ $\\mu$m$^2$/s$^\\alpha$\n"
                    f"$\\alpha = {alpha_fit:.2f}$ ({mode})")

        ax.text(0.05, 0.95, fit_text, transform=ax.transAxes, fontsize=11,
                color=COLOR_FIT, va='top', ha='left', linespacing=1.6,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_BOX, edgecolor='#dddddd', alpha=0.8))

    # 4. 图表排版细节
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(axis='both', direction='in', top=True, right=True, length=5, labelsize=11)

    ax.set_xlabel(r'Time Lag $\tau$ (s)', fontsize=14, fontweight='bold', labelpad=6)
    ax.set_ylabel(r'MSD ($\mu$m$^2$)', fontsize=14, fontweight='bold', labelpad=6)

    # 强制坐标轴从 0 开始，展现数据的真实幅度
    ax.set_xlim(left=0, right=max(tau) * 1.05)
    ax.set_ylim(bottom=0, top=max(msd) * 1.15)

    # 图例
    ax.legend(frameon=False, fontsize=11, loc='lower right')

    plt.tight_layout()
    output_pdf = "Figure_5f_Mito_MSD_Dynamics.pdf"
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.savefig("Figure_5e_Mito_MSD_Dynamics.png", format='png', bbox_inches='tight', dpi=300)
    print(f"✨ MSD 动力学定量图渲染完成！已保存为: {output_pdf}")
    plt.show()


if __name__ == "__main__":
    main()