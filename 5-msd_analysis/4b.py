import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit

# 设置全局字体为 Arial (Nature 级别标配)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


def calculate_msd(x, y, time_interval, pixel_size, max_lag_ratio=0.25):
    """
    计算均方位移 (MSD)
    max_lag_ratio: 最大时间滞后比例 (默认只取前 25% 保证统计学上的高信噪比)
    """
    N = len(x)
    max_lag = int(N * max_lag_ratio)

    # 将像素坐标转换为物理坐标 (微米)
    x_um = x * pixel_size
    y_um = y * pixel_size

    lags = np.arange(1, max_lag + 1)
    msd = np.zeros(max_lag)

    for i, lag in enumerate(lags):
        # 计算所有间隔为 lag 的位移平方和的平均值
        dx = x_um[lag:] - x_um[:-lag]
        dy = y_um[lag:] - y_um[:-lag]
        sq_dist = dx ** 2 + dy ** 2
        msd[i] = np.mean(sq_dist)

    time_lags = lags * time_interval
    return time_lags, msd


def anomalous_diffusion_2D(tau, D, alpha):
    """2D 异常扩散公式: MSD = 4 * D * tau^alpha"""
    return 4 * D * (tau ** alpha)


def track_and_plot_msd(tif_path, init_x, init_y, window_size, time_interval, pixel_size):
    """
    追踪单根微绒毛并绘制轨迹、MSD 曲线及动力学拟合
    """
    print(f"Loading TIFF: {tif_path}")
    try:
        img_stack = tiff.imread(tif_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if img_stack.ndim == 2:
        print("Error: The image is 2D. A time-series stack (3D) is required.")
        return

    num_frames = img_stack.shape[0]
    print(f"Total frames: {num_frames}")

    # 预分配坐标数组
    tracked_x = np.zeros(num_frames)
    tracked_y = np.zeros(num_frames)

    current_x = init_x
    current_y = init_y
    half_w = window_size // 2

    print("Tracking microvillus...")
    for t in range(num_frames):
        frame = img_stack[t]

        # 确定追踪窗口边界，防止越界
        y_min = max(0, int(current_y - half_w))
        y_max = min(frame.shape[0], int(current_y + half_w + 1))
        x_min = max(0, int(current_x - half_w))
        x_max = min(frame.shape[1], int(current_x + half_w + 1))

        # 提取局部窗口
        roi = frame[y_min:y_max, x_min:x_max].astype(float)

        # 减去局部背景 (以窗口内的最小值作为基准)，增强高亮结构的比重
        roi_bg_subtracted = roi - np.min(roi)

        if np.sum(roi_bg_subtracted) > 0:
            cy_local, cx_local = center_of_mass(roi_bg_subtracted)
            current_x = x_min + cx_local
            current_y = y_min + cy_local

        tracked_x[t] = current_x
        tracked_y[t] = current_y

    print("Tracking completed. Calculating MSD...")
    time_lags, msd = calculate_msd(tracked_x, tracked_y, time_interval, pixel_size, max_lag_ratio=0.25)

    # 拟合 MSD = 4 * D * tau^alpha
    print("Fitting anomalous diffusion model...")
    try:
        # 给定初始猜测值 p0=(D=0.01, alpha=1.0) 和参数边界 bounds
        popt, pcov = curve_fit(anomalous_diffusion_2D, time_lags, msd,
                               p0=[0.01, 1.0], bounds=([1e-6, 0.1], [10.0, 2.0]))
        D_fit, alpha_fit = popt
        msd_fit = anomalous_diffusion_2D(time_lags, D_fit, alpha_fit)
        fit_success = True
        print(f"Fit Results: D = {D_fit:.4e}, alpha = {alpha_fit:.2f}")
    except Exception as e:
        print(f"Fitting failed: {e}")
        fit_success = False

    # ==========================
    # 开始绘图 (1x2 排版)
    # ==========================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    # --- 图 1：运动轨迹 ---
    ax1 = axes[0]
    base_frame = img_stack[0]
    vmin, vmax = np.percentile(base_frame, 1), np.percentile(base_frame, 99.5)

    ax1.imshow(base_frame, cmap='magma', vmin=vmin, vmax=vmax)
    ax1.plot(tracked_x, tracked_y, color='white', linewidth=1.5, alpha=0.6)

    # 颜色映射代表时间流逝
    scatter = ax1.scatter(tracked_x, tracked_y, c=np.arange(num_frames), cmap='cool', s=15, zorder=5)

    # 标出起点和终点
    ax1.plot(tracked_x[0], tracked_y[0], marker='^', markeredgecolor='white', markerfacecolor='lime', markersize=9,
             zorder=6, label='Start')
    ax1.plot(tracked_x[-1], tracked_y[-1], marker='s', markeredgecolor='white', markerfacecolor='red', markersize=8,
             zorder=6, label='End')

    # 放大显示 ROI 区域
    padding = 20
    ax1.set_xlim(np.min(tracked_x) - padding, np.max(tracked_x) + padding)
    ax1.set_ylim(np.max(tracked_y) + padding, np.min(tracked_y) - padding)  # Y轴反向
    ax1.axis('off')
    ax1.legend(loc='upper right', fontsize=11, frameon=False, labelcolor='white')
    ax1.set_title("Trajectory on Frame 0", fontsize=15, fontweight='bold')

    # --- 图 2：MSD 曲线与拟合 ---
    ax2 = axes[1]
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)
    ax2.spines['top'].set_linewidth(1.5)
    ax2.spines['right'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.tick_params(width=1.5, labelsize=12, direction='in')

    # 绘制原始 MSD 数据 (散点+浅色连线)
    color_data = (92 / 255, 174 / 255, 177 / 255)
    ax2.plot(time_lags, msd, marker='o', color=color_data, markerfacecolor='none',
             markeredgewidth=1.5, markersize=6, linestyle='-', linewidth=1.5, alpha=0.7, label='Raw Data')

    # 绘制拟合曲线
    if fit_success:
        ax2.plot(time_lags, msd_fit, color='dimgray', linestyle='--', linewidth=2.5,
                 label='Fit: $MSD = 4D\\tau^\\alpha$')

        # 在图表中写出拟合得到的物理参数
        fit_text = f'$D$ = {D_fit:.3e} $\\mu$m$^2$/s$^\\alpha$\n$\\alpha$ = {alpha_fit:.2f}'

        # 根据 alpha 的值判断运动模式
        if alpha_fit < 0.9:
            mode = "(Sub-diffusion)"
        elif alpha_fit > 1.1:
            mode = "(Super-diffusion)"
        else:
            mode = "(Normal diffusion)"

        fit_text += f'\n{mode}'

        ax2.text(0.05, 0.95, fit_text, transform=ax2.transAxes, fontsize=12,
                 fontweight='bold', color='black', va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.8))

    ax2.set_xlabel(r'Time lag $\tau$ (s)', fontsize=15, fontweight='bold', color='black')
    ax2.set_ylabel(r'MSD ($\mu$m$^2$)', fontsize=15, fontweight='bold', color='black')

    # 调整 X 轴始终从 0 开始
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    ax2.legend(loc='lower right', fontsize=11, frameon=False, labelcolor='black')

    plt.tight_layout()
    output_pdf = "Microvillus_MSD_Fit_Analysis.pdf"
    plt.savefig(output_pdf, bbox_inches='tight', transparent=True)
    print(f"Figure saved to: {output_pdf}")
    plt.show()


# ==========================================
# 用户自定义参数区
# ==========================================
if __name__ == "__main__":
    FILE_TIFF = "D2-SIM.tif"

    # 1. 初始坐标: 请在 ImageJ/Fiji 中找到那根微绒毛在第 0 帧的大致中心坐标 (x, y)
    INITIAL_X = 250  # 替换为实际的 X 坐标
    INITIAL_Y = 250  # 替换为实际的 Y 坐标

    # 2. 追踪窗口大小 (像素): 必须能完全包住微绒毛的横截面
    WINDOW_SIZE = 15

    # 3. 实验参数
    MY_TIME_INTERVAL = 1.02  # 单位: s
    MY_PIXEL_SIZE = 0.03  # 单位: um/px

    track_and_plot_msd(
        tif_path=FILE_TIFF,
        init_x=INITIAL_X,
        init_y=INITIAL_Y,
        window_size=WINDOW_SIZE,
        time_interval=MY_TIME_INTERVAL,
        pixel_size=MY_PIXEL_SIZE
    )