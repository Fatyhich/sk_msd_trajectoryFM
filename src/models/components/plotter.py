import matplotlib.pyplot as plt
import numpy as np

def create_2d_trajectory_plot(pred_traj, full_traj, noise_pred, full_time, title=""):
    """
    Создает 2D график траекторий с предсказаниями и неопределенностью
    
    Args:
        pred_traj: (N, 2) - предсказанная траектория
        full_traj: (N, 2) - ground truth траектория  
        noise_pred: (N, 2) - предсказания неопределенности
        full_time: (N,) - временная последовательность
        title: str - заголовок графика
    
    Returns:
        fig: matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Конвертируем в numpy если нужно
    if hasattr(pred_traj, 'cpu'):
        pred_traj = pred_traj.cpu().numpy()
    if hasattr(full_traj, 'cpu'):
        full_traj = full_traj.cpu().numpy()
    if hasattr(noise_pred, 'cpu'):
        noise_pred = noise_pred.cpu().numpy()
    if hasattr(full_time, 'cpu'):
        full_time = full_time.cpu().numpy()
    
    # Рисуем ground truth траекторию
    ax.plot(full_traj[:, 0], full_traj[:, 1], 'o-', 
            color='blue', linewidth=2, markersize=4, 
            label='Ground Truth', alpha=0.8)
    
    # Рисуем предсказанную траекторию
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], 's-', 
            color='orange', linewidth=2, markersize=4, 
            label='Prediction', alpha=0.8)
    
    # Рисуем начальную и конечную точки
    ax.scatter(full_traj[0, 0], full_traj[0, 1], 
               s=100, color='green', marker='o', 
               label='Start', zorder=5, edgecolors='black')
    ax.scatter(full_traj[-1, 0], full_traj[-1, 1], 
               s=100, color='red', marker='x', 
               label='End', zorder=5, linewidth=3)
    
    # Рисуем неопределенность как области вокруг предсказаний
    for i in range(len(pred_traj)):
        if i < len(noise_pred):
            # Создаем эллипс неопределенности
            from matplotlib.patches import Ellipse
            ellipse = Ellipse((pred_traj[i, 0], pred_traj[i, 1]), 
                            width=2*noise_pred[i, 0], 
                            height=2*noise_pred[i, 1],
                            alpha=0.3, color='gray', 
                            label='Uncertainty' if i == 0 else "")
            ax.add_patch(ellipse)
    
    # Настройки графика
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    
    # Автоматическое масштабирование
    all_pts = np.concatenate([full_traj, pred_traj])
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 0.5) * 1.15
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    ax.set_xlim(cx - span/2, cx + span/2)
    ax.set_ylim(cy - span/2, cy + span/2)
    
    plt.tight_layout()
    return fig