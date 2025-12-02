import os
import re
import time
import matplotlib
# 设置无界面后端，防止在服务器上报错 'UserWarning: FigureCanvasAgg is non-interactive'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# ================= 配置区域 =================
# 日志根目录
LOG_ROOT = "/data/zhuldz/lunwen/rl/train/verl1/outputs/log"

# 输出目录 (图片和表格都存这)
OUTPUT_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/train_para"
OUTPUT_IMG = os.path.join(OUTPUT_DIR, "live_monitor.png")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "training_history.csv")

# 刷新间隔 (秒)
REFRESH_RATE = 30

# 想要画图的核心指标 (表格会记录所有指标，只有这些会画图)
METRICS_TO_PLOT = [
    ("Reward Score", "critic/score/mean", "tab:green"),
    ("Policy Loss", "actor/pg_loss", "tab:red"),
    ("KL Divergence", "actor/ppo_kl", "tab:orange"),
    ("Entropy", "actor/entropy", "tab:purple"),
    ("Gradient Norm", "actor/grad_norm", "tab:blue"),
    ("Clip Fraction", "actor/pg_clipfrac", "tab:brown"),
]
# ===========================================

def get_latest_log():
    """找到最新的日志文件"""
    if not os.path.exists(LOG_ROOT): return None
    subdirs = [os.path.join(LOG_ROOT, d) for d in os.listdir(LOG_ROOT) if os.path.isdir(os.path.join(LOG_ROOT, d))]
    if not subdirs: return None
    latest_dir = max(subdirs, key=os.path.getmtime)
    return os.path.join(latest_dir, "out.txt")

def parse_and_generate():
    """解析日志，生成图片和表格"""
    log_file = get_latest_log()
    if not log_file or not os.path.exists(log_file):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ 等待日志文件生成...")
        return

    data = []
    step_pattern = re.compile(r'step:(\d+)\s+-\s+(.*)')
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = step_pattern.search(line)
                if match:
                    step = int(match.group(1))
                    metrics_str = match.group(2)
                    
                    # 使用当前列表长度作为连续序列ID (防止 restart 后 step 重置)
                    row = {'_seq': len(data) + 1, 'step': step}
                    
                    # 提取该行所有指标
                    segments = metrics_str.split(' - ')
                    for seg in segments:
                        if ':' in seg:
                            k, v = seg.split(':', 1)
                            k = k.strip()
                            v = v.strip()
                            
                            # 过滤掉耗时统计 (timing_s/...) 如果不想看可以过滤
                            # if k.startswith('timing_'): continue
                            
                            # 清洗数据格式 np.float64(...)
                            v_clean = v.replace('np.float64(', '').replace(')', '').strip()
                            try:
                                row[k] = float(v_clean)
                            except:
                                pass # 无法转数字的跳过
                    
                    data.append(row)
    except Exception as e:
        print(f"解析日志出错: {e}")
        return

    if not data: 
        return

    # 转为 DataFrame
    df = pd.DataFrame(data)
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 1. 保存表格 (包含所有数据) ===
    try:
        # 将 '_seq' 放在第一列，方便查看
        cols = ['_seq', 'step'] + [c for c in df.columns if c not in ['_seq', 'step']]
        df[cols].to_csv(OUTPUT_CSV, index=False)
        # print(f"📊 表格已更新: {OUTPUT_CSV}")
    except Exception as e:
        print(f"保存表格失败: {e}")

    # === 2. 绘制图片 (只画核心指标) ===
    try:
        plt.figure(figsize=(12, 10))
        plt.suptitle(f"Training Monitor: {os.path.basename(os.path.dirname(log_file))}\nUpdated: {datetime.now().strftime('%H:%M:%S')}", fontsize=14)
        
        for i, (title, key, color) in enumerate(METRICS_TO_PLOT):
            plt.subplot(3, 2, i+1)
            
            if key in df.columns:
                # 绘制曲线
                plt.plot(df['_seq'], df[key], marker='o', markersize=3, linestyle='-', color=color, alpha=0.8, linewidth=1.5)
                
                # 标注最新值
                last_val = df[key].iloc[-1]
                plt.title(f"{title} (Current: {last_val:.4f})", fontsize=10, fontweight='bold')
                plt.grid(True, linestyle='--', alpha=0.4)
                
                if i >= 4: # 最后一行显示X轴
                    plt.xlabel("Steps (Continuous)")
            else:
                plt.text(0.5, 0.5, "Waiting for data...", ha='center', va='center', color='gray')
                plt.title(title)
                plt.grid(True, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_IMG, dpi=100)
        plt.close() # 释放内存
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 更新完成 | 📊 CSV: {os.path.basename(OUTPUT_CSV)} | 📈 Img: {os.path.basename(OUTPUT_IMG)}")
        
    except Exception as e:
        print(f"绘图失败: {e}")

if __name__ == "__main__":
    print("🚀 全能监控脚本已启动 (图片 + 表格)...")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print("💡 您可以在 VS Code 左侧文件列表中找到 .csv 和 .png 文件查看。")
    
    while True:
        parse_and_generate()
        time.sleep(REFRESH_RATE)