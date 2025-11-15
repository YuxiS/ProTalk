import numpy as np
import matplotlib.pyplot as plt

# 生成x值
x = np.linspace(0, 4 * np.pi, 400)

# 计算y值
y1 = np.sin(x)
y2 = 0.2 * np.sin(8 * x)+4
y3 = np.sin(x + np.pi / 3) + 4

# 创建图形和轴
plt.figure(figsize=(8, 6))

# 绘制三条曲线
plt.plot(x, y1, label=r'$s$', color='b', linewidth=2)
# plt.plot(x, y2, label=r'$m$', color='g', linewidth=2)
plt.plot(x, y3, label=r'$m$', color='r', linewidth=2)

# 添加标题和标签
# plt.title('Three Sine Waves', fontsize=16)
plt.xlabel('t', fontsize=12)
plt.ylabel('B', fontsize=12)

# 显示网格
# plt.grid(True)

# 显示图例
plt.legend()

# 展示图形
# plt.tight_layout()

# 保存图形为 PNG 文件
plt.savefig('sine_waves2.png', dpi=300)

# 展示图形
plt.show()
