import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.randn(1000)

plt.hist(x, bins=30, edgecolor='white')
plt.title('Histogram')
plt.xlabel('x')
plt.ylabel('Test')
plt.tight_layout()
plt.show()
