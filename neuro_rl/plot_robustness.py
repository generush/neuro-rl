import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy import interpolate

DATA_PATH = '/home/gene/Desktop/frontiers2023/'
df = pd.read_csv(DATA_PATH + 'DIST-TERR.csv', index_col=0)

# Create a meshgrid for x and y values
x = df.columns.astype(float)
y = (df.index.astype(float)-5)/0.4
X, Y = np.meshgrid(x, y)

# Create a smooth representation of the data using interpolation
x_smooth = np.linspace(x.min(), x.max(), 100)
y_smooth = np.linspace(y.min(), y.max(), 100)
X_smooth, Y_smooth = np.meshgrid(x_smooth, y_smooth)
f = interpolate.interp2d(x, y, df.values/100, kind='linear')
Z_smooth = f(x_smooth, y_smooth)

# Create the surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

orig_map=plt.cm.get_cmap('plasma')
reversed_map = orig_map.reversed()

# Reverse the y-axis
ax.invert_yaxis()

surf = ax.plot_surface(X_smooth, Y_smooth, Z_smooth, cmap=reversed_map, vmin=0, vmax=1, linewidth=0, edgecolor='none', alpha=0.8)

# Specify azimuth and elevation angles (change these values as needed)
ax.view_init(azim=145, elev=30)

# Remove gridlines
ax.grid(False)
plt.rcParams['grid.linewidth'] = 0

# Add color bar
cbar = fig.colorbar(surf, shrink=0.75)

# Set labels for axes
ax.set_xlabel('Lateral Force')
ax.set_ylabel('Gait Cycle')
ax.set_zlabel('Recovery Rate')

# Save the plot as an SVG file
plt.savefig('3d_surface_plot.svg', format='svg', bbox_inches='tight')

# Show the plot
plt.show()

print("hi")
