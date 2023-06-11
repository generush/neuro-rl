
import matplotlib.pyplot as plt
import numpy as np

fig_tmp, ax_tmp = plt.subplots()

x, y = np.mgrid[0:2.5:1000j, -2.5:2.5:1000j]
vx, vy = np.cos(x - y), np.sin(x - y)
res = ax_tmp.streamplot(x.T, y.T, vx, vy, color='k')

lines = res.lines.get_paths()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Streamplot on XY plane (z=0)
for i, line in enumerate(lines):
    x = line.vertices.T[0]
    y = line.vertices.T[1]
    z = np.zeros_like(x)
    ax.plot(x, y, z, 'r')
    if i % 2 == 0:
        ax.quiver(x[0], y[0], z[0], x[1]-x[0], y[1]-y[0], z[1]-z[0], length=2, color='r')

# Streamplot on YZ plane (x=0)
for i, line in enumerate(lines):
    x = np.zeros_like(line.vertices.T[0])
    y = line.vertices.T[0]
    z = line.vertices.T[1]
    ax.plot(x, y, z, 'b')
    if i % 2 == 0:
        ax.quiver(x[0], y[0], z[0], x[1]-x[0], y[1]-y[0], z[1]-z[0], length=2, color='b')

# Streamplot on XZ plane (y=0)
for i, line in enumerate(lines):
    x = line.vertices.T[0]
    y = np.zeros_like(line.vertices.T[1])
    z = line.vertices.T[1]
    ax.plot(x, y, z, 'g')
    if i % 2 == 0:
        ax.quiver(x[0], y[0], z[0], x[1]-x[0], y[1]-y[0], z[1]-z[0], length=2, color='g')

plt.show()

