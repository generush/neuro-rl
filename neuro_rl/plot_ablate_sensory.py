import matplotlib.pyplot as plt

# Data
labels = [r'$u$', r'$v$', r'$w$', r'$p$', r'$q$', r'$r$', r'$\cos(\phi)$', r'$\cos(\theta)$', r'$\cos(\psi)$', r'$u*$', r'$v*$', r'$r*$', 'joint pos', 'joint vel', 'height']
values = [0.59, 0.00, 0.81, 0.25, 1.00, 0.99, 0.99, 0.27, 1.00, 0.83, 1.00, 1.00, 0.02, 0.73, 0.99]

# Create bar chart
plt.figure(figsize=(10, 4))
plt.bar(labels, values, width=0.6)

# Title and labels
# plt.title('Bar Chart')
plt.ylim([0,1])
plt.xlabel('Sensory Neuron Ablated')
plt.ylabel('Recovery Rate')
plt.xticks(rotation=45, ha='right')  # Rotate x labels for better visibility
plt.tight_layout()

# To support LaTeX formatting in labels
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.show()

print('hi')
# Show the chart