import matplotlib.pyplot as plt

# Define your data
data = [
    (0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (5, 6), (6, 7), (7, 10), (8, 10),
    (9, 12), (10, 15), (11, 14), (12, 12), (13, 18), (14, 7), (15, 5),
    (16, 3), (17, 1), (18, 3), (19, 0), (20, 0), (21, 0), (22, 1), (23, 1),
    (24, 1), (25, 0), (26, 0), (27, 1), (28, 0)
]

# Separate the data into values and frequencies
values, frequencies = zip(*data)

# Create a histogram with intervals every 5
bin_width = 5
bins = range(0, max(values) + bin_width, bin_width)
plt.hist(values, bins=bins, weights=frequencies, edgecolor='black', alpha=0.7)

# Set labels and title
plt.xlabel('Frequency of Failed Recoveries (Interval of 5)')
plt.ylabel('Number of Recurrent Cell Neurons')
plt.ylim([0, 20])

# Show the histogram
plt.show()

print('hi')
