import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from shapely import Polygon, Point
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.autolayout"] = True

plt.figure()
plt.title("The Environment")
# Draw the environment
rows, columns = 11, 11
env_points = [(0,0), (0,10), (10,10), (10,0)]
env_points.append(env_points[0])
x, y = zip(*env_points)
plt.plot(x, y)

# Draw the obstacles
obs_1 = [(1,8), (2,1), (3,1), (2, 9)]
obs_1.append(obs_1[0])
obs_2 = [(4, 7), (4,6), (7,6), (7,2), (9, 2), (9, 7)]
obs_2.append(obs_2[0])
x, y = zip(*obs_1)
plt.plot(x, y)
x, y = zip(*obs_2)
plt.plot(x, y)

# Start and goal location
start, goal = (1,3), (9, 8)
x, y = start
plt.plot(x, y, "-xb")
x, y = goal
plt.plot(x, y, "-pg")

plt.show(block=False)


cmap = colors.ListedColormap(['red', 'blue'])
bounds = [0,10,20]
norm = colors.BoundaryNorm(bounds, cmap.N)

# Initialize all the points in the environment having reward -1
rewards = np.full((rows, columns), -1)
# Set the reward for the goal 
rewards[goal[0], goal[1]] = 100
# print(rewards)

#Create obstacle polygons
obs_poly_1 = Polygon(obs_1)
obs_poly_2 = Polygon(obs_2)

for i in range(rows):
    for j in range(columns):
        p = Point(i,j)
        if obs_poly_1.intersects(p):
            rewards[i,j] = -100
        if obs_poly_2.intersects(p):
            rewards[i,j] = -100
print(rewards)





plt.show()



