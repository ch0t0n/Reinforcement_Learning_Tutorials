import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
delta = 0.5
box = [(0.5,0.5),(0.5,10.5),(10.5,10.5),(10.5,0.5)]
box.append(box[0])
x, y = zip(*box)
fig = plt.figure()
delta = 0.5
plt.plot(x,y, linewidth = 2)
plt.grid(color='black', linestyle='-', linewidth=1, animated=True)
plt.xticks(np.arange(0.5, 10, 1))
plt.yticks(np.arange(0.5, 10, 1))

for i in range(1, 10):
    # plt.cla()    
    plt.plot(i, i, "-Xb")
    plt.pause(.1)



plt.show()