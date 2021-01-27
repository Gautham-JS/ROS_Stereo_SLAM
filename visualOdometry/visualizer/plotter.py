import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from itertools import count, combinations
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import itertools



plt.style.use('ggplot')

x_values = []
y_values = []
z_values = []

xgt_values = []
ygt_values = []
zgt_values = []

chi = []
chi_op = []

xopt = []
yopt = []
zopt = []

data = pd.read_csv("/home/gautham/Documents/Projects/LargeScaleMapping/trajectory.csv")
dataOpt = pd.read_csv("/home/gautham/Documents/Projects/LargeScaleMapping/trajectoryOptimized.csv")

index = count()
print(dataOpt.iloc[:5,0], dataOpt.iloc[:5,2])

xBound = np.max( [np.max(data.iloc[:,0]), np.max(data.iloc[:,3])] )
ybound = np.max( [np.max(data.iloc[:,2]), np.max(data.iloc[:,5])] )




def animate(i):
    # x_values = data.iloc[:,1]
    # y_values = data.iloc[:,3]
    
    x_values.append(data.iloc[i,0])
    z_values.append(data.iloc[i,1])
    y_values.append(-1*data.iloc[i,2])

    xgt_values.append(data.iloc[i,3])
    ygt_values.append(data.iloc[i,5])

    xopt.append(dataOpt.iloc[i,0])
    yopt.append(-1*dataOpt.iloc[i,2])



    plt.cla()
    plt.scatter(x_values, y_values, label="Noisy Est.",c="b",s=5)
    #plt.plot(x_values, y_values, label="Predicted Trajectory",c="r")
    plt.scatter(xgt_values, ygt_values, label="True Trajectory",c="g",s=5)
   # plt.plot(xgt_values, ygt_values, label="True Trajectory",c="g")
    plt.scatter(xopt, yopt, label="Optimized Est.",c="r",s=5)
    #plt.plot(xopt, yopt, label="Optimized Prediction",c="b")
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    leg = plt.legend(loc="lower right")
    for handles in leg.legendHandles:
        handles.set_sizes([50.0])
    plt.axis("scaled")
    plt.grid(True)


ani = FuncAnimation(plt.gcf(), animate, len(data.iloc[:,0]), interval=0.1)

plt.show()