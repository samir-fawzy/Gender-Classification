from model import predictions
import numpy as np
from cleaner_data import x_testing , y_testing
import matplotlib.pyplot as plt

theta = np.load("src\\data\\opt_theta.npy")

prediction = predictions(x_testing,theta)



# for i in range(len(prediction)):
#     if prediction[i] >= 0.5:
#         prediction[i] = 1
#     else:
#         prediction[i] = 0

prediction_label = (prediction >= 0.5).astype(int)

data = np.column_stack([y_testing.astype(int), prediction, prediction_label])  # شكل (m, 3)

np.savetxt("src\\data\\output.txt", data, fmt="%d\t%.6f\t%d",
           header="actual\tprobability\tlabel", comments='')

fix , axs = plt.subplots(1,2)

x_axis = list(range(len(x_testing)))

axs[0].plot(x_axis,y_testing,label="Actual Values")
axs[0].plot(x_axis,prediction,label="Prediction Values",c='r')
axs[0].grid(True)
axs[0].set_title("Line Plot")

axs[1].scatter(x_axis,y_testing,label="Actual Values")
axs[1].scatter(x_axis,prediction,label="Predition Values",alpha=0.4)
axs[1].set_title("Scatter Plot")

plt.savefig("src\\images\\my_plot")
plt.tight_layout()
plt.show()


