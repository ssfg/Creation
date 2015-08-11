# from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import base64

my_data = np.genfromtxt('mnist_coords.csv', delimiter=',')
my_labels = np.genfromtxt('mnist_ys.csv', delimiter=',')

X = my_data[:1000,0]
y = my_data[:1000,1]
label = my_labels[:1000,0]

lab64 = my_labels[:,0]

np.savetxt("test_csv.csv", lab64, delimiter=",")
new_data = np.genfromtxt('test_csv.csv', dtype=np.float32, delimiter=',')
# print new_data
print type(new_data[0])
print new_data.shape
print new_data
# # new_data = new_data.astype(np.float32, copy=False)
# # y = new_data.view('float32')
# # y[:] = new_data[:]


# print new_data
# print type(new_data[0])

# new_data = '\0' * (5 - len(new_data)) + new_data

new_data = base64.encodestring(new_data)
print new_data

with open("save_b64.txt", "w") as text_file:
    text_file.write(new_data)

# labshape = lab64.shape[0]
# # get the type of the array element  = float64
# # print type(lab64[0])
# np.ascontiguousarray(lab64, dtype=np.float64)
# np.reshape(lab64, (labshape,1))
# print lab64.flags['C_CONTIGUOUS']
# np.asfarray(lab64, dtype=np.float64)
# my_labels64 = base64.b64encode(lab64)
# print my_labels64





# print my_data.shape
# print my_data
# print my_labels
# print my_labels


# N = 500
# # x = np.random.rand(N)
# # y = np.random.rand(N)
# # colors = np.random.rand(N)
# area = 20 #np.pi * (15 * np.random.rand(N))**2 # 0 to 15 point radiuses

# # for some reason the input from output has a NaN appended: which is the '\n' character.
# plt.scatter(X, y, s=area, c=label, alpha=0.5)
# plt.show()