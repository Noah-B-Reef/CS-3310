import numpy as np
import matplotlib.pyplot as plt
import time

# Matrix printing function
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
  
# Generate nxn matrices with random entries
def genMats(n):
    A = np.random.rand(n,n)
    B = np.random.rand(n,n)
    return A,B

# Standard matrix multiplication on nxn matrices
def matmul(A,B):
    n = np.shape(A)
    C = np.zeros(n)

    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[1]):
                C[i,j] += A[i,k]*B[k,j]
    return C

# Divide and conquer matrix multiplication on nxn matrices
def divideMatMul(A,B):
    n = np.shape(A)
    if n[0] == 1:
        return A*B
    
    else:
        C11 = divideMatMul(A[0:n[0]//2,0:n[1]//2],B[0:n[0]//2,0:n[1]//2])+divideMatMul(A[0:n[0]//2,n[1]//2:],B[n[0]//2:,0:n[1]//2])
        C12 = divideMatMul(A[0:n[0]//2,0:n[1]//2],B[0:n[0]//2,n[1]//2:])+divideMatMul(A[0:n[0]//2,n[1]//2:],B[n[0]//2:,n[1]//2:])
        C21 = divideMatMul(A[n[0]//2:,0:n[1]//2],B[0:n[0]//2,0:n[1]//2])+divideMatMul(A[n[0]//2:,n[1]//2:],B[n[0]//2:,0:n[1]//2])
        C22 = divideMatMul(A[n[0]//2:,0:n[1]//2],B[0:n[0]//2,n[1]//2:])+divideMatMul(A[n[0]//2:,n[1]//2:],B[n[0]//2:,n[1]//2:])

        C = np.zeros(n)

        C[0:n[0]//2,0:n[1]//2] = C11
        C[0:n[0]//2,n[1]//2:] = C12
        C[n[0]//2:,0:n[1]//2] = C21
        C[n[0]//2:,n[1]//2:] = C22

        return C
    
# Strassen's matrix multiplication on nxn matrices
def strassenMatMul(A,B):
    n = np.shape(A)
    
    if n[0] == 2:
        return matmul(A,B)
    
    else:
        P = strassenMatMul(A[0:n[0]//2,0:n[1]//2]+A[n[0]//2:,n[1]//2:],B[0:n[0]//2,0:n[1]//2]+B[n[0]//2:,n[1]//2:])
        Q = strassenMatMul(A[n[0]//2:,0:n[1]//2]+A[n[0]//2:,n[1]//2:],B[0:n[0]//2,0:n[1]//2])
        R = strassenMatMul(A[0:n[0]//2,0:n[1]//2],B[0:n[0]//2,n[1]//2:]-B[n[0]//2:,n[1]//2:])
        S = strassenMatMul(A[n[0]//2:,n[1]//2:],B[n[0]//2:,0:n[1]//2]-B[0:n[0]//2,0:n[1]//2])
        T = strassenMatMul(A[0:n[0]//2,0:n[1]//2]+A[0:n[0]//2,n[1]//2:],B[n[0]//2:,n[1]//2:])
        U = strassenMatMul(A[n[0]//2:,0:n[1]//2]-A[0:n[0]//2,0:n[1]//2],B[0:n[0]//2,0:n[1]//2]+B[0:n[0]//2,n[1]//2:])
        V = strassenMatMul(A[0:n[0]//2,n[1]//2:]-A[n[0]//2:,n[1]//2:],B[n[0]//2:,0:n[1]//2]+B[n[0]//2:,n[1]//2:])

        C11 = P+S-T+V
        C12 = R+T
        C21 = Q+S
        C22 = P+R-Q+U

        C = np.zeros(n)
        C[0:n[0]//2,0:n[1]//2] = C11
        C[0:n[0]//2,n[1]//2:] = C12
        C[n[0]//2:,0:n[1]//2] = C21
        C[n[0]//2:,n[1]//2:] = C22

        return C

# Experiment

std_avg_time = []
div_avg_time = []
stras_avg_time = []
iters = 10

for i in range(1,iters):
    
    n = 2**i
    print(n)

    std_time = []
    div_time = []
    stras_time = []

    for j in range(10):
        A,B = genMats(n)

        start = time.time()
        matmul(A,B)
        end = time.time()

        std_time.append(end-start)
        
        start = time.time()
        divideMatMul(A,B)
        end = time.time()

        div_time.append(end-start)

        start = time.time()
        strassenMatMul(A,B)
        end = time.time()
        
        stras_time.append(end-start)

    # clean up trials
    std_time.sort()
    div_time.sort()
    stras_time.sort()

    std_time = std_time[1:-1]
    div_time = div_time[1:-1]
    stras_time = stras_time[1:-1]

    std_avg_time.append(np.mean(std_time))
    div_avg_time.append(np.mean(div_time))
    stras_avg_time.append(np.mean(stras_time))

print("Standard average time: ")
print(std_avg_time)
print("Divide and conquer average time: ")
print(div_avg_time)
print("Strassen's average time: ")
print(stras_avg_time)

# Plotting

# Plot Standard Matrix Multiplication
plt.grid()
plt.title("Standard Matrix Multiplication")
plt.xlabel("2^n")
plt.ylabel("Average time (s)")
plt.plot([n for n in range(1,iters)], std_avg_time,label="Standard")
plt.savefig("Standard.png")

# clear plot
plt.clf()

# plot Divide and Conquer Matrix Multiplication
plt.grid()
plt.title("Divide and Conquer Matrix Multiplication")
plt.xlabel("2^n")
plt.ylabel("Average time (s)")
plt.plot([n for n in range(1,iters)],div_avg_time,label="Divide and conquer")
plt.savefig("Divide_and_Conquer.png")

# clear plot
plt.clf()

# plot Strassen's Matrix Multiplication
plt.grid()
plt.title("Strassen's Matrix Multiplication")
plt.xlabel("2^n")
plt.ylabel("Average time (s)")
plt.plot([n for n in range(1,iters)],stras_avg_time,label="Strassen's")
plt.savefig("Strassens.png")

# clear plot
plt.clf()

# plot all three
plt.grid()
plt.title("Matrix Multiplication")
plt.xlabel("2^n")
plt.ylabel("Average time (s)")
plt.plot([n for n in range(1,iters)], std_avg_time,label="Standard")
plt.plot([n for n in range(1,iters)],div_avg_time,label="Divide and conquer")
plt.plot([n for n in range(1,iters)],stras_avg_time,label="Strassen's")
plt.legend()
plt.savefig("All.png")

# Test the algorithms

# Testing case 1


# A = np.matrix([[1,1],[1,1]])
# B = np.matrix([[1,1],[1,1]])
# C = A*B

# print("Test case 1: ")

# print("C = ")
# print(C)

# std = matmul(A,B)
# div = divideMatMul(A,B)
# stras = strassenMatMul(A,B)



# if np.array_equal(std,C):
#     print("Standard matrix multiplication passed!")

# if  np.array_equal(div,C):
#     print("Divide and conquer matrix multiplication passed!")

# if np.array_equal(stras,C):
#     print("Strassen's matrix multiplication passed!")

# Testing case 2

# A = np.matrix([[1,2],[3,4]])
# B = np.matrix([[5,6],[7,8]])
# C = A*B

# print("Test case 2: ")

# print("C = ")
# print(C)

# std = matmul(A,B)
# div = divideMatMul(A,B)
# stras = strassenMatMul(A,B)



# if np.array_equal(std,C):
#     print("Standard matrix multiplication passed!")

# if  np.array_equal(div,C):
#     print("Divide and conquer matrix multiplication passed!")

# if np.array_equal(stras,C):
#     print("Strassen's matrix multiplication passed!")

# Testing case 3


# A = np.matrix([[1,2,3,4],[4,5,6,7],[7,8,9,10],[11,12,13,14]])
# B = np.matrix([[15,16,17,18],[19,20,21,22],[23,24,25,26],[27,28,29,30]])
# C = A*B

# print("Test case 3: ")

# print("C = ")
# print(C)

# std = matmul(A,B)
# div = divideMatMul(A,B)
# stras = strassenMatMul(A,B)



# if np.array_equal(std,C):
#     print("Standard matrix multiplication passed!")

# if  np.array_equal(div,C):
#     print("Divide and conquer matrix multiplication passed!")

# if np.array_equal(stras,C):
#     print("Strassen's matrix multiplication passed!")

# Testing case 4


# A = np.matrix([[4,-3],[2,8]])
# B = np.matrix([[4/19,3/28],[-1/19,2/19]])
# C = A*B

# print("Test case 4: ")

# print("C = ")
# print(C)

# std = matmul(A,B)
# div = divideMatMul(A,B)
# stras = strassenMatMul(A,B)

# print("Standard matrix multiplication: ")
# print(std)
# print("Divide and conquer matrix multiplication: ")
# print(div)
# print("Strassen's matrix multiplication: ")
# print(stras)

# if np.array_equal(std,C):
#     print("Standard matrix multiplication passed!")

# if  np.array_equal(div,C):
#     print("Divide and conquer matrix multiplication passed!")

# if np.array_equal(stras,C):
#     print("Strassen's matrix multiplication passed!")

# Testing case 5


# A = np.matrix([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# B = np.matrix([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# C = A*B

# print("Test case 5: ")

# print("C = ")
# print(C)

# std = matmul(A,B)
# div = divideMatMul(A,B)
# stras = strassenMatMul(A,B)



# if np.array_equal(std,C):
#     print("Standard matrix multiplication passed!")

# if  np.array_equal(div,C):
#     print("Divide and conquer matrix multiplication passed!")

# if np.array_equal(stras,C):
#     print("Strassen's matrix multiplication passed!")

# Testing case 6
    

# A = np.matrix([[4,-3],[2,8]])
# B = np.matrix([[0,1],[1,0]])
# C = A*B

# print("Test case 6: ")

# print("C = ")
# print(C)

# std = matmul(A,B)
# div = divideMatMul(A,B)
# stras = strassenMatMul(A,B)



# if np.array_equal(std,C):
#     print("Standard matrix multiplication passed!")

# if  np.array_equal(div,C):
#     print("Divide and conquer matrix multiplication passed!")

# if np.array_equal(stras,C):
#     print("Strassen's matrix multiplication passed!")

# Testing case 7


# A = np.matrix([[1,3],[7,5]])
# B = np.matrix([[1,7],[3,5]])
# C = A*B

# print("Test case 7: ")

# print("C = ")
# print(C)

# std = matmul(A,B)
# div = divideMatMul(A,B)
# stras = strassenMatMul(A,B)



# if np.array_equal(std,C):
#     print("Standard matrix multiplication passed!")

# if  np.array_equal(div,C):
#     print("Divide and conquer matrix multiplication passed!")

# if np.array_equal(stras,C):
#     print("Strassen's matrix multiplication passed!")

# Testing case 8


# A = np.matrix([[2,2],[2,2]])
# B = np.matrix([[2,2],[2,2]])
# C = A*B

# print("Test case 8: ")

# print("C = ")
# print(C)

# std = matmul(A,B)
# div = divideMatMul(A,B)
# stras = strassenMatMul(A,B)



# if np.array_equal(std,C):
#     print("Standard matrix multiplication passed!")

# if  np.array_equal(div,C):
#     print("Divide and conquer matrix multiplication passed!")

# if np.array_equal(stras,C):
#     print("Strassen's matrix multiplication passed!")

# Testing case 9


# A = np.matrix([[0,0],[0,0]])
# B = np.matrix([[10,20],[30,40]])
# C = A*B

# print("Test case 9: ")

# print("C = ")
# print(C)

# std = matmul(A,B)
# div = divideMatMul(A,B)
# stras = strassenMatMul(A,B)



# if np.array_equal(std,C):
#     print("Standard matrix multiplication passed!")

# if  np.array_equal(div,C):
#     print("Divide and conquer matrix multiplication passed!")

# if np.array_equal(stras,C):
#     print("Strassen's matrix multiplication passed!")

# Testing case 10


# A = np.matrix([[0,1],[1,0]])
# B = np.matrix([[0,1],[1,0]])
# C = A*B

# print("Test case 10: ")

# print("C = ")
# print(C)

# std = matmul(A,B)
# div = divideMatMul(A,B)
# stras = strassenMatMul(A,B)

# if np.array_equal(std,C):
#     print("Standard matrix multiplication passed!")

# if  np.array_equal(div,C):
#     print("Divide and conquer matrix multiplication passed!")

# if np.array_equal(stras,C):
#     print("Strassen's matrix multiplication passed!")