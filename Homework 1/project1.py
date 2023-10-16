import numpy as np

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

# Test the algorithms

# Testing case 1

'''
A = np.matrix([[1,1],[1,1]])
B = np.matrix([[1,1],[1,1]])
'''

# Testing case 2

'''
A = np.matrix([[1,2],[3,4]])
B = np.matrix([[5,6],[7,8]])
'''

# Testing case 3

'''
A = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
B = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
'''

# Testing case 4

'''
A = np.matrix([[4,-3],[2,8]])
B = np.matrix([[4/19,3/28],[-1/19,2/19]])
'''

# Testing case 5

'''
A = np.matrix([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
B = np.matrix([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
'''

# Testing case 6
    
'''
A = np.matrix([[4,-3],[2,8]])
B = np.matrix([[0,1],[1,0]])
'''

