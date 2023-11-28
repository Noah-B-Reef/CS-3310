
import math
import random
import time
import statistics
import pandas as pd
import matplotlib.pyplot as plt

# Standard Merge Sort Algorithm Implementation
# Merge Sort
def mergesort(lst):
  if len(lst) <= 1:
    return lst

  else:
    mid = len(lst)//2

    L = mergesort(lst[:mid])
    R = mergesort(lst[mid:])
    result = merge(L,R)
    return result
  
# Merge
def merge(L,R):
  result = []
  i = 0
  j = 0

  while i < len(L) and j < len(R):
    if L[i] <= R[j]:
      result.append(L[i])
      i += 1

    else:
      result.append(R[j])
      j += 1

  result = result + L[i:] + R[j:]
  return result

# Partition
def partition(A, pivot=None):
  low = 0

  if pivot == None:
    pivot = A[0]
  i = 0

  for j in range(1,len(A)):
    if A[j] < pivot:
      i = i + 1
      A[i],A[j] = A[j],A[i]
  pivot_pos = i
  A[0],A[pivot_pos] = A[pivot_pos],A[0]
  return i

 
def algo1(A,k):
  # sort the list
  A = mergesort(A)
  # return the kth smallest element
  return A[k]

# Use partition to find kth smallest element

def algo2(A,k):
  # partition the list around the last element
  v = partition(A)
  # if the kth smallest element is the last element, return it
  if v == k:
    return A[v]
  # if the kth smallest element is in the left sublist, recurse on the left sublist
  elif k < v:
    return algo2(A[:v],k)
  # if the kth smallest element is in the right sublist, recurse on the right sublist
  else:
    return algo2(A[v+1:],k-v-1)


# Use recursive partition and median of medians to find kth smallest element 0 indexed
def algo3(A, i):

    #divide A into sublists of len 5
    sublists = [A[j:j+5] for j in range(0, len(A), 5)]
    medians = [sorted(sublist)[len(sublist)//2] for sublist in sublists]
    if len(medians) <= 5:
        pivot = sorted(medians)[len(medians)//2]
    else:
        #the pivot is the median of the medians
        pivot = algo3(medians, len(medians)//2)

    #partitioning step
    low = [j for j in A if j < pivot]
    high = [j for j in A if j > pivot]

    k = len(low)
    if i < k:
        return algo3(low,i)
    elif i > k:
        return algo3(high,i-k-1)
    else: #pivot = k
        return pivot 

""" def algo3(A,k):
  print(A)
  print(k)
  if len(A) <= 5:
    return sorted(A)[k-1]
  else:
    size = len(A)//5
    medians = []
    for i in range(0,size,5):
      medians.append(sorted(A[i:i+5])[len(A[i:i+5])//2])
    if size*5 < len(A):
      medians.append(sorted(A[size*5:])[len(A[size*5:])//2])


    v = algo3(medians,math.ceil(size/2))
    # partition the list around the last element
    pivot_pos = partition(A,v)

    if k == pivot_pos:
      return v
    elif k < pivot_pos:
      return algo3(A[:pivot_pos],k)
    else:
      return algo3(A[pivot_pos+1:],k-pivot_pos-1)ß
 """
# Testing Cases:

# Test Case 1: A = [1,2,3,4,5,6,7,8,9,10], k = 5, return 6
A = [1,2,3,4,5,6,7,8,9,10]
k = 5
n = len(A)

print("Algorithm 1: ",algo1(A,k))
A = [1,2,3,4,5,6,7,8,9,10]
k = 5
print("Algorithm 2: ",algo2(A,k))
A = [1,2,3,4,5,6,7,8,9,10]
k = 5
print("Algorithm 3: ",algo3(A,k))

# Testing Case 2: A = [10,2,8,4,6,5,7,3,9,1], k = 0, return 1
A = [10,2,8,4,6,5,7,3,9,1]
k = 0
n = len(A)

print("Algorithm 1: ",algo1(A,k))
print("Algorithm 2: ",algo2(A,k))
print("Algorithm 3: ",algo3(A,k))

# Testing Case 3: A = [10,2,8,4,6,5,7,3,9,1], k = 9, return 10

A = [10,2,8,4,6,5,7,3,9,1]
k = 9
n = len(A)

print("Algorithm 1: ",algo1(A,k))
print("Algorithm 2: ",algo2(A,k))
print("Algorithm 3: ",algo3(A,k))  

# Testing Case 4: A = [10,2,8,4,6,5,7,3,9,1], k = 5, return 6

print("Testing Case 4: A = [11,2,8,4,6,5,7,3,9,1], k = 5, return 6")
A = [10,2,8,4,6,5,7,3,9,1]
k = 5
n = len(A)

print("Algorithm 1: ",algo1(A,k))

A = [10,2,8,4,6,5,7,3,9,1]
k = 5

print("Algorithm 2: ",algo2(A,k))

A = [10,2,8,4,6,5,7,3,9,1]
k = 5
print("Algorithm 3: ",algo3(A,k)) 

# Testing Case 5: A = [10,2,8,4,6,5,7,3,9,1], k = 4, return 5

A = [11,2,8,4,6,5,7,3,9,1]
k = 4
n = len(A)

print("Algorithm 1: ",algo1(A,k))
print("Algorithm 2: ",algo2(A,k))
print("Algorithm 3: ",algo3(A,k)) 

# Testing Case 6: A = [5,2,3] k = 1, return 3
A = [5,2,3]
k = 1
n = len(A)
print("Algorithm 1: ",algo1(A,k))
print("Algorithm 2: ",algo2(A,k))
print("Algorithm 3: ",algo3(A,k))

# Testing Case 7: A = [5,2,3] k = 2, return 5
A = [5,2,3]
k = 2
n = len(A)
print("Algorithm 1: ",algo1(A,k))
print("Algorithm 2: ",algo2(A,k))
print("Algorithm 3: ",algo3(A,k))  

# Testing Case 8: A = [5,2,3] k = 0, return 2
A = [5,2,3]
k = 0
n = len(A)
print("Algorithm 1: ",algo1(A,k))
print("Algorithm 2: ",algo2(A,k))
print("Algorithm 3: ",algo3(A,k))  

# Testing Case 9: A = [11,2,8,4,6,5,7,3,9,1,10] k = 5, return 6
A = [11,2,8,4,6,5,7,3,9,1,10]
k = 5
n = len(A)
print("Algorithm 1: ",algo1(A,k))
print("Algorithm 2: ",algo2(A,k))
print("Algorithm 3: ",algo3(A,k)) 

# Testing Case 10: A = [11,2,8,4,6,5,7,3,9,1,10] k = 10, return 11
A = [11,2,8,4,6,5,7,3,9,1,10]
k = 10
n = len(A)
print("Algorithm 1: ",algo1(A,k))
print("Algorithm 2: ",algo2(A,k))
print("Algorithm 3: ",algo3(A,k))  

input()


# Experiment

algo1_avg = []
algo2_avg = []
algo3_avg = []
iters = 50

df = pd.DataFrame(columns=['n','algo1','algo2','algo3'])

for i in range(1,iters):

  algo1_time = []
  algo2_time = []
  algo3_time = []

  n = 2**i

  for j in range(10):
    # set size of the list
    A = random.sample(range(1,2**(i+1)),n)
    k = random.randint(0,n-1)

    # Algorithm 1
    start_time = time.time()
    algo1(A,k)
    end_time = time.time()
    algo1_time.append(end_time-start_time)

    # Algorithm 2
    start_time = time.time()
    algo2(A,k)
    end_time = time.time()
    algo2_time.append(end_time-start_time)

    # Algorithm 3
    start_time = time.time()
    algo3(A,k)
    end_time = time.time()
    algo3_time.append(end_time-start_time)
  
  # clean up the data
  algo1_time.sort()
  algo2_time.sort()
  algo3_time.sort()

  algo1_time = algo1_time[1:-1]
  algo2_time = algo2_time[1:-1]
  algo3_time = algo3_time[1:-1]

  # calculate the average
  algo1_avg.append(sum(algo1_time)/len(algo1_time))
  algo2_avg.append(sum(algo2_time)/len(algo2_time))
  algo3_avg.append(sum(algo3_time)/len(algo3_time))

  # add to the dataframe
  df = pd.concat([df,pd.DataFrame({'n':[n],'algo1':[algo1_avg[-1]],'algo2':[algo2_avg[-1]],'algo3':[algo3_avg[-1]]})],ignore_index=True)

# print the result
df.to_csv('data.csv',index=False)

# Plotting

# Plot Standard Matrix Multiplication
plt.grid()
plt.title("Algorithm 1: Mergesort")
plt.xlabel("2^n")
plt.ylabel("Average time (s)")
plt.plot([x for x in range(1,iters)], algo1_avg,label="Algorithm 1")
plt.savefig("algo1.png")

# clear plot
plt.clf()

# plot Divide and Conquer Matrix Multiplication
plt.grid()
plt.title("Algorithm 2: Partition")
plt.xlabel("2^n")
plt.ylabel("Average time (s)")
plt.plot([x for x in range(1,iters)],algo2_avg,label="Algorithm 2")
plt.savefig("algo2.png")

# clear plot
plt.clf()

# plot Strassen's Matrix Multiplication
plt.grid()
plt.title("Algorithm 3: Median of Medians")
plt.xlabel("2^n")
plt.ylabel("Average time (s)")
plt.plot([x for x in range(1,iters)],algo3_avg,label="Algorithm 3")
plt.savefig("algo3.png")

# clear plot
plt.clf()

# plot all three
plt.grid()
plt.title("Kth Smallest Element")
plt.xlabel("2^n")
plt.ylabel("Average time (s)")
plt.plot([x for x in range(1,iters)], algo1_avg,label="Algorithm 1")
plt.plot([x for x in range(1,iters)],algo2_avg,label="Algorithm 2")
plt.plot([x for x in range(1,iters)],algo3_avg,label="Algorithm 3")
plt.legend()
plt.savefig("All.png")
