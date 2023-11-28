
import math
import random
import time
import statistics
import pandas as pd

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
def partition(A,low,high, pivot=None):
  if pivot == None:
    pivot = A[high]
  i = low - 1
  for j in range(low,high):
    if A[j] <= pivot:
      i = i + 1
      A[i],A[j] = A[j],A[i]

  A[i+1],A[high] = A[high],A[i+1]
  return (i+1)


def algo1(A,k):
  # sort the list
  A = mergesort(A)
  # return the kth smallest element
  return A[k]

# Use partition to find kth smallest element

def algo2(A,k):
  # partition the list around the last element
  v = partition(A,0,len(A)-1)
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
def algo3(A, k):
    print(A)
    print("k = " + str(k))

    #divide A into sublists of len 5
    sublists = [A[j:j+5] for j in range(0, len(A), 5)]
    medians = [sorted(sublist)[len(sublist)//2] for sublist in sublists]
    if len(A) <= 5:
        pivot = sorted(A)[k]
        return pivot
    
    else:
        #the pivot is the median of the medians
        pivot = algo3(medians, len(medians)//2)

        #partitioning step
        pos = partition(A, 0, len(A)-1, pivot)

        if pos == k:
            return pivot
        elif pos > k:
            return algo3(A[:pos], k)
        else:
            return algo3(A[pos+1:], k-pos)
    
""" 
    #partitioning step
    low = [j for j in A if j < pivot]
    high = [j for j in A if j > pivot] 

    k = len(low)
    if i < k:
        return algo3(low,i)
    elif i > k:
        return algo3(high,i-k-1)
    else: #pivot = k
        return pivot """
# Testing Cases:

""" # Test Case 1: A = [1,2,3,4,5,6,7,8,9,10], k = 5, return 6
A = [1,2,3,4,5,6,7,8,9,10]
k = 5
n = len(A)

print("Algorithm 1: ",algo1(A,k))
print("Algorithm 2: ",algo2(A,k))
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
print("Algorithm 3: ",algo3(A,k))  """

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

""" 
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

 """
input()

# Experiment

algo1_avg = []
algo2_avg = []
algo3_avg = []

df = pd.DataFrame(columns=['n','algo1','algo2','algo3'])

for i in range(20):

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
print("Algorithm 1: ",algo1_avg)
print("Algorithm 2: ",algo2_avg)
print("Algorithm 3: ",algo3_avg) 

df.to_csv('data.csv',index=False)