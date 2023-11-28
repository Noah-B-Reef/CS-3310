# Merge Sort Implementation
def mergesort(lst):
    if len(lst) <= 1:
        return lst
    
    else:
        mid = len(lst)//2
    
        L = mergesort(lst[:mid])
        R = mergesort(lst[mid:])
        result = merge(L,R)
        return result
    
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

