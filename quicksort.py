def test():
	A = [(1, -2), (2, 5), (3,-4), (4,-10), (5,1)]

	quicksort(A, 0, len(A)-1, 1)
	print(A)


def quicksort(A, p, n, sort_by):
	if p < n:
		pivot = partition(A, p, n, sort_by)
		quicksort(A, p, pivot-1, sort_by)
		quicksort(A, pivot+1, n, sort_by)
	return A


def partition(A, p, n, sort_by):
	
	x = A[n][sort_by]
	i = p - 1

	for j in range(p, n):
		if A[j][sort_by] <= x:
			i = i + 1
			
			temp = A[j]
			A[j] = A[i]
			A[i] = temp
	
	temp = A[n]
	A[n] = A[i+1]
	A[i+1] = temp
	
	return i+1

if __name__ == '__main__':
	test()

