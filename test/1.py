a = [[5, 2], [1, 3], [2, 1], [1, 2]]
print a
b = sorted(a, key=lambda x:x[0])
print b
print zip(*b)
print list(zip(*b)[0])