n = 2
list = ["A", "B", "C", "D", "E"]

def rotate_left(lst, n):
    n = n % len(lst)  # Handle cases where n is greater than the list length
    return lst[n:] + lst[:n]

def rotate_right(lst, n):
    n = n % len(lst)  # Handle cases where n is greater than the list length
    return  lst[:n] + lst[n:]

#print(rotate_left(list, n))
print(rotate_right(list, n))