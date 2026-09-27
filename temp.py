import time

n = 2
list = ["A", "B", "C", "D", "E"]

expected_output_left = ["C", "D", "E", "A", "B"]
expected_output_right = ["D", "E", "A", "B", "C"]

def rotate_left(lst, n):
    n = n % len(lst)  # Handle cases where n is greater than the list length
    return lst[n:] + lst[:n]

def rotate_right(lst, n):
    n = n % len(lst)  # Handle cases where n is greater than the list length
    return  lst[-n:] + lst[:-n]

print(rotate_left(list, n))
print(rotate_right(list, n))

import time

def retry(max_attempts=3, delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as ex:
                    print(f"Attempt {attempt + 1}/{max_attempts} failed")
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator
