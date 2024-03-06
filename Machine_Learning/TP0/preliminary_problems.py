import numpy as np

# Help: https://numpy.org/doc/

# Let's start with something easy.
def task1_naive():
    np.random.seed(1)
    a = np.random.rand(100, 5)
    s = np.zeros((5,))
    for i in range(a.shape[0]):
        m = 0
        for j in range(a.shape[1]):
            m += a[i, j]**2
        m = m**0.5
        for j in range(a.shape[1]):
            s[j] += a[i, j] / m
    for j in range(s.shape[0]):
        s[j] /= a.shape[0]
    return s

def task1():
    np.random.seed(1)
    a = np.random.rand(100, 5)
    # norm across the columns, then mean across the rows
    # TODO

# Not bad. Let's see what you've got!
def task2_naive():
    np.random.seed(2)
    a = np.random.randn(32, 30, 10)
    b = np.random.randn(32, 10)
    c = np.zeros((32, 30))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                c[i, j] += a[i, j, k] * b[i, k]
    return c

def task2():
    np.random.seed(2)
    a = np.random.randn(32, 30, 10)
    b = np.random.randn(32, 10)
    # c = a * b, for batches of matrices
    # TODO

# Well done! Let's up the ante a little bit!
def task3_naive():
    np.random.seed(3)
    a = np.random.rand(200, 100) + 1e-6
    b = np.random.rand(200, 100)
    r = 0
    k = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if b[i, j] > 0.2:
                r += np.log(a[i, j]) * 42013**-k
                k += 1
    return r

def task3():
    np.random.seed(3)
    a = np.random.rand(200, 100) + 1e-6
    b = np.random.rand(200, 100)
    # have to figure out that indexing stuff
    # TODO

#######################################################################
#                   The following are not mandatory                   #
#######################################################################

# OK, let's leave Beginner Zone. Can you take this one?
def task4_naive():
    np.random.seed(4)
    a = np.random.randint(low=1, high=7, size=(32, 20))
    b = np.random.rand(32, 8, 20)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            c = a[i, j]
            for k in range(c - 1, c + 2):
                b[i, k, j] = 0
    return b

def task4():
    np.random.seed(4)
    a = np.random.randint(low=1, high=7, size=(32, 20))
    b = np.random.rand(32, 8, 20)
    # have to figure out some clever masking
    # TODO

# This is the last one, don't give up now!
def task5_naive():
    np.random.seed(5)
    a = np.random.rand(32, 8)
    b = np.random.randn(32, 8, 3, 3)
    c = np.zeros((8, 3, 3))
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            for k in range(b.shape[2]):
                for l in range(b.shape[3]):
                    c[j, k, l] += a[i, j] * b[i, j, k, l]
    return c

def task5():
    np.random.seed(5)
    a = np.random.rand(32, 8)
    b = np.random.randn(32, 8, 3, 3)
    # Did you know there is a function in Numpy called Einstein?
    # TODO

def main():
    # Comment out the ones you haven't implemented yet
    tasks = [(task1_naive, task1), (task2_naive, task2), (task3_naive, task3),
             (task4_naive, task4), (task5_naive, task5)]
    for i, (fref, fsol) in enumerate(tasks):
        ref = fref()
        sol = fsol()
        print(f'Task{i+1}:', np.allclose(ref, sol))

if __name__ == "__main__":
    main()
