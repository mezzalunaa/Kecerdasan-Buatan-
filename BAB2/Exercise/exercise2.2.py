# Exercise 2.2: sortarray
def sortarray(xs):
    n = len(xs)
    for i in range(n):
        # Cari indeks angka terkecil dari sisa array
        min_idx = i
        for j in range(i + 1, n):
            if xs[j] < xs[min_idx]:
                min_idx = j
        
        # Tukar angka terkecil ke posisi i
        xs[i], xs[min_idx] = xs[min_idx], xs[i]
    return xs

# Uji coba dengan data acak
data = [5, 2, 0, 4, 1, 3]
t = sortarray(data)
print(t)  # Output: [0, 1, 2, 3, 4, 5]