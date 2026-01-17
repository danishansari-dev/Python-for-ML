# NumPy Complete Notes for Revision

## 1. NumPy Basics

- **NumPy** is a library for fast numerical computing in Python using n-dimensional arrays
- **Import convention**: `import numpy as np`
- **Core object**: `np.ndarray` (homogeneous, contiguous, fast)
- NumPy arrays are much faster than Python lists for numerical operations

---

## 2. Creating Arrays

### From Lists
```python
np.array([1, 3, 4])          # 1D array
np.array([1, 2, 3, 4, 5, 6]) # 1D array
```

### Using reshape()
```python
np.arange(1, 13).reshape(3, 4)    # 2D array (3×4)
np.arange(8).reshape(2, 2, 2)     # 3D array
```

### Data Types
```python
np.array([2, 3, 4], dtype=float)  # Specify dtype
```

### Range-like Creators
```python
np.arange(start, stop, step)      # Like Python range (stop is exclusive)
np.linspace(start, stop, num)     # num equally spaced points (both ends included)

# Examples:
np.arange(1, 11)                  # [1, 2, 3, ..., 10]
np.arange(1, 11, 2)               # [1, 3, 5, 7, 9]
np.linspace(-10, 10, 100)         # 100 points from -10 to 10
```

### Special Arrays
```python
np.ones((3, 4))        # Array of ones, shape (3, 4)
np.zeros((3, 4))       # Array of zeros, shape (3, 4)
np.identity(3)         # 3×3 identity matrix
np.random.random((3, 4)) # Random values in [0, 1), shape (3, 4)
```

---

## 3. Array Attributes

For any array `a`:

```python
a.ndim       # Number of dimensions
a.shape      # Size of each dimension (as tuple)
a.size       # Total number of elements
a.itemsize   # Bytes per element
a.dtype      # Data type (int64, float64, etc.)
```

### Changing Data Type
```python
a.astype(np.int32)     # Convert to int32
a.astype(float)        # Convert to float
```

---

## 4. Indexing & Slicing

Assume:
```python
a1 = np.arange(10)                 # 1D: [0, 1, 2, ..., 9]
a2 = np.arange(12).reshape(3, 4)   # 2D: 3×4 matrix
a3 = np.arange(8).reshape(2, 2, 2) # 3D array
```

### 1D Indexing
```python
a1[0]        # First element
a1[-1]       # Last element
a1[2:5]      # Elements from index 2 to 4 (stop exclusive)
a1[2:5:2]    # Every 2nd element from 2 to 4
a1[::2]      # Every 2nd element from start to end
```

### 2D Indexing
```python
a2[0, 0]     # Element at row 0, col 0
a2[1, 2]     # Element at row 1, col 2
a2[0]        # Entire first row
a2[:, 1]     # Entire second column
a2[1:3, 1:3] # 2×2 subarray from rows 1-2, cols 1-2
a2[::2, ::2] # Every 2nd row and every 2nd column
a2[1, :]     # Entire second row
a2[:, 2]     # Entire third column
```

### 3D Indexing
```python
a3[0, 1, 1]  # Element at specific position
a3[0, :, :]  # First 2D slice
a3[:, 0, 0]  # First column
```

---

## 5. Iteration

```python
# 1D
for i in a1:
    print(i)

# 2D (iterates over rows)
for row in a2:
    print(row)

# 3D (iterates over 2D slices)
for block in a3:
    print(block)

# Iterate all elements in any dimension
for x in np.nditer(a3):
    print(x)
```

---

## 6. Reshaping & Transposing

```python
a.reshape(new_shape)     # Reshape to new dimensions
a.T                      # Transpose (flip rows and columns)
np.transpose(a)          # Same as a.T
a.ravel()                # Flatten nD array to 1D
```

**Examples:**
```python
a = np.arange(12)
a.reshape(3, 4)    # 12 elements → 3×4 matrix
a.reshape(2, 2, 3) # 12 elements → 2×2×3 array

b = np.arange(12).reshape(3, 4)
b.T                # Transpose: 3×4 → 4×3
b.ravel()          # Flatten to 1D: [0, 1, 2, ..., 11]
```

---

## 7. Stacking & Splitting

### Stacking
Given `a4`, `a5` both shape (3, 4):

```python
np.hstack((a4, a5))  # Horizontal stack: (3, 4) + (3, 4) → (3, 8)
np.vstack((a4, a5))  # Vertical stack: (3, 4) + (3, 4) → (6, 4)
```

### Splitting
```python
np.hsplit(a4, 2)  # Split into 2 parts horizontally (by columns)
np.vsplit(a4, 3)  # Split into 3 parts vertically (by rows)
```

---

## 8. Array Operations

### Scalar Operations
```python
a1 + 2              # Add 2 to every element
a1 * 2              # Multiply every element by 2
a1 ** 2             # Square every element
a2 > 15             # Relational: returns boolean array
```

### Vectorized Operations (Same-shaped arrays)
```python
a1 + a2             # Element-wise addition
a1 - a2             # Element-wise subtraction
a1 * a2             # Element-wise multiplication
a1 / a2             # Element-wise division
```

---

## 9. Array Functions

### Reduction Functions
```python
np.min(a)           # Minimum value
np.max(a)           # Maximum value
np.sum(a)           # Sum of all elements
np.prod(a)          # Product of all elements
np.mean(a)          # Mean/average
np.median(a)        # Median
np.std(a)           # Standard deviation
np.var(a)           # Variance
```

### With axis parameter (for 2D/3D)
```python
np.min(a, axis=0)   # Min along axis 0 (down columns)
np.min(a, axis=1)   # Min along axis 1 (across rows)
np.sum(a, axis=0)   # Sum down columns
np.sum(a, axis=1)   # Sum across rows
```

### Linear Algebra
```python
np.dot(A, B)        # Dot product (matrix multiplication)
# A shape (m, n), B shape (n, p) → result shape (m, p)
```

### Mathematical Functions
```python
np.exp(a)           # e^x for each element
np.log(a)           # Natural logarithm
np.sqrt(a)          # Square root
np.sin(a), np.cos(a), np.tan(a)  # Trigonometric functions
```

### Rounding
```python
np.round(a)         # Round to nearest integer
np.floor(a)         # Round down
np.ceil(a)          # Round up
```

---

## 10. Advanced Indexing

### Fancy Indexing (Using arrays/lists as indices)
```python
a = np.arange(24).reshape(6, 4)

a[[0, 2, 3]]        # Select rows 0, 2, 3
a[:, [0, 2, 3]]     # Select columns 0, 2, 3
```

### Boolean Indexing
```python
a = np.random.randint(1, 100, (6, 4))

# Find values greater than 50
mask = a > 50
a[mask]             # All values > 50

# Multiple conditions
a[(a > 50) & (a % 2 == 0)]  # >50 AND even
a[(a % 7) != 0]             # Not divisible by 7
```

**Note**: Use `&` (AND), `|` (OR), `~` (NOT) for boolean arrays, NOT `and`/`or`

---

## 11. Broadcasting

**Broadcasting** automatically expands smaller arrays to match larger ones for element-wise operations.

### Broadcasting Rules
1. If arrays have different dimensions, add dimensions of size 1 to the front of the smaller array
2. For each dimension: sizes must match OR one must be 1
3. If a dimension size doesn't match and neither is 1 → Error

### Examples
```python
# (2, 3) + (1, 3) → broadcasts to (2, 3)
a = np.arange(6).reshape(2, 3)
b = np.arange(3).reshape(1, 3)
a + b  # Works fine

# (3, 4) + (4, 3) → ERROR (incompatible shapes)
```

---

## 12. NumPy in Machine Learning

### Sigmoid Function
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

sigmoid(0)      # 0.5
sigmoid(np.array([0, 1, -1]))
```

### Mean Squared Error (MSE)
```python
def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)

actual = np.array([1, 2, 3, 4])
predicted = np.array([1.1, 2.2, 2.9, 3.8])
mse(actual, predicted)
```

### Binary Cross Entropy
```python
bce = -np.mean(actual * np.log(predicted) + 
               (1 - actual) * np.log(1 - predicted))
```

---

## 13. Handling Missing Values (NaN)

```python
a = np.array([1, 2, 3, 4, np.nan, 6])

np.isnan(a)         # Boolean array showing NaN locations
a[~np.isnan(a)]     # Filter out NaNs
a[np.isnan(a)]      # Select only NaNs
```

---

## 14. Utility Functions & Tricks

### Sorting
```python
np.sort(a)          # Returns sorted copy

# 2D sorting
np.sort(b, axis=1)  # Sort each row
np.sort(b, axis=0)  # Sort each column
```

### Append
```python
np.append(a, 100)   # Append value at end

# 2D: Add column of ones
np.append(b, np.ones((b.shape[0], 1)), axis=1)
```

### Concatenate
```python
np.concatenate((a, b), axis=0)  # Stack rows (vertical)
np.concatenate((a, b), axis=1)  # Stack columns (horizontal)
```

### Unique Elements
```python
np.unique(a)        # Sorted unique values
```

### Expand Dimensions
```python
a = np.arange(10)           # Shape (10,)
np.expand_dims(a, axis=0)   # Shape (1, 10) - row vector
np.expand_dims(a, axis=1)   # Shape (10, 1) - column vector
```

### np.where
```python
np.where(condition, x, y)   # Pick from x if condition true, else y

# Examples:
np.where(a > 50, 0, a)      # Replace values >50 with 0
np.where(a % 2 == 0, 0, a)  # Replace even numbers with 0

# Get indices where condition is true
np.where(a > 50)            # Returns tuple of indices
```

### argmax / argmin
```python
np.argmax(a)        # Index of maximum value
np.argmin(a)        # Index of minimum value

# For 2D:
np.argmax(b, axis=1)  # Index of max in each row
```

### Cumulative Operations
```python
np.cumsum(a)         # Cumulative sum
np.cumprod(a)        # Cumulative product

# 2D with axis
np.cumsum(b, axis=0) # Cumsum down each column
np.cumsum(b, axis=1) # Cumsum across each row
```

### Percentile & Median
```python
np.percentile(a, 50)   # 50th percentile (median)
np.percentile(a, 75)   # 75th percentile
np.percentile(a, 100)  # 100th percentile (max)
np.median(a)           # Same as percentile(a, 50)
```

### Histogram
```python
counts, bin_edges = np.histogram(a, bins=[0, 50, 100])
# Counts: how many elements in each bin
# bin_edges: [0, 50, 100]
```

### Correlation
```python
np.corrcoef(x, y)     # 2×2 correlation matrix between x and y
```

### Membership Test
```python
items = [10, 20, 30, 40, 50]
np.isin(a, items)     # Boolean: is element in items?
a[np.isin(a, items)]  # Keep only elements in items
```

### Flip / Reverse
```python
np.flip(a)            # Reverse all elements
np.flip(b, axis=1)    # Reverse each row
np.flip(b, axis=0)    # Reverse order of rows
```

### Put & Delete
```python
np.put(a, [0, 1], [110, 530])  # Set elements at indices
np.delete(a, [0, 2, 4])        # Remove elements at indices
```

### Set Operations (1D Arrays)
```python
m = np.array([1, 2, 3, 4, 5])
n = np.array([3, 4, 5, 6, 7])

np.union1d(m, n)         # [1, 2, 3, 4, 5, 6, 7]
np.intersect1d(m, n)     # [3, 4, 5]
np.setdiff1d(m, n)       # [1, 2]
np.setxor1d(m, n)        # [1, 2, 6, 7]
```

### Clip Values
```python
np.clip(a, amin=40, amax=75)  # Values <40→40, >75→75
```

---

## 15. NumPy vs Python Lists

### Speed Comparison
- **Pure Python lists**: Very slow for large operations (10M elements)
- **NumPy vectorized operations**: 10-100x faster for same task

### Memory Efficiency
- NumPy arrays use less memory than Python lists
- Example: int8 dtype uses only 1 byte per element vs Python int objects

```python
import sys
python_list = list(range(10000000))
sys.getsizeof(python_list)  # Much larger

numpy_array = np.arange(10000000, dtype=np.int8)
sys.getsizeof(numpy_array)  # Much smaller
```

---

## 16. Quick Reference: Most Common Functions

| Function | Purpose |
|----------|---------|
| `np.array()` | Create array from list |
| `np.arange()` | Create array with range |
| `np.linspace()` | Create evenly spaced array |
| `np.ones()`, `np.zeros()` | Create ones or zeros |
| `np.reshape()` | Change shape |
| `np.T` | Transpose |
| `np.min()`, `np.max()` | Min/max value |
| `np.sum()`, `np.mean()` | Sum/average |
| `np.sort()` | Sort array |
| `np.where()` | Conditional selection |
| `np.dot()` | Matrix multiplication |
| `np.hstack()`, `np.vstack()` | Stack arrays |
| `np.concatenate()` | Join arrays |
| `np.unique()` | Get unique elements |
| `np.cumsum()` | Cumulative sum |
| `np.argmax()`, `np.argmin()` | Index of max/min |

---

## 17. Common Mistakes to Avoid

1. **Using `and`/`or` with boolean arrays**: Use `&` and `|` instead
   ```python
   # WRONG: a > 50 and a < 100
   # RIGHT: (a > 50) & (a < 100)
   ```

2. **Forgetting `axis` parameter**: May flatten unexpectedly
   ```python
   np.sum(a)         # Sum all elements
   np.sum(a, axis=1) # Sum each row
   ```

3. **Broadcasting confusion**: Always check shapes match or one has size 1
   ```python
   # (3, 4) + (1, 4) ✓ Works
   # (3, 4) + (4, 3) ✗ Error
   ```

4. **Modifying original array**: Some operations create views, not copies
   ```python
   a_copy = a.copy()  # Explicit copy
   ```

---

## 18. Important Data Types

```python
np.int8, np.int16, np.int32, np.int64       # Integer types
np.float32, np.float64                       # Float types
np.complex64, np.complex128                  # Complex numbers
np.bool_                                     # Boolean
```

---

## 19. Practice Tips for Revision

1. **Create arrays** in different ways and practice reshaping
2. **Index and slice** multi-dimensional arrays
3. **Use boolean indexing** to filter data
4. **Apply functions** with and without axis parameters
5. **Practice broadcasting** with different shaped arrays
6. **Implement ML functions** (sigmoid, MSE) from scratch
7. **Compare performance** between lists and NumPy for large data

---

## 20. Summary: NumPy Workflow

```python
import numpy as np

# 1. Create data
data = np.array([1, 2, 3, 4, 5])

# 2. Inspect
print(data.shape, data.dtype, data.size)

# 3. Manipulate
reshaped = data.reshape(5, 1)
result = np.sum(data)

# 4. Index/slice
subset = data[1:4]
mask = data > 2
filtered = data[mask]

# 5. Apply functions
sorted_data = np.sort(data)
unique_vals = np.unique(data)

# 6. Combine arrays
combined = np.concatenate((data, data))
```

---

**Last Updated**: January 2026
**For Revision**: Cover one section per study session for comprehensive understanding
