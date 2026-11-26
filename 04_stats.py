import statistics
import math

a = float(input("Enter first number: "))
b = float(input("Enter second number: "))
print("Addition:", a + b)
print("Subtraction:", a - b)
print("Multiplication:", a * b)
print("Division:", a / b if b != 0 else "Undefined (division by zero)")

data = list(map(float, input("Enter numbers separated by spaces: ").split()))
print("Mean:", statistics.mean(data))
print("Median:", statistics.median(data))
print("Standard deviation (population):", statistics.pstdev(data))
print("Square root of first number:", math.sqrt(a) if a >= 0 else "Not real")
