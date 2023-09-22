import tensor
from tensor import Squre_func
x = tensor.Tensor(2)
y = tensor.Tensor(3)
c = tensor.add(x, y)
print(x)

# func1 = Squre_func()
# func2 = Squre_func()
# y = func1(x)
# z = func2(y)
c.grad = 1
c.backward()
print(x.grad)
