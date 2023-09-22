from typing import List

import numpy as np


class Tensor:

    def __init__(self, data: List) -> None:
        self.data = np.asarray(data)
        self.grad = None
        self.data = np.atleast_1d(data)
        self.generator = None

    def __str__(self) -> str:
        """
        :return: Type of data, data and shape of data.
        """
        return f"Tensor, data:{self.data}, shape:{self.data.shape}"

    def backward(self):
        funcs = []
        funcs.append(self.generator)
        while len(funcs):
            generator = funcs.pop()
            if generator is None:
                break
            inputs, outputs = generator.inputs, generator.outputs

            gys = [o.grad for o in outputs]

            gxs = generator.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(inputs, gxs):
                x.grad = gx
                if x.generatpr is not None:
                    funcs.append(inputs.generator)
        # generator = self.generator
        # if generator is not None:
        #     inputs = generator.inputs
        #     inputs.grad = generator.backward(self.grad)
        #     inputs.backward(inputs.grad)


class Function:
    """
    Function class gives forward method and backward method.
    Each instance is supposed to rewrite these two methods .
    """
    def __call__(self, *inputs):

        ys = self.forward(*inputs)
        self.inputs = inputs
        self.outputs = (ys, )if not isinstance(ys, tuple)  else ys
        for y in self.outputs:
            y.generator = self

        return self.outputs if(len(self.outputs) > 1) else self.outputs[0]

    def forward(self):
        raise NotImplementedError

    def backword(self):
        raise NotImplementedError


class Squre_func(Function):

    def forward(self, inputs):
        return Tensor(self.inputs.data ** 2)

    def backward(self, gradient_from_last_layer):
        return (2 * self.inputs.data * gradient_from_last_layer)

class Add(Function):
    def forward(self, x0, x1):
        return Tensor(x0.data + x1.data)

    def backward(self, gys):
        return gys, gys

def add(x0, x1):
    return Add()(x0, x1)

