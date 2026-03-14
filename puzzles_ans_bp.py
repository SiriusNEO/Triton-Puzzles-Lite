import argparse
from typing import List
import os

import torch
import triton
import triton.language as tl

# Local imports
from display import print_end_line
from tensor_type import Float32, Int32
from test_puzzle import test


"""
# Triton Puzzles Lite

Programming for accelerators such as GPUs is critical for modern AI systems.
This often means programming directly in proprietary low-level languages such as CUDA. Triton is 
an alternative open-source language that allows you to code at a higher-level and compile to accelerators 
like GPU.

Coding for Triton is very similar to Numpy and PyTorch in both syntax and semantics. However, as a lower-level 
language there are a lot of details that you need to keep track of. In particular, one area that learners have 
trouble with is memory loading and storage which is critical for speed on low-level devices.

This set is puzzles is meant to teach you how to use Triton from first principles in an interactive fashion. 
You will start with trivial examples and build your way up to real algorithms like Flash Attention and 
Quantized neural networks. These puzzles **do not** need to run on GPU since they use a Triton interpreter.
"""


r"""
## Introduction

To begin with, we will only use `tl.load` and `tl.store` in order to build simple programs.
"""


"""
### Demo 1

Here's an example of load. It takes an `arange` over the memory. By default the indexing of
torch tensors with column, rows, depths or right-to-left. It also takes in a mask as the second
argument. Mask is critically important because all shapes in Triton need to be powers of two.

Expected Results:

[0 1 2 3 4 5 6 7]
[1. 1. 1. 1. 1. 0. 0. 0.]

Explanation:

tl.load(ptr, mask)
tl.load use mask: [0 1 2 3 4 5 6 7] < 5 = [1 1 1 1 1 0 0 0]
"""


@triton.jit
def demo1(x_ptr):
    range = tl.arange(0, 8)
    # print works in the interpreter
    print(range)
    x = tl.load(x_ptr + range, range < 5, 0)
    print(x)


def run_demo1():
    print("Demo1 Output: ")
    demo1[(1, 1, 1)](torch.ones(4, 3))
    print_end_line()


"""
### Demo 2:

You can also use this trick to read in a 2d array.

Expected Results:

[[ 0  1  2  3]
[ 4  5  6  7]
[ 8  9 10 11]
[12 13 14 15]
[16 17 18 19]
[20 21 22 23]
[24 25 26 27]
[28 29 30 31]]
[[1. 1. 1. 0.]
[1. 1. 1. 0.]
[1. 1. 1. 0.]
[1. 1. 1. 0.]
[0. 0. 0. 0.]
[0. 0. 0. 0.]
[0. 0. 0. 0.]
[0. 0. 0. 0.]]

Explanation:

tl.load use mask: i < 4 and j < 3.
"""


@triton.jit
def demo2(x_ptr):
    i_range = tl.arange(0, 8)[:, None]
    j_range = tl.arange(0, 4)[None, :]
    range = i_range * 4 + j_range
    # print works in the interpreter
    print(range)
    x = tl.load(x_ptr + range, (i_range < 4) & (j_range < 3), 0)
    print(x)


def run_demo2():
    print("Demo2 Output: ")
    demo2[(1, 1, 1)](torch.ones(4, 4))
    print_end_line()


"""
### Demo 3

The `tl.store` function is quite similar. It allows you to write to a tensor.

Expected Results:

tensor([[10., 10., 10.],
    [10., 10.,  1.],
    [ 1.,  1.,  1.],
    [ 1.,  1.,  1.]])

Explanation:

tl.store(ptr, value, mask)
here range < 5 corresponds to the 2D-mask

[[1. 1. 1.]
[1. 1. 0.]
[0. 0. 0.]
[0. 0. 0.]]
"""


@triton.jit
def demo3(z_ptr):
    range = tl.arange(0, 8)
    z = tl.store(z_ptr + range, 10, range < 5)


def run_demo3():
    print("Demo3 Output: ")
    z = torch.ones(4, 3)
    demo3[(1, 1, 1)](z)
    print(z)
    print_end_line()


"""
### Demo 4

You can only load in relatively small `blocks` at a time in Triton. To work 
with larger tensors you need to use a program id axis to run multiple blocks in 
parallel. 

Here is an example with one program axis with 3 blocks.

Expected Results:

Print for each [0] [1. 1. 1. 1. 1. 1. 1. 1.]
Print for each [1] [1. 1. 1. 1. 1. 1. 1. 1.]
Print for each [2] [1. 1. 1. 1. 0. 0. 0. 0.]

Explanation:

This program launch 3 blocks in parallel. For each block (pid=0, 1, 2), it loads 8 
elements. Note that similar to demo3, multi-dimensional tensors are flattened when we 
use pointer (i.e. continuous in memory).
"""


@triton.jit
def demo4(x_ptr):
    pid = tl.program_id(0)
    range = tl.arange(0, 8) + pid * 8
    x = tl.load(x_ptr + range, range < 20)
    print("Print for each", pid, x)


def run_demo4():
    print("Demo4 Output: ")
    x = torch.ones(2, 4, 4)
    demo4[(3, 1, 1)](x)
    print_end_line()


r"""
## Puzzle 1: Constant Add

Add a constant to a vector. Uses one program id axis. 
Block size `B0` is always the same as vector `x` with length `N0`.

.. math::
    z_i = 10 + x_i \text{ for } i = 1\ldots N_0
"""


def add_spec(x: Float32[32,]) -> Float32[32,]:
    "This is the spec that you should implement. Uses typing to define sizes."
    return x + 10.0


@triton.jit
def add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    row_tile_idx = tl.program_id(0)
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N0,),
        strides=(1,),
        offsets=(row_tile_idx * B0,),
        block_shape=(B0,),
        order=(0,)
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(N0,),
        strides=(1,),
        offsets=(row_tile_idx * B0,),
        block_shape=(B0,),
        order=(0,)
    )
    x = tl.load(x_block_ptr,boundary_check=(0,),padding_option='zero')
    z = x + 10.0
    tl.store(z_block_ptr, z, boundary_check=(0,))
    # Finish me!
    return


r"""
## Puzzle 2: Constant Add Block

Add a constant to a vector. Uses one program block axis (no `for` loops yet). 
Block size `B0` is now smaller than the shape vector `x` which is `N0`.

.. math::
    z_i = 10 + x_i \text{ for } i = 1\ldots N_0
"""


def add2_spec(x: Float32[200,]) -> Float32[200,]:
    return x + 10.0


@triton.jit
def add_mask2_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    # Finish me!
    row_tile_idx = tl.program_id(0)
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N0,),
        strides=(1,),
        offsets=(row_tile_idx * B0,),
        block_shape=(B0,),
        order=(0,)
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(N0,),
        strides=(1,),
        offsets=(row_tile_idx * B0,),
        block_shape=(B0,),
        order=(0,)
    )
    x = tl.load(x_block_ptr,boundary_check=(0,),padding_option='zero')
    z = x + 10.0
    tl.store(z_block_ptr, z, boundary_check=(0,))
    return


r"""
## Puzzle 3: Outer Vector Add

Add two vectors.

Uses one program block axis. Block size `B0` is always the same as vector `x` length `N0`.
Block size `B1` is always the same as vector `y` length `N1`.

.. math::
    z_{j, i} = x_i + y_j\text{ for } i = 1\ldots B_0,\ j = 1\ldots B_1
"""


def add_vec_spec(x: Float32[32,], y: Float32[32,]) -> Float32[32, 32]:
    return x[None, :] + y[:, None]


@triton.jit
def add_vec_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    # Finish me!
    row_tile_idx = tl.program_id(0)
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N0,),
        strides=(1,),
        offsets=(row_tile_idx * B0,),
        block_shape=(B0,),
        order=(0,)
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(N1,),
        strides=(1,),
        offsets=(row_tile_idx * B1,),
        block_shape=(B1,),
        order=(0,)
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(N1, N0),
        strides=(N0, 1),
        offsets=(row_tile_idx * B1, row_tile_idx * B0),
        block_shape=(B1, B0),
        order=(0, 1)
    )
    x = tl.load(x_block_ptr,boundary_check=(0,),padding_option='zero')
    y = tl.load(y_block_ptr,boundary_check=(0,),padding_option='zero')
    z = x[None, :] + y[:, None]
    tl.store(z_block_ptr, z, boundary_check=(0,))

    return


r"""
## Puzzle 4: Outer Vector Add Block

Add a row vector to a column vector.

Uses two program block axes. Block size `B0` is always less than the vector `x` length `N0`.
Block size `B1` is always less than vector `y` length `N1`.

.. math::
    z_{j, i} = x_i + y_j\text{ for } i = 1\ldots N_0,\ j = 1\ldots N_1
"""


def add_vec_block_spec(x: Float32[100,], y: Float32[90,]) -> Float32[90, 100]:
    return x[None, :] + y[:, None]


@triton.jit
def add_vec_block_kernel(
    x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    x_tile_idx = tl.program_id(0)
    y_tile_idx = tl.program_id(1)

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N0,),
        strides=(1,),
        offsets=(x_tile_idx * B0,),
        block_shape=(B0,),
        order=(0,)
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(N1,),
        strides=(1,),
        offsets=(y_tile_idx * B1,),
        block_shape=(B1,),
        order=(0,)
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(N1, N0),
        strides=(N0, 1),
        offsets=(y_tile_idx * B1, x_tile_idx * B0),
        block_shape=(B1, B0),
        order=(0, 1)
    )
    x = tl.load(x_block_ptr,boundary_check=(0,),padding_option='zero')
    y = tl.load(y_block_ptr,boundary_check=(0,),padding_option='zero')
    z = x[None, :] + y[:, None]
    tl.store(z_block_ptr, z, boundary_check=(0,1))
    # Finish me!
    return


r"""
## Puzzle 5: Fused Outer Multiplication

Multiply a row vector to a column vector and take a relu.

Uses two program block axes. Block size `B0` is always less than the vector `x` length `N0`.
Block size `B1` is always less than vector `y` length `N1`.

.. math::
    z_{j, i} = \text{relu}(x_i \times y_j)\text{ for } i = 1\ldots N_0,\ j = 1\ldots N_1
"""


def mul_relu_block_spec(x: Float32[100,], y: Float32[90,]) -> Float32[90, 100]:
    return torch.relu(x[None, :] * y[:, None])


@triton.jit
def mul_relu_block_kernel(
    x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    block_id_x = tl.program_id(0)
    block_id_y = tl.program_id(1)
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N0,),
        strides=(1,),
        offsets=(block_id_x * B0,),
        block_shape=(B0,),
        order=(0,)
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(N1,),
        strides=(1,),
        offsets=(block_id_y * B1,),
        block_shape=(B1,),
        order=(0,)
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(N1, N0),
        strides=(N0, 1),
        offsets=(block_id_y * B1, block_id_x * B0),
        block_shape=(B1, B0),
        order=(0, 1)
    )
    x = tl.load(x_block_ptr,boundary_check=(0,),padding_option='zero')
    y = tl.load(y_block_ptr,boundary_check=(0,),padding_option='zero')
    z = x[None, :] * y[:, None]
    z = tl.maximum(0,z)
    tl.store(z_block_ptr, z, boundary_check=(0,1))
    # Finish me!
    return


r"""
## Puzzle 6: Fused Outer Multiplication - Backwards

Backwards of a function that multiplies a matrix with a row vector and take a relu.

Uses two program blocks. Block size `B0` is always less than the vector `x` length `N0`.
Block size `B1` is always less than vector `y` length `N1`. Chain rule backward `dz`
is of shape `N1` by `N0`

.. math::
    f(x, y) = \text{relu}(x_{j, i} \times y_j)\text{ for } i = 1\ldots N_0,\ j = 1\ldots N_1

.. math::
    dx_{j, i} = f_x'(x, y)_{j, i} \times dz_{j, i}
"""


def mul_relu_block_back_spec(
    x: Float32[90, 100], y: Float32[90,], dz: Float32[90, 100]
) -> Float32[90, 100]:
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = torch.relu(x * y[:, None])
    z.backward(dz)
    dx = x.grad
    return dx


@triton.jit
def mul_relu_block_back_kernel(
    x_ptr, y_ptr, dz_ptr, dx_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    block_id_i = tl.program_id(0)
    block_id_j = tl.program_id(1)
    #x:[N1,N0]
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N1, N0),
        strides=(N0, 1),
        offsets=(block_id_j * B1, block_id_i * B0),
        block_shape=(B1, B0),
        order=(1, 0),
    )
    #y:[N1]
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(N1,),
        strides=(1,),
        offsets=(block_id_j * B1,),
        block_shape=(B1,),
        order=(0,),
    )
    #dz:[N1,N0]
    dz_block_ptr = tl.make_block_ptr(
        base=dz_ptr,
        shape=(N1, N0),
        strides=(N0,1),
        offsets=(block_id_j * B1, block_id_i * B0),
        block_shape=(B1, B0),
        order=(1, 0),
    )
    #dx:[N1,N0]
    dx_block_ptr = tl.make_block_ptr(
        base=dx_ptr,
        shape=(N1, N0),
        strides=(N0, 1),
        offsets=(block_id_j * B1, block_id_i * B0),
        block_shape=(B1, B0),
        order=(1, 0),
    )
    x = tl.load(x_block_ptr,boundary_check=(0, 1),padding_option='zero')
    y = tl.load(y_block_ptr,boundary_check=(0,),padding_option='zero')

    dz = tl.load(dz_block_ptr,boundary_check=(0, 1),padding_option='zero')
    dx = tl.where(x * y[:, None] > 0, y[:, None] * dz, 0)

    tl.store(dx_block_ptr, dx,boundary_check=(0,1))
    # Finish me!
    return


r"""
## Puzzle 7: Long Sum

Sum of a batch of numbers.

Uses one program blocks. Block size `B0` represents a range of batches of  `x` of length `N0`.
Each element is of length `T`. Process it `B1 < T` elements at a time.  

.. math::
    z_{i} = \sum^{T}_j x_{i,j} =  \text{ for } i = 1\ldots N_0

Hint: You will need a for loop for this problem. These work and look the same as in Python.
"""


def sum_spec(x: Float32[4, 200]) -> Float32[4,]:
    return x.sum(1)


@triton.jit
def sum_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    # Finish me!
    tile_id = tl.program_id(0)
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N0, T),
        strides=(T, 1),
        offsets=(tile_id * B0, 0),
        block_shape=(B0, B1),
        order=(1, 0)
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(N0,),
        strides=(1,),
        offsets=(tile_id * B0,),
        block_shape=(B0,),
        order=(0,)
    )
    z = tl.zeros((B0,), dtype=tl.float32)
    for i in range(0, T, B1):
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option='zero')
        z += tl.sum(x,1)
        x_block_ptr = tl.advance(x_block_ptr,(0,B1))
    tl.store(z_block_ptr, z, boundary_check=(0,))
    return


r"""
## Puzzle 8: Long Softmax

Softmax of a batch of logits.

Uses one program block axis. Block size `B0` represents the batch of `x` of length `N0`.
Block logit length `T`.   Process it `B1 < T` elements at a time.  

.. math::
    z_{i, j} = \text{softmax}(x_{i,1} \ldots x_{i, T}) \text{ for } i = 1\ldots N_0

Note softmax needs to be computed in numerically stable form as in Python. In addition in Triton 
they recommend not using `exp` but instead using `exp2`. You need the identity

.. math::
    \exp(x) = 2^{\log_2(e) x}

Advanced: there one way to do this with 3 loops. You can also do it with 2 loops if you are clever. 
Hint: you will find this identity useful:

.. math::
    \exp(x_i - m) =  \exp(x_i - m/2 - m/2) = \exp(x_i - m/ 2) /  \exp(m/2)
"""


def softmax_spec(x: Float32[4, 200]) -> Float32[4, 200]:
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    return x_exp / x_exp.sum(1, keepdim=True)


@triton.jit
def softmax_kernel_brute_force(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    """2 loops ver."""
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N0, T),
        strides=(T, 1),
        offsets=(block_id_i * B0, 0),
        block_shape=(B0, B1),
        order=(1, 0)
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(N0, T),
        strides=(T, 1),
        offsets=(block_id_i * B0, 0),
        block_shape=(B0, B1),
        order=(1, 0)
    )
    z = tl.zeros((B0, B1), dtype=tl.float32)
    x_max = tl.full((B0,), float("-inf"), dtype=tl.float32)
    x_block_ptr1 = x_block_ptr
    for i in range(0,T,B1):
        x = tl.load(x_block_ptr1, boundary_check=(0, 1), padding_option='zero')
        x_max = tl.maximum(x_max, tl.max(x,1))
        x_block_ptr1 = tl.advance(x_block_ptr1,(0,B1))
    sum = tl.zeros((B0,), dtype=tl.float32)
    x_block_ptr1 = x_block_ptr
    for i in range(0,T,B1):
        x = tl.load(x_block_ptr1, boundary_check=(0, 1), padding_option='zero')
        x = x - x_max[:, None]
        x_exp = tl.exp2(log2_e * x)
        sum += tl.sum(x_exp,1)
        x_block_ptr1 = tl.advance(x_block_ptr1,(0,B1))
    for i in range(0,T,B1):
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option='zero')
        x = x - x_max[:, None]
        x_exp = tl.exp2(log2_e * x)
        z = x_exp / sum[:, None]
        tl.store(z_block_ptr, z, boundary_check=(0, 1))
        x_block_ptr = tl.advance(x_block_ptr,(0,B1))
        z_block_ptr = tl.advance(z_block_ptr,(0,B1))

    # Finish me!
    return


@triton.jit
def softmax_kernel(
    x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr
):
    """3 loops ver."""
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N0, T),
        strides=(T, 1),
        offsets=(block_id_i * B0, 0),
        block_shape=(B0, B1),
        order=(1, 0)
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(N0, T),
        strides=(T, 1),
        offsets=(block_id_i * B0, 0),
        block_shape=(B0, B1),
        order=(1, 0)
    )
    z = tl.zeros((B0, B1), dtype=tl.float32)
    x_max = tl.full((B0,), float("-inf"), dtype=tl.float32)
    sum = tl.zeros((B0,), dtype=tl.float32)
    x_block_ptr1 = x_block_ptr
    for i in range(0,T,B1):
        x = tl.load(x_block_ptr1, boundary_check=(0, 1), padding_option='zero')
        new_x_max = tl.maximum(x_max, tl.max(x,1))
        if i == 0:
            x_max = new_x_max
            sum = tl.sum(tl.exp2(log2_e * (x - x_max[:, None])),1)
        else:
            scale = tl.exp2(log2_e * (x_max - new_x_max))
            sum = sum * scale + tl.sum(tl.exp2(log2_e * (x - new_x_max[:, None])),1)
            x_max = new_x_max
        x_block_ptr1 = tl.advance(x_block_ptr1,(0,B1))
    for i in range(0,T,B1):
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option='zero')
        x = x - x_max[:, None]
        x_exp = tl.exp2(log2_e * x)
        z = x_exp / sum[:, None]
        tl.store(z_block_ptr, z, boundary_check=(0, 1))
        x_block_ptr = tl.advance(x_block_ptr,(0,B1))
        z_block_ptr = tl.advance(z_block_ptr,(0,B1))
        # Finish me!
    return


r"""
## Puzzle 9: Simple FlashAttention

A scalar version of FlashAttention.

Uses zero programs. Block size `B0` represent the batches of `q` to process out of `N0`. Sequence length is `T`. Process it `B1 < T` elements (`k`, `v`) at a time for some `B1`.

.. math::
    z_{i} = \sum_{j=1}^{T} \text{softmax}(q_i k_1, \ldots, q_i k_T)_j v_{j} \text{ for } i = 1\ldots N_0

This can be done in 1 loop using a similar trick from the last puzzle.

Hint: Use `tl.where` to mask `q dot k` to -inf to avoid overflow (NaN).
"""


def flashatt_spec(
    q: Float32[200,], k: Float32[200,], v: Float32[200,]
) -> Float32[200,]:
    x = q[:, None] * k[None, :]
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    soft = x_exp / x_exp.sum(1, keepdim=True)
    return (v[None, :] * soft).sum(1)


@triton.jit
def flashatt_kernel(
    q_ptr, k_ptr, v_ptr, z_ptr, N0, T, B0: tl.constexpr, B1: tl.constexpr
):
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504
    myexp = lambda x: tl.exp2(log2_e * x)
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=(N0,),
        strides=(1,),
        offsets=(block_id_i * B0,),
        block_shape=(B0,),
        order=(0,)
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr,
        shape=(T,),
        strides=(1,),
        offsets=(0,),
        block_shape=(B1,),
        order=(0,)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr,
        shape=(T,),
        strides=(1,),
        offsets=(0,),
        block_shape=(B1,),
        order=(0,)
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(N0,),
        strides=(1,),
        offsets=(block_id_i * B0,),
        block_shape=(B0,),
        order=(0,)
    )

    qk_max = tl.full((B0,), float("-inf"), dtype=tl.float32)
    qk_sum = tl.zeros((B0,), dtype=tl.float32)
    o = tl.zeros((B0,), dtype=tl.float32)
    for i in range(0, T, B1):
        q = tl.load(q_block_ptr, boundary_check=(0,), padding_option='zero')
        k = tl.load(k_block_ptr, boundary_check=(0,), padding_option='zero')
        v = tl.load(v_block_ptr, boundary_check=(0,), padding_option='zero')
        qk = q[:, None] * k[None, :]
        new_qk_max = tl.maximum(qk_max,tl.max(qk, 1))
        if i == 0:
            qk_max = new_qk_max
            qk_exp = myexp(qk - qk_max[:,None])
            qk_sum = tl.sum(qk_exp, 1)
            o = tl.sum(qk_exp * v[None, :] / qk_sum[:,None], 1)
        else:
            qk_exp = myexp(qk - new_qk_max[:,None])
            scale = myexp(qk_max - new_qk_max)
            scale1 = scale * qk_sum
            qk_sum = qk_sum * scale + tl.sum(qk_exp, 1)
            o = o * scale1 / qk_sum + tl.sum(qk_exp * v[None, :] / qk_sum[:,None], 1)
            qk_max = new_qk_max
        k_block_ptr = tl.advance(k_block_ptr,(B1,))
        v_block_ptr = tl.advance(v_block_ptr,(B1,))
    tl.store(z_block_ptr, o, boundary_check=(0,))
    # Finish me!
    return


r"""
## Puzzle 10: Two Dimensional Convolution

A batched 2D convolution.

Uses one program id axis. Block size `B0` represent the batches to process out of `N0`.
Image `x` is size is `H` by `W` with only 1 channel, and kernel `k` is size `KH` by `KW`.

.. math::
    z_{i, j, l} = \sum_{oj, ol}^{j+oj\le H, l+ol\le W} k_{oj,ol} \times x_{i,j + oj, l + ol} 
    \text{ for } i = 1\ldots N_0 \text{ for } j = 1\ldots H \text{ for } l = 1\ldots W
"""


def conv2d_spec(x: Float32[4, 8, 8], k: Float32[4, 4]) -> Float32[4, 8, 8]:
    z = torch.zeros(4, 8, 8)
    x = torch.nn.functional.pad(x, (0, 4, 0, 4, 0, 0), value=0.0)
    # print(x.shape, k.shape)
    for i in range(8):
        for j in range(8):
            z[:, i, j] = (k[None, :, :] * x[:, i : i + 4, j : j + 4]).sum(1).sum(1)
    return z


@triton.jit
def conv2d_kernel(
    x_ptr, k_ptr, z_ptr, N0, H, W, KH: tl.constexpr, KW: tl.constexpr, B0: tl.constexpr
):
    block_id_i = tl.program_id(0)
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr,
        shape=(KH, KW),
        strides=(KW, 1),
        offsets=(0, 0),
        block_shape=(KH, KW),
        order=(1, 0),
    )
    k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")

    x_row_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N0, H, W),
        strides=(H * W, W, 1),
        offsets=(block_id_i * B0, 0, 0),
        block_shape=(B0, KH, KW),
        order=(2, 1, 0),
    )
    z_row_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(N0, H, W),
        strides=(H * W, W, 1),
        offsets=(block_id_i * B0, 0, 0),
        block_shape=(B0, 1, 1),
        order=(2, 1, 0),
    )

    for _ in range(H):
        x_block_ptr = x_row_block_ptr
        z_block_ptr = z_row_block_ptr
        for _ in range(W):
            x = tl.load(x_block_ptr, boundary_check=(0, 1, 2), padding_option="zero")
            z = (x * k[None, :, :]).sum(1).sum(1)
            tl.store(z_block_ptr, z[:, None, None], boundary_check=(0, 1, 2))
            x_block_ptr = tl.advance(x_block_ptr, (0, 0, 1))
            z_block_ptr = tl.advance(z_block_ptr, (0, 0, 1))
        x_row_block_ptr = tl.advance(x_row_block_ptr, (0, 1, 0))
        z_row_block_ptr = tl.advance(z_row_block_ptr, (0, 1, 0))

    return


r"""
## Puzzle 11: Matrix Multiplication

A blocked matrix multiplication.

Uses three program id axes. Block size `B2` represent the batches to process out of `N2`.
Block size `B0` represent the rows of `x` to process out of `N0`. Block size `B1` represent the cols 
of `y` to process out of `N1`. The middle shape is `MID`.

.. math::
    z_{i, j, k} = \sum_{l} x_{i,j, l} \times y_{i, l, k} \text{ for } i = 1\ldots N_2, j = 1\ldots N_0, k = 1\ldots N_1

You are allowed to use `tl.dot` which computes a smaller mat mul.

Hint: the main trick is that you can split a matmul into smaller parts.

.. math::
    z_{i, j, k} = \sum_{l=1}^{L/2} x_{i,j, l} \times y_{i, l, k} +  \sum_{l=L/2}^{L} x_{i,j, l} \times y_{i, l, k}
"""


def dot_spec(x: Float32[4, 32, 32], y: Float32[4, 32, 32]) -> Float32[4, 32, 32]:
    return x @ y


@triton.jit
def dot_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    N0,
    N1,
    N2,
    MID,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B_MID: tl.constexpr,
):
    block_id_j = tl.program_id(0)
    block_id_k = tl.program_id(1)
    block_id_i = tl.program_id(2)
    # Finish me!
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N2,N0,MID),
        strides=(N0 * MID,MID,1),
        offsets=(block_id_i * B2, block_id_j * B0, 0),
        block_shape=(B2, B0, B_MID),
        order=(0, 1, 2),
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(N2,MID,N1),
        strides=(MID * N1,N1,1),
        offsets=(block_id_i * B2, 0, block_id_k * B1),
        block_shape=(B2, B_MID, B1),
        order=(0, 1, 2),
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(N2,N0,N1),
        strides=(N0 * N1,N1,1),
        offsets=(block_id_i * B2, block_id_j * B0, block_id_k * B1),
        block_shape=(B2, B0, B1),
        order=(0, 1, 2),
    )
    z = tl.zeros((B2, B0, B1), dtype=tl.float32)
    for i in range(0, MID, B_MID):
        x = tl.load(x_block_ptr, boundary_check=(0,1,2),padding_option='zero')
        y = tl.load(y_block_ptr, boundary_check=(0,1,2),padding_option='zero')
        z += tl.dot(x,y)
        x_block_ptr = tl.advance(x_block_ptr,(0,0,B_MID))
        y_block_ptr = tl.advance(y_block_ptr,(0,B_MID,0))
    tl.store(z_block_ptr, z, boundary_check=(0,1,2))
    return


r"""
## Puzzle 12: Quantized Matrix Mult

When doing matrix multiplication with quantized neural networks a common strategy is to store the weight matrix in lower precision, with a shift and scale term.

For this problem our `weight` will be stored in 4 bits. We can store `FPINT` of these in a 32 bit integer. In addition for every `group` weights in order we will store 1 `scale` float value and 1 `shift` 4 bit value. We store these for the column of weight. The `activation`s are stored separately in standard floats.

Mathematically it looks like.

.. math::
    z_{j, k} = \sum_{l} sc_{j, \frac{l}{g}} (w_{j, l} - sh_{j, \frac{l}{g}}) \times y_{l, k} 
    \text{ for } j = 1\ldots N_0, k = 1\ldots N_1

Where `g` is the number of groups (`GROUP`).

However, it is a bit more complex since we need to also extract the 4-bit values into floats to begin.

Note:
- We don't consider batch size, i.e. `i`, in this puzzle.
- Remember to unpack the `FPINT` values into separate 4-bit values. This contains some shape manipulation.
"""

FPINT = 32 // 4
GROUP = 8


def quant_dot_spec(
    scale: Float32[32, 8],
    offset: Int32[32,],
    weight: Int32[32, 8],
    activation: Float32[64, 32],
) -> Float32[32, 32]:
    offset = offset.view(32, 1)

    def extract(x):
        over = torch.arange(8) * 4
        mask = 2**4 - 1
        return (x[..., None] >> over) & mask

    scale = scale[..., None].expand(-1, 8, GROUP).contiguous().view(-1, 64)
    offset = (
        extract(offset)[..., None].expand(-1, 1, 8, GROUP).contiguous().view(-1, 64)
    )
    return (scale * (extract(weight).view(-1, 64) - offset)) @ activation


@triton.jit
def quant_dot_kernel(
    scale_ptr,
    offset_ptr,
    weight_ptr,
    activation_ptr,
    z_ptr,
    N0,
    N1,
    MID,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B_MID: tl.constexpr,
):
    block_id_j = tl.program_id(0)
    block_id_k = tl.program_id(1)

    def extract(x):
        over = tl.arange(0,8) * 4
        mask = 2**4 - 1
        return (x[:,:, None] >> over) & mask

    activation_block_ptr = tl.make_block_ptr(
        base = activation_ptr,
        shape = (MID,N1),
        strides = (N1,1),
        offsets = (0, block_id_k * B1),
        block_shape = (B_MID, B1),
        order = (0,1)
    )
    weight_block_ptr = tl.make_block_ptr(
        base = weight_ptr,
        shape = (N0,MID // FPINT),
        strides = (MID // FPINT,1),
        offsets = (block_id_j * B0, 0),
        block_shape = (B0, B_MID // FPINT),
        order = (1,0)
    )
    scale_block_ptr = tl.make_block_ptr(
        base = scale_ptr,
        shape = (N0, GROUP),
        strides = (GROUP,1),
        offsets = (block_id_j * B0, 0),
        block_shape = (B0, GROUP),
        order = (1,0)
    )
    offset_block_ptr = tl.make_block_ptr(
        base = offset_ptr,
        shape = (N0,GROUP // FPINT),
        strides = (GROUP // FPINT,1),
        offsets = (block_id_j * B0, 0),
        block_shape = (B0, GROUP // FPINT),
        order = (1,0)
    )
    z_block_ptr = tl.make_block_ptr(
        base = z_ptr,
        shape = (N0,N1),
        strides = (N1,1),
        offsets = (block_id_j * B0, block_id_k * B1),
        block_shape = (B0, B1),
        order = (1,0)
    )
    z = tl.zeros((B0, B1), dtype=tl.float32)
    for i in range(0, MID, B_MID):
        scale_fp32 = tl.load(scale_block_ptr, boundary_check=(0,1), padding_option='zero')#[B0,GROUP]
        
        offset_int32 = extract(tl.load(offset_block_ptr, boundary_check=(0,1), padding_option='zero'))#[B0, GROUP // FPINT, FPINT]
        offset_int32 = offset_int32.reshape(B0, GROUP)#[B0, GROUP]
        
        weight_int32 = extract(tl.load(weight_block_ptr, boundary_check=(0,1), padding_option='zero'))#[B0, B_MID // FPINT, FPINT]
        weight_int32 = weight_int32.reshape(B0, B_MID)#[B0, B_MID]
        weight_int32 = weight_int32.reshape(B0,GROUP,B_MID // GROUP)#[B0, GROUP, B_MID // GROUP]

        activation_fp32 = tl.load(activation_block_ptr, boundary_check=(0,1), padding_option='zero')
        
        weight_fp32 = scale_fp32[:,:,None] * (weight_int32 - offset_int32[:,:,None])
        weight_fp32 = weight_fp32.reshape(B0,B_MID)

        z += tl.dot(weight_fp32, activation_fp32)
        scale_block_ptr = tl.advance(scale_block_ptr,(0, GROUP))
        offset_block_ptr = tl.advance(offset_block_ptr,(0, GROUP // FPINT))
        weight_block_ptr = tl.advance(weight_block_ptr,(0, B_MID // FPINT))
        activation_block_ptr = tl.advance(activation_block_ptr,(B_MID, 0))
    tl.store(z_block_ptr, z, boundary_check=(0,1))

    # Finish me!
    return


def run_demos():
    run_demo1()
    run_demo2()
    run_demo3()
    run_demo4()


def run_puzzles(args, puzzles: List[int]):
    print_log = args.log
    device = args.device

    if 1 in puzzles:
        print("Puzzle #1:")
        ok = test(
            add_kernel,
            add_spec,
            nelem={"N0": 32},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 2 in puzzles:
        print("Puzzle #2:")
        ok = test(
            add_mask2_kernel,
            add2_spec,
            nelem={"N0": 200},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 3 in puzzles:
        print("Puzzle #3:")
        ok = test(
            add_vec_kernel,
            add_vec_spec,
            nelem={"N0": 32, "N1": 32},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 4 in puzzles:
        print("Puzzle #4:")
        ok = test(
            add_vec_block_kernel,
            add_vec_block_spec,
            nelem={"N0": 100, "N1": 90},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 5 in puzzles:
        print("Puzzle #5:")
        ok = test(
            mul_relu_block_kernel,
            mul_relu_block_spec,
            nelem={"N0": 100, "N1": 90},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 6 in puzzles:
        print("Puzzle #6:")
        ok = test(
            mul_relu_block_back_kernel,
            mul_relu_block_back_spec,
            nelem={"N0": 100, "N1": 90},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 7 in puzzles:
        print("Puzzle #7:")
        ok = test(
            sum_kernel,
            sum_spec,
            B={"B0": 1, "B1": 32},
            nelem={"N0": 4, "N1": 32, "T": 200},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 8 in puzzles:
        print("Puzzle #8:")
        ok = test(
            softmax_kernel,
            softmax_spec,
            B={"B0": 1, "B1": 32},
            nelem={"N0": 4, "N1": 32, "T": 200},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 9 in puzzles:
        print("Puzzle #9:")
        ok = test(
            flashatt_kernel,
            flashatt_spec,
            B={"B0": 64, "B1": 32},
            nelem={"N0": 200, "T": 200},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 10 in puzzles:
        print("Puzzle #10:")
        ok = test(
            conv2d_kernel,
            conv2d_spec,
            B={"B0": 1},
            nelem={"N0": 4, "H": 8, "W": 8, "KH": 4, "KW": 4},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 11 in puzzles:
        print("Puzzle #11:")
        ok = test(
            dot_kernel,
            dot_spec,
            B={"B0": 16, "B1": 16, "B2": 1, "B_MID": 16},
            nelem={"N0": 32, "N1": 32, "N2": 4, "MID": 32},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 12 in puzzles:
        print("Puzzle #12:")
        ok = test(
            quant_dot_kernel,
            quant_dot_spec,
            B={"B0": 16, "B1": 16, "B_MID": 64},
            nelem={"N0": 32, "N1": 32, "MID": 64},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    print("All tests passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--puzzle", type=int, metavar="N", help="Run Puzzle #N")
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Run all Puzzles. Stop at first failure.",
    )
    parser.add_argument("-l", "--log", action="store_true", help="Print log messages.")
    parser.add_argument(
        "-i",
        "--intro",
        action="store_true",
        help="Run all demos in the introduction part.",
    )

    args = parser.parse_args()

    if os.getenv("TRITON_INTERPRET", "0") == "1":
        torch.set_default_device("cpu")
        args.device = "cpu"
    else:  # GPU mode
        torch.set_default_device("cuda")
        args.device = "cuda"

    if args.intro:
        run_demos()
    elif args.all:
        run_puzzles(args, list(range(1, 13)))
    elif args.puzzle:
        run_puzzles(args, [int(args.puzzle)])
    else:
        parser.print_help()
