from typing import Callable
import torch
from torch import Tensor
import torch.nn.functional as F


def pad_at_dim(t, pad, dim = -1, value = 0.):
    """
    在指定维度对张量进行填充。

    参数:
        t (torch.Tensor): 需要填充的输入张量。
        pad (Tuple[int, int]): 填充的宽度，一个包含两个整数的元组，分别表示在指定维度的起始和结束位置填充的数量。
        dim (int, 可选): 指定要填充的维度。默认为最后一个维度（-1）。
        value (float, 可选): 用于填充的值，默认为0。

    返回:
        torch.Tensor: 填充后的张量。
    """
    # 计算从右侧开始计数的维度索引
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)

    # 创建一个包含填充信息的元组，格式为 ((before_1, after_1), (before_2, after_2), ..., (before_n, after_n))
    # 这里只对指定维度进行填充，其他维度填充为0
    zeros = ((0, 0) * dims_from_right)

    # 调用 F.pad 进行填充
    return F.pad(t, (*zeros, *pad), value = value)


@torch.jit.script
def binary_operator(
    a: tuple[Tensor, Tensor],
    b: tuple[Tensor, Tensor]
):
    """
    二元操作符，用于 associative_scan 函数。

    该操作符对输入的两个元组进行操作：
    1. 对第一个张量执行逐元素相乘。
    2. 对第二个张量执行逐元素累加乘积（addcmul）。

    参数:
        a (Tuple[torch.Tensor, torch.Tensor]): 第一个输入元组，包含两个张量。
        b (Tuple[torch.Tensor, torch.Tensor]): 第二个输入元组，包含两个张量。

    返回:
        Tuple[torch.Tensor, torch.Tensor]: 操作后的结果元组，包含两个张量。
    """
    # 解包第一个输入元组
    a_i, kv_i = a
    # 解包第二个输入元组
    a_j, kv_j = b
    # 返回操作后的结果元组
    return a_j * a_i, torch.addcmul(kv_j, a_j, kv_i)


def associative_scan(
    operator: Callable,
    elems: tuple[Tensor, Tensor]
):
    """
    对输入的元组执行关联扫描（associative scan）。

    该函数实现了类似于 JAX 的 lax.associative_scan 的功能，专门用于处理序列建模中的 token 序列。

    参数:
        operator (Callable): 二元操作符函数，接受两个输入元组并返回一个输出元组。
        elems (Tuple[torch.Tensor, torch.Tensor]): 输入的元组，包含两个张量，形状为 (batch_size, sequence_length, ...)。

    返回:
        Tuple[torch.Tensor, torch.Tensor]: 扫描后的结果元组，包含两个张量。
    """
    # 获取序列长度
    num_elems = int(elems[0].shape[1])

    # 检查所有输入张量的第一个维度是否相同
    if not all(int(elem.shape[1]) == num_elems for elem in elems[1:]):
        raise ValueError('Array inputs to associative_scan must have the same '
                         'first dimension. (saw: {})'
                         .format([elem.shape for elem in elems]))

    def _scan(elems):
        """
        对输入的元组执行扫描操作。

        参数:
            elems (Tuple[torch.Tensor, torch.Tensor]): 输入的元组，包含两个张量。

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 扫描后的结果元组，包含两个张量。
        """
        # 获取序列长度
        num_elems = elems[0].shape[1]

        if num_elems < 2:
            # 如果序列长度小于2，直接返回输入
            return elems

        # 将相邻的元素对进行合并（reduce）
        reduced_elems = operator(
          [elem[:, :-1:2] for elem in elems], # 选择偶数索引的元素
          [elem[:, 1::2] for elem in elems]) # 选择奇数索引的元素

        # 递归地对部分合并后的张量执行扫描操作
        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            # 如果序列长度为偶数，则将奇数索引的扫描结果与原始偶数索引的元素合并
            even_elems = operator(
                [e[:, :-1] for e in odd_elems], # 选择奇数索引扫描结果的偶数索引元素
                [e[:, 2::2] for e in elems]) # 选择原始元素的偶数索引元素
        else:
            # 如果序列长度为奇数，则将奇数索引的扫描结果与原始偶数索引的元素合并
            even_elems = operator( 
                odd_elems,  # 使用奇数索引的扫描结果
                [e[:, 2::2] for e in elems])  # 选择原始元素的偶数索引元素

        # 将扫描结果的第一个元素替换为原始元素的第一个元素
        even_elems = [
          torch.cat([elem[:, :1], result], dim=1)
          for (elem, result) in zip(elems, even_elems)]

        # 将偶数索引和奇数索引的扫描结果交替合并
        return list(map(_interleave, even_elems, odd_elems))

    # 执行扫描操作并返回结果
    return _scan(elems)


def _interleave(a, b):
    """
    将两个张量交替合并。

    参数:
        a (torch.Tensor): 第一个输入张量。
        b (torch.Tensor): 第二个输入张量。

    返回:
        torch.Tensor: 交替合并后的张量。
    """
    # 获取两个张量在指定维度的长度
    a_axis_len, b_axis_len = a.shape[1], b.shape[1]
    # 计算输出张量的长度
    output_axis_len = a_axis_len + b_axis_len

    if (a_axis_len == (b_axis_len + 1)):
        # 如果第一个张量的长度比第二个张量长1，则对第二个张量进行填充
        b = pad_at_dim(b, (0, 1), dim = 1)

    # 将两个张量在指定维度上堆叠
    stacked = torch.stack([a, b], dim=2)
    # 将堆叠后的张量在指定维度上交替合并
    interleaved = torch.flatten(stacked, start_dim=1, end_dim=2)

    # 返回交替合并后的张量，截取到所需的长度
    return interleaved[:, :output_axis_len]
