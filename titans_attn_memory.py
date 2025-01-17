import math
from functools import partial
import einx
from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange, Reduce

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch.func import functional_call, vmap, grad
from tensordict import TensorDict

from associative_scan import associative_scan, binary_operator, pad_at_dim


# 使用 partial 创建一个不带偏置的线性层函数
LinearNoBias = partial(Linear, bias = False)


"""
ein 符号说明：
b - 批处理大小（batch）
n - 序列长度（sequence）
d - 特征维度（feature dimension）
c - 块内维度（intra-chunk）
"""


def exists(v):
    """
    检查变量是否存在（不为 None）。

    参数:
        v: 任意变量。

    返回:
        bool: 如果 v 不为 None，则返回 True，否则返回 False。
    """
    return v is not None


def default(v, d):
    """
    如果变量存在（不为 None），则返回变量本身；否则返回默认值。

    参数:
        v: 任意变量。
        d: 默认值。

    返回:
        任意类型: 如果 v 存在，则返回 v；否则返回 d。
    """
    return v if exists(v) else d


def round_down_multiple(seq, mult):
    """
    将一个整数向下取整到最接近的指定倍数的整数。

    参数:
        seq (int): 要取整的整数。
        mult (int): 倍数。

    返回:
        int: 向下取整后的整数。
    """
    return seq // mult * mult


def round_up_multiple(seq, mult):
    """
    将一个整数向上取整到最接近的指定倍数的整数。

    参数:
        seq (int): 要取整的整数。
        mult (int): 倍数。

    返回:
        int: 向上取整后的整数。
    """
    return math.ceil(seq / mult) * mult


def pack_one_with_inverse(t, pattern):
    """
    将一个张量按照指定模式打包，并返回一个用于解包的逆函数。

    参数:
        t (torch.Tensor): 要打包的张量。
        pattern (Tuple[int, ...]): 打包模式，指定每个维度如何分割。

    返回:
        Tuple[torch.Tensor, Callable]: 打包后的张量和一个用于解包的逆函数。
    """
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, packed_shape, inv_pattern)[0]

    return packed, inverse


class MemoryAttention(Module):
    """
    临时注意力作为内存模块。

    该模块使用自注意力机制作为内存访问方式，通过查询、键和值权重来计算隐藏表示。
    """
    def __init__(
        self,
        dim
    ):
        """
        初始化 MemoryAttention 模块。

        参数:
            dim (int): 特征的维度。
        """
        super().__init__()
        # 定义四个可学习的权重参数，分别对应查询（queries）、键（keys）、值权重1（values weight 1）和值权重2（values weight 2）
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(dim, dim)), # queries
            nn.Parameter(torch.randn(dim, dim)), # keys
            nn.Parameter(torch.randn(dim, dim)), # values weight 1
            nn.Parameter(torch.randn(dim, dim)), # values weight 2
        ])

    def forward(self, x):
        """
        前向传播方法。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, dim)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, sequence_length, dim)。
        """
        assert x.shape[-2] > 1, 'chunk size needs to be greater than 1 for using attention as memory'

        # 解包权重参数
        wq, wk, wv1, wv2 = self.weights

        # 计算查询、键和值，形状为 (b, n, d)
        q = x @ wq
        k = x @ wk
        v = x @ wv1

        # 使用缩放点积注意力机制计算隐藏表示
        # is_causal=True 表示使用因果掩码，确保当前时间步只能看到过去的时间步
        hidden = F.scaled_dot_product_attention(
            q, k, v,
            is_causal = True
        )

        # 对隐藏表示应用 SiLU 激活函数，并乘以值权重2，得到最终输出
        return F.silu(hidden) @ wv2  # 输出形状为 (b, n, d)


def default_loss_fn(pred, target):
    """
    默认的损失函数：均方误差（MSE）损失。

    参数:
        pred (Tensor): 模型的预测值，形状为 (batch_size, ...)。
        target (Tensor): 目标值，形状与 pred 相同。

    返回:
        Tensor: 计算得到的损失值，形状为 (batch_size,)。
    """
    return (pred - target).pow(2).mean(dim = -1).sum()


class NeuralMemory(Module):
    """
    神经记忆模块（Neural Memory Module）。

    该模块通过记忆机制增强模型的能力，允许模型在处理序列数据时存储和检索信息。
    记忆机制基于 MemoryAttention 模型，并使用梯度下降来更新记忆内容。
    """
    def __init__(
        self,
        dim,
        chunk_size = 1,
        dim_head = None,
        heads = 1,
        model: MemoryAttention | None = None,
        store_memory_loss_fn = default_loss_fn,
        pre_rmsnorm = True,
        post_rmsnorm = True,
        use_accelerated_scan = False,
        default_model_kwargs: dict = dict()
    ):
        """
        初始化神经记忆模块。

        参数:
            dim (int): 特征的维度。
            chunk_size (int, 可选): 块大小，用于分组处理，默认为1。
            dim_head (int, 可选): 每个注意力头的维度，默认为 None，表示与 dim 相同。
            heads (int, 可选): 注意力头的数量，默认为1。
            model (MemoryAttention, 可选): 记忆模型，默认为 None。如果为 None，则使用默认的 MemoryAttention 模型。
            store_memory_loss_fn (Callable, 可选): 用于存储记忆的损失函数，默认为均方误差损失。
            pre_rmsnorm (bool, 可选): 是否在存储和检索之前应用 RMSNorm，默认为 True。
            post_rmsnorm (bool, 可选): 是否在存储和检索之后应用 RMSNorm，默认为 True。
            use_accelerated_scan (bool, 可选): 是否使用加速扫描，默认为 False。
            default_model_kwargs (dict, 可选): 传递给记忆模型的默认关键字参数，默认为空字典。
        """
        super().__init__()

        # 定义归一化层
        # 检索前的归一化
        self.retrieve_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        # 存储前的归一化
        self.store_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        # 检索或存储后的归一化
        self.post_rmsnorm = nn.RMSNorm(dim) if post_rmsnorm else nn.Identity()

        # 处理多头注意力
        # 如果 dim_head 未指定，则默认为 dim
        dim_head = default(dim_head, dim)
        # 计算内部维度
        dim_inner = dim_head * heads

        # 将张量重塑为多头形式
        self.split_heads = Rearrange('b n (h d) -> (b h) n d', h = heads)
        # 将多头张量合并回原始形状
        self.merge_heads = Rearrange('(b h) n d -> b n (h d)', h = heads)
        # 如果有多头，则使用线性层合并多头；否则，使用恒等映射
        self.combine_heads = LinearNoBias(dim_inner, dim) if heads > 1 else nn.Identity()

        # 初始化记忆模型
        if not exists(model):
            # 如果未提供模型，则使用默认的 MemoryAttention 模型
            model = MemoryAttention(dim_head, **default_model_kwargs)

        assert not exists(next(model.buffers(), None)), 'model cannot have buffers for now'

        # 将模型赋值给记忆模型
        self.memory_model = model

        # 保存块大小
        self.chunk_size = chunk_size

        # 定义前向传播和损失计算函数
        def forward_and_loss(params, inputs, target):
            # 使用参数 params 调用记忆模型进行前向传播
            pred = functional_call(self.memory_model, params, inputs)
            # 计算损失，默认为均方误差损失
            loss = self.store_memory_loss_fn(pred, target) # simple mse loss in paper - eq (12) - |M(k) - v|²
            return loss

        # 对每个样本计算梯度
        self.per_sample_grad_fn = vmap(grad(forward_and_loss), in_dims = (None, 0, 0))

        # 定义用于检索的查询线性层
        self.to_queries = LinearNoBias(dim, dim_inner)

        # 定义用于存储的键和值线性层
        self.to_keys_values = LinearNoBias(dim, dim_inner * 2)
        # 保存损失函数
        self.store_memory_loss_fn = store_memory_loss_fn

        # 定义用于计算自适应学习率和动量的模块
        self.to_momentum = nn.Sequential(
            Reduce('b (n c) ... -> b n ...', 'mean', c = chunk_size),  # 对块内维度进行平均
            LinearNoBias(dim, heads),  # 线性层，将维度映射到注意力头的数量
            Rearrange('b n h -> (b h) n 1')  # 重塑张量形状
        )

        self.to_adaptive_step = nn.Sequential(
            Reduce('b (n c) ... -> b n ...', 'mean', c = chunk_size),  # 对块内维度进行平均
            LinearNoBias(dim, heads),  # 线性层，将维度映射到注意力头的数量
            Rearrange('b n h -> (b h) n')  # 重塑张量形状
        )

        # 定义用于计算权重衰减因子的模块
        self.to_decay_factor = nn.Sequential(
            Reduce('b (n c) ... -> b n ...', 'mean', c = chunk_size),  # 对块内维度进行平均
            LinearNoBias(dim, heads),  # 线性层，将维度映射到注意力头的数量
            Rearrange('b n h -> (b h) n 1')  # 重塑张量形状
        )

        # 是否使用加速扫描
        self.use_accelerated_scan = use_accelerated_scan

    def init_weights_and_momentum(self):
        """
        初始化记忆模型的权重和动量。

        返回:
            Tuple[TensorDict, TensorDict]: 初始化的权重和动量，分别为 TensorDict 对象。
        """
        # 获取记忆模型的所有参数，并将其转换为 TensorDict 对象
        params = TensorDict(dict(self.memory_model.named_parameters()))

        # 初始化权重为零张量
        init_weights = params.clone().zero_()
        # 初始化动量为零张量
        init_momentum = params.clone().zero_()

        # 返回初始化的权重和动量
        return init_weights, init_momentum

    def store_memories(
        self,
        seq,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]]
    ):
        """
        存储记忆并更新记忆模型的权重和动量。

        参数:
            seq (torch.Tensor): 输入序列，形状为 (batch_size, seq_len, dim)。
            past_state (Tuple[Dict[str, Tensor], Dict[str, Tensor]]): 过去的状态，包含权重和动量。

        返回:
            Tuple[Dict[str, Tensor], Tuple[Dict[str, Tensor], Dict[str, Tensor]]]: 更新后的权重和动量，以及新的状态。
        """
        # 对输入序列应用存储前的归一化
        seq = self.store_norm(seq)

        # 计算序列长度和块大小
        seq_len, chunk_size = seq.shape[-2], self.chunk_size
        # 将序列长度向下取整到块大小的倍数，确保每个块完整
        round_down_seq_len = round_down_multiple(seq_len, self.chunk_size)

        # 截断序列，使其长度为块大小的倍数
        seq = seq[:, :round_down_seq_len]

        # 获取当前记忆模型的权重
        curr_weights = TensorDict(dict(self.memory_model.named_parameters()))

        # 将过去的状态转换为 TensorDict 对象
        past_state = tuple(TensorDict(d) for d in past_state)
        past_weights, past_momentum = past_state

        # 将当前权重与过去权重相加
        curr_weights = curr_weights + past_weights

        # 打包批次和序列维度
        # 这里的 'b' 表示批次，'n' 表示序列长度，'c' 表示块内维度

        # 计算自适应学习率：
        # 1. 对输入序列应用自适应步长模块（to_adaptive_step），得到形状为 (batch, n, heads) 的张量。
        # 2. 使用 sigmoid 函数将其值压缩到 (0, 1) 之间。
        # 3. 乘以 -15 并取指数，将值映射到 (1e-7, 1) 之间。
        adaptive_lr = (self.to_adaptive_step(seq).sigmoid() * -15).exp() # 学习率范围从 1 到 1e-7

        # 计算自适应动量：
        # 对输入序列应用动量模块（to_momentum），然后使用 sigmoid 函数将其值压缩到 (0, 1) 之间。
        adaptive_momentum = self.to_momentum(seq).sigmoid()

        # 计算权重衰减因子：
        # 对输入序列应用衰减因子模块（to_decay_factor），然后使用 sigmoid 函数将其值压缩到 (0, 1) 之间。
        decay_factor = self.to_decay_factor(seq).sigmoid()

        # 分离键和值：
        # 对输入序列应用键值模块（to_keys_values），然后将其在最后一个维度上分割成两部分，分别作为键和值。
        keys, values = self.to_keys_values(seq).chunk(2, dim = -1)

        # 处理多头：
        # 对键和值应用多头重塑（split_heads），将批次和头数维度合并。
        keys, values = map(self.split_heads, (keys, values))

        # 获取批次大小
        batch = keys.shape[0]

        # 处理块：
        # 将键和值在序列维度上重塑为 (batch * n, c, d)，其中 c 是块内维度，d 是特征维度。
        keys, values = tuple(rearrange(t, 'b (n c) d -> (b n) c d', c = self.chunk_size) for t in (keys, values))

        # 计算梯度并计算辅助损失：
        # 使用 per_sample_grad_fn 计算每个样本的梯度，传入当前权重、键和值。
        grads = self.per_sample_grad_fn(dict(curr_weights), keys, values)

        # 将梯度转换为 TensorDict 对象
        grads = TensorDict(grads)

        # 恢复批次和序列维度：
        # 将梯度张量从 (batch * n, ...) 重塑为 (batch, n, ...)。
        grads = grads.apply(lambda t: rearrange(t, '(b n) ... -> b n ...', b = batch))

        # 乘以自适应学习率：
        # 对每个梯度张量应用乘法操作，将梯度乘以负的学习率。
        surprises = grads.apply(lambda t: einx.multiply('b n ..., b n -> b n ...', t, -adaptive_lr))

        # 定义默认的关联扫描函数：
        # 使用 associative_scan 和 binary_operator 对输入的 gates 和 inputs 进行扫描。
        def default_associative_scan(gates, inputs):
            _, outputs = associative_scan(binary_operator, (gates, inputs))
            return outputs

        # 如果使用加速扫描
        if self.use_accelerated_scan:
            # 从 triton 和 accelerated_scan 模块导入扫描函数
            from triton import scan as triton_scan
            from accelerated_scan import scan as warp_scan

            # 根据设备选择扫描函数
            scan = triton_scan if seq.is_cuda else warp_scan

            # 定义加速扫描函数：
            # 1. 对 gates 和 inputs 进行扩展和重塑。
            # 2. 对序列长度进行填充，使其为2的幂。
            # 3. 调用扫描函数。
            # 4. 截取填充后的结果，并恢复原始形状。
            def accelerate_scan_fn(gates, inputs):
                gates = gates.expand_as(inputs)
                gates, inputs = tuple(rearrange(t, 'b n d -> b d n') for t in (gates, inputs))

                seq_len = gates.shape[-1]
                next_power_two_seq_len = 2 ** max(5, int(math.ceil(math.log2(seq_len))))

                gates = F.pad(gates, (0, next_power_two_seq_len - seq_len))
                inputs = F.pad(inputs, (0, next_power_two_seq_len - seq_len))

                outputs = scan(gates, inputs)

                outputs = outputs[..., :seq_len]
                outputs = rearrange(outputs, 'b d n -> b n d')
                return outputs

            scan_fn = accelerate_scan_fn
        else:
            scan_fn = default_associative_scan

        # momentum + weight decay - momentum is the new contribution, as most linear RNNs have learned forgetting gates
        # 计算动量和更新：
        # 1. 对每个参数名和对应的惊喜（surprise）进行迭代。
        # 2. 使用 pack_one_with_inverse 对惊喜进行打包，并获取逆函数。
        # 3. 使用 scan_fn 计算动量。
        # 4. 再次使用 scan_fn 计算更新（考虑权重衰减）。
        # 5. 将更新和动量逆打包，并存储到 updates 和 next_momentum 中。
        next_momentum = TensorDict()
        updates = TensorDict()

        for param_name, surprise in surprises.items():

            surprise, inverse_pack = pack_one_with_inverse(surprise, 'b n *')

            # 计算动量：
            # 使用关联扫描函数，根据自适应动量和惊喜计算动量。
            momentum = scan_fn(adaptive_momentum, surprise) # momentum is S / surprise in the paper

            # use associative scan again for learned forgetting (weight decay) - eq (13)
            # 计算更新：
            # 使用关联扫描函数，根据权重衰减因子和动量计算更新。
            update = scan_fn(1. - decay_factor, momentum) # momentum is S / surprise in the paper

            updates[param_name] = inverse_pack(update)
            next_momentum[param_name] = inverse_pack(momentum)

        # compute the next weight per batch
        # 计算每个批次的下一个权重：
        # 对每个参数，获取最后一个更新，并将其添加到当前权重中。
        last_update = updates.apply(lambda t: t[:, -1])

        next_state = (curr_weights + last_update, next_momentum)
        
        # 返回更新后的权重和动量，以及新的状态
        return updates, next_state

    def retrieve_memories(
        self,
        seq,
        past_weights: dict[str, Tensor] | None = None,
    ):
        """
        从记忆中检索信息。

        参数:
            seq (torch.Tensor): 输入序列，形状为 (batch_size, seq_len, dim)。
            past_weights (Dict[str, Tensor], 可选): 过去的权重，用于记忆检索。

        返回:
            torch.Tensor: 检索到的记忆，形状为 (batch_size, seq_len - chunk_size, dim)。
        """
        # 获取块大小
        chunk_size = self.chunk_size
        # 获取序列长度
        seq_len = seq.shape[1]

        # 对输入序列应用检索前的归一化
        seq = self.retrieve_norm(seq)

        assert seq_len > chunk_size

        # 截取序列，从第 chunk_size 个时间步开始
        seq = seq[:, chunk_size:]
        # 获取截取后的序列长度
        curtailed_seq_len = seq.shape[-2]

        # 计算下一个块的序列长度，向上取整到块大小的倍数
        next_seq_len = round_up_multiple(curtailed_seq_len + 1, chunk_size)

        # 计算需要填充的长度
        padding = next_seq_len - curtailed_seq_len

        # 在序列维度上填充，填充值为0
        seq = pad_at_dim(seq, (0, padding), dim = 1)

        # 记忆模型的参数存储了键/值的记忆
        # 当 MLP 只有1个权重矩阵时，它等同于线性注意力文献中的 `kv` 快速权重记忆（回忆记忆的获取是 q @ (kv)）

        # 获取当前记忆模型的权重
        curr_weights = TensorDict(dict(self.memory_model.named_parameters()))

        if exists(past_weights):
            # 将过去权重转换为 TensorDict 对象
            past_weights = TensorDict(past_weights)
            assert past_weights.keys() == curr_weights.keys()

            # 将当前权重与过去权重相加
            curr_weights = curr_weights + past_weights

        # 将序列从 Float['b n d'] 转换为查询
        # 对序列应用查询线性层，生成查询
        queries = self.to_queries(seq)

        # 处理多头
        # 对查询应用多头重塑
        queries = self.split_heads(queries)

        # 获取批次大小
        batch = queries.shape[0]

        # 从记忆模型中获取值
        # 重塑权重张量形状为 (batch * n, ...)
        curr_weights = curr_weights.apply(lambda t: rearrange(t, 'b n ... -> (b n) ...'))
        # 重塑查询张量形状为 (batch * n, c, d)
        queries = rearrange(queries, 'b (n c) d -> (b n) c d', c = chunk_size)

        # 前向传播函数调用
        # 使用当前权重和查询调用记忆模型，获取值
        values = functional_call(self.memory_model, dict(curr_weights), queries)

        # 恢复批次维度
        # 重塑值张量形状为 (batch, n * c, d)
        values = rearrange(values, '(b n) c d -> b (n c) d', b = batch)

        # 合并多头并组合
        # 对值应用多头合并
        values = self.merge_heads(values)
        # 对值应用多头组合
        values = self.combine_heads(values)

        # 后归一化
        # 论文中没有提到，但为了稳定训练，添加了后归一化
        values = self.post_rmsnorm(values)

        # 恢复填充
        # 在序列维度上填充，填充值为0（待改进：使用学习到的空记忆嵌入代替0）
        values = pad_at_dim(values, (chunk_size, 0), dim = 1, value = 0.) 
        # 截取填充后的序列，去除末尾的填充部分
        values = values[:, :-padding]

        # 返回检索到的记忆
        return values

    def forward(
        self,
        seq,
        store_seq = None,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
        return_next_memories = False
    ):
        """
        前向传播方法。

        参数:
            seq (torch.Tensor): 输入序列，形状为 (batch_size, seq_len, dim)。
            store_seq (torch.Tensor, 可选): 用于存储的序列，默认为 None。如果为 None，则使用输入序列。
            past_state (Tuple[Dict[str, Tensor], Dict[str, Tensor]], 可选): 过去的状态，包含权重和动量。
            return_next_memories (bool, 可选): 是否返回下一个记忆状态，默认为 False。

        返回:
            Tuple[torch.Tensor, Optional[Tuple[Dict[str, Tensor], Dict[str, Tensor]]]]: 检索到的记忆，以及可选的下一个记忆状态。
        """
        # 获取批次大小和序列长度
        batch, seq_len = seq.shape[:2]

        if seq_len <= self.chunk_size:
            # 如果序列长度小于等于块大小，则返回全零张量
            return torch.zeros_like(seq)

        if exists(past_state):
            # 将过去状态转换为 TensorDict 对象
            past_state = tuple(TensorDict(d) for d in past_state)

        if not exists(past_state):
            # 如果过去状态不存在，则初始化权重和动量
            past_state = self.init_weights_and_momentum()

        # 如果未提供存储序列，则使用输入序列
        store_seq = default(store_seq, seq)

        # 调用存储记忆的方法，获取更新和下一个记忆状态
        updates, next_memories = self.store_memories(store_seq, past_state)

        # 获取过去的权重
        past_weights, _ = past_state

        # 调用检索记忆的方法，获取检索到的记忆
        retrieved = self.retrieve_memories(seq, past_weights + updates)

        if not return_next_memories:
            # 如果不返回下一个记忆状态，则返回检索到的记忆
            return retrieved

        # 返回检索到的记忆和下一个记忆状态
        return retrieved, next_memories
