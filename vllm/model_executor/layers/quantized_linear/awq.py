from typing import Optional

import torch
from torch.nn.parameter import Parameter

from vllm import quantization_ops
from vllm import cutlass_quantization_ops
from vllm.model_executor.parallel_utils.layers import (ColumnParallelLinear,
                                                       RowParallelLinear)
order_map = [0, 2, 4, 6, 1, 3, 5, 7]
w_bit = 4
pack_num = len(order_map)
# use_cutlass = False
use_cutlass = True


def to_column_major_tile_interleave(data, thread_block_k=64, col_interleave=4):
    data = data.t()
    n, k = data.shape
    # tile interleave
    data = data.reshape([n//col_interleave, col_interleave, k//thread_block_k, thread_block_k])
    data = data.transpose(1, 2)
    data = data.reshape(-1, 4, 8).transpose(1, 2)

    # row major -> column major
    return data.reshape(k, n).contiguous()


def revert_order(data):
    out = torch.zeros((data.shape[0], data.shape[1] * pack_num), dtype=torch.int32, device=data.device)
    for col in range(data.shape[1]):
        d8 = data[:, col]
        for i in range(pack_num):
            out[:, col * pack_num + order_map[i]] = (d8 >> (i * w_bit)) & 15
    return out


def pack_bits(data):
    out = torch.zeros((data.shape[0], data.shape[1] // pack_num), dtype=torch.int32, device=data.device)
    for col in range(data.shape[1] // pack_num):
        for i in range(pack_num):
            dcol = data[:, col * pack_num + i]
            out[:, col] |= dcol << (i * w_bit)
    return out


class AWQColumnParallelLinear(ColumnParallelLinear):

    def create_weights(self, dtype: torch.dtype) -> None:
        assert self.input_size % self.quant_config.weight_bits == 0
        assert (self.output_size_per_partition %
                self.quant_config.pack_factor == 0)
        self.qweight = Parameter(
            torch.empty(
                self.input_size,
                self.output_size_per_partition //
                self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        self.qzeros = Parameter(
            torch.empty(
                self.input_size // self.quant_config.group_size,
                self.output_size_per_partition //
                self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        self.scales = Parameter(
            torch.empty(
                self.input_size // self.quant_config.group_size,
                self.output_size_per_partition,
                device="cuda",
                dtype=dtype,
            ),
            requires_grad=False,
        )
        self.use_cutlass = use_cutlass
        self.is_cutlass_weight = False


    def apply_weights(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pack_factor = self.quant_config.pack_factor
        out_shape = (x.shape[-2], self.qweight.shape[-1] * pack_factor)
        reshaped_x = x.reshape(-1, x.shape[-1])
        if self.use_cutlass:
            # out0 = quantization_ops.awq_gemm(reshaped_x, self.qweight, self.scales,
            #                                 self.qzeros, pack_factor)
            if not self.is_cutlass_weight:

                assert pack_factor == 8
                qweight = revert_order(self.qweight)
                qweight = to_column_major_tile_interleave(qweight)
                self.qweight.data = pack_bits(qweight)
                qzeros = revert_order(self.qzeros).to(self.scales.dtype)
                self.qzeros.data = (8 - qzeros) * self.scales
                self.is_cutlass_weight = True

            out = cutlass_quantization_ops.awq_gemm_cutlass(
                    reshaped_x, self.qweight, self.scales, self.qzeros)
            # is_all_close = torch.allclose(out0, out)
        else:
            out = quantization_ops.awq_gemm(reshaped_x, self.qweight, self.scales,
                                            self.qzeros, pack_factor)
        if bias is not None:
            out = out + bias
        return out.reshape(out_shape)


class AWQRowParallelLinear(RowParallelLinear):

    def create_weights(self, dtype: torch.dtype) -> None:
        assert (self.input_size_per_partition %
                self.quant_config.weight_bits == 0)
        assert self.output_size % self.quant_config.pack_factor == 0
        self.qweight = Parameter(
            torch.empty(
                self.input_size_per_partition,
                self.output_size // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        self.qzeros = Parameter(
            torch.empty(
                self.input_size_per_partition // self.quant_config.group_size,
                self.output_size // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        self.scales = Parameter(
            torch.empty(
                self.input_size_per_partition // self.quant_config.group_size,
                self.output_size,
                device="cuda",
                dtype=dtype,
            ),
            requires_grad=False,
        )
        self.use_cutlass = use_cutlass
        self.is_cutlass_weight = False

    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        pack_factor = self.quant_config.pack_factor
        out_shape = (x.shape[-2], self.qweight.shape[-1] * pack_factor)
        reshaped_x = x.reshape(-1, x.shape[-1])
        if self.use_cutlass:
            #out0 = quantization_ops.awq_gemm(reshaped_x, self.qweight, self.scales,
            #                                self.qzeros, pack_factor)
            # dir_name = "x".join([str(x) for x in self.qweight.shape])
            # torch.save(reshaped_x, f'dump/{dir_name}/reshaped_x.pth')
            # torch.save(self.qweight, f'dump/{dir_name}/qweight.pth')
            # torch.save(self.scales, f'dump/{dir_name}/scales.pth')
            # torch.save(self.qzeros, f'dump/{dir_name}/qzeros.pth')
            # import sys
            # sys.exit(-1)
 
            if not self.is_cutlass_weight:
                assert pack_factor == 8
                qweight = revert_order(self.qweight)
                qweight = to_column_major_tile_interleave(qweight)
                self.qweight.data = pack_bits(qweight)
                qzeros = revert_order(self.qzeros).to(self.scales.dtype)
                self.qzeros.data = (8 - qzeros) * self.scales
                self.is_cutlass_weight = True

            out = cutlass_quantization_ops.awq_gemm_cutlass(
                    reshaped_x, self.qweight, self.scales, self.qzeros)
            # is_all_close = torch.allclose(out0, out)
        else: 
            out = quantization_ops.awq_gemm(reshaped_x, self.qweight, self.scales,
                                            self.qzeros, pack_factor)
        return out.reshape(out_shape)
