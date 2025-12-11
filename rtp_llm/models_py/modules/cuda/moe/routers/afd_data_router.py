from queue import Queue
from typing import Any, Optional, Tuple

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.distributed.deepep_initializer import DeepEpInitializer
from rtp_llm.models_py.modules.common.moe.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoeDataRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.quant_config import FusedMoEQuantConfig
from rtp_llm.models_py.modules.factory.fused_moe.type import RouterType

# 理想的overlap（完全掩盖通信）顺序, 按照step3流水线：
# FFN: R1,F1,R2,S1,F2...
# ATTN: A1,S1,A2,S2,R1,A3...
# TODO AfdDataRouterAttn适应任意micro_batch_num(>2)大小, 提前fetch的时机更前: 即2次S后就R


# 目前只有当micro_batch_num=3时, AfdDataRouterAttn才能配合上FFN
class AfdDataRouterAttn(FusedMoeDataRouter):
    """
    A data router for Mixture-of-Experts that utilizes deep_ep's m2n low-latency communication primitives.

    This router is used for attention service, it receives compute result from FFN ranks.
    """

    @classmethod
    def router_type(cls):
        return RouterType.AFD_ATTN_ROUTER

    @classmethod
    def check_conditions(cls, checker: Any, config: GptInitModelParameters) -> None:
        """Check if AfdDataRouterAttn can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.use_low_latency(config))
        checker.check(resolver.is_afd_enabled(config))
        checker.check(not resolver.is_afd_ffn_rank(config))
        checker.check(DeepEpInitializer.supported())

    def __init__(
        self,
        config: GptInitModelParameters,
        use_fp8_dispatch: bool = False,
        zero_copy: bool = False,
        async_finish: bool = True,
        return_recv_hook: bool = False,
    ):
        super().__init__()
        self.workspace: Optional[torch.Tensor] = None
        self.prefetch_output: Optional[torch.Tensor] = None
        # self.cached_topk_ids: Optional[torch.Tensor] = None
        # self.cached_topk_weights: Optional[torch.Tensor] = None

        self.is_last_layer = False

        self.config = config
        wrapper = DeepEpInitializer.get_deepep_wrapper(self.config)
        self.buffer = wrapper.buffer
        self.rank = config.gpt_init_params.parallelism_distributed_config.world_rank
        self.world_size = config.world_size
        self.num_attn_ranks = (
            config.gpt_init_params.ffn_disaggregate_config.attention_dp_size
            * config.gpt_init_params.ffn_disaggregate_config.attention_tp_size
        )
        self.num_experts = config.expert_num
        # self.comm_handle = None
        self.num_max_dispatch_tokens_per_rank = wrapper.ll_num_max_token_per_rank
        self.last_send_event: Optional[DeepEPEventOverlap] = None

        self.micro_batch_num = config.device_resource_config.micro_batch_num
        assert self.micro_batch_num > 1

        self.cached_topk_ids_queue: Queue[torch.Tensor] = Queue(
            maxsize=self.micro_batch_num - 1
        )
        self.cached_topk_weights_queue: Queue[torch.Tensor] = Queue(
            maxsize=self.micro_batch_num - 1
        )
        self.comm_handle_queue: Queue[deep_ep.EventOverlap] = Queue(
            maxsize=self.micro_batch_num - 1
        )

    def fetch(self) -> torch.Tensor:
        # assert self.cached_topk_ids is not None
        # assert self.cached_topk_weights is not None
        # assert self.comm_handle is not None

        assert not self.cached_topk_ids_queue.empty()
        assert not self.cached_topk_weights_queue.empty()
        assert not self.comm_handle_queue.empty()

        if self.last_send_event is not None:
            self.last_send_event.current_stream_wait()  # 缺失会导致内存访问错误,why?
            self.last_send_event = None

        cached_topk_ids = self.cached_topk_ids_queue.get()
        cached_topk_weights = self.cached_topk_weights_queue.get()
        comm_handle = self.comm_handle_queue.get()

        num_tokens, num_topk = cached_topk_ids.size()

        combined_x, event, _ = self.buffer.low_latency_combine_recv(
            cached_topk_ids,
            cached_topk_weights,
            comm_handle,
            self.num_attn_ranks,
            num_tokens,
            num_topk,
            zero_copy=False,
            async_finish=True,
            return_recv_hook=False,
            # out=self.workspace,
        )
        # print(f"combine recv done, self.cached_topk_ids_queue.size: {self.cached_topk_ids_queue.qsize()}, self.cached_topk_weights_queue.size: {self.cached_topk_weights_queue.qsize()}, self.comm_handle_queue.size: {self.comm_handle_queue.qsize()}", flush=True)
        event.current_stream_wait()  # 计算流等待recv完成
        # self.comm_handle = None
        # self.cached_topk_ids = None
        # self.cached_topk_weights = None

        output = combined_x

        assert isinstance(output, torch.Tensor)

        return output

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
    ):
        assert a1.dim() == 2 and topk_ids.dim() == 2
        assert a1.size(0) == topk_ids.size(0)

        self.workspace = torch.zeros_like(a1)

        if self.comm_handle_queue.full():
            self.prefetch_output = self.fetch()

        num_tokens, hidden_dim = a1.size()
        num_topk = topk_ids.size(1)

        assert num_tokens <= self.num_max_dispatch_tokens_per_rank
        assert self.rank < self.num_attn_ranks, ""

        _, _, handle, event, _ = self.buffer.low_latency_dispatch_send(
            a1,
            topk_ids,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            self.num_attn_ranks,
            num_topk,
            use_fp8=False,
            async_finish=True,
            return_recv_hook=False,
        )
        # print(f"dispatch send done", flush=True)
        # event.current_stream_wait()
        self.last_send_event = event

        self.cached_topk_ids_queue.put_nowait(topk_ids)
        self.cached_topk_weights_queue.put_nowait(topk_weights)
        self.comm_handle_queue.put_nowait(handle)

    def finalize(
        self,
        fused_expert_output: torch.Tensor = None,
        topk_weights: torch.Tensor = None,
        topk_ids: torch.Tensor = None,
        apply_router_weight_on_input: bool = True,
        extra_finalize_args: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        if self.prefetch_output is not None:
            output = self.prefetch_output
            self.prefetch_output = None
            return output
        return self.fetch()


class AfdDataRouterFfn(FusedMoeDataRouter):
    """
    A data router for Mixture-of-Experts that utilizes deep_ep's m2n low-latency communication primitives.

    This router is used for FFN service, it receives tokens from attention ranks and computes the FFN output.
    """

    @classmethod
    def router_type(cls):
        return RouterType.AFD_FFN_ROUTER

    @classmethod
    def check_conditions(cls, checker: Any, config: GptInitModelParameters) -> None:
        """Check if AfdDataRouterFfn can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.use_low_latency(config))
        checker.check(resolver.is_afd_enabled(config))
        checker.check(resolver.is_afd_ffn_rank(config))
        checker.check(DeepEpInitializer.supported())

    def __init__(
        self,
        config: GptInitModelParameters,
        use_fp8_dispatch: bool = False,
        zero_copy: bool = False,
        async_finish: bool = True,
        return_recv_hook: bool = False,
    ):
        super().__init__()
        self.config = config
        wrapper = DeepEpInitializer.get_deepep_wrapper(self.config)
        self.buffer = wrapper.buffer
        self.rank = config.gpt_init_params.parallelism_distributed_config.world_rank
        self.world_size = config.world_size
        self.num_attn_ranks = (
            config.gpt_init_params.ffn_disaggregate_config.attention_dp_size
            * config.gpt_init_params.ffn_disaggregate_config.attention_tp_size
        )
        self.num_experts = config.expert_num
        self.comm_handle = None
        self.num_max_dispatch_tokens_per_rank = wrapper.ll_num_max_token_per_rank
        self.last_send_event: Optional[deep_ep.EventOverlap] = None

        self.prefetch_payload: Optional[ExpertForwardPayload] = None

        self.layer_num = config.layer_num
        self.micro_batch_num = config.device_resource_config.micro_batch_num
        self.max_count = self.layer_num * self.micro_batch_num
        self.count: int = 0  # 0 ~ layer_num * micro_batch_num

        # print(f"max count: {self.max_count}, count: {self.count}", flush=True)

        self.hidden_dim = config.hidden_size
        self.topk = config.moe_k

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
    ) -> ExpertForwardPayload:

        if self.prefetch_payload is not None:
            output = self.prefetch_payload
            self.prefetch_payload = None
            return output
        return self.fetch()

    def fetch(self) -> ExpertForwardPayload:

        assert (
            self.comm_handle is None
        ), "Communication handle should be clean before prepare()."

        assert self.count < self.max_count, "exceed max count"

        assert self.rank >= self.num_attn_ranks

        if self.last_send_event is not None:
            self.last_send_event.current_stream_wait()  # ?
            self.last_send_event = None

        packed_recv_x, packed_recv_count, handle, event, _ = (
            self.buffer.low_latency_dispatch_recv(
                self.hidden_dim,
                self.topk,
                self.num_max_dispatch_tokens_per_rank,
                self.num_experts,
                self.num_attn_ranks,
                use_fp8=False,
                async_finish=True,
                return_recv_hook=False,
            )
        )
        # print(f"dispatch recv done, count: {self.count}", flush=True)
        self.count += 1

        event.current_stream_wait()  # 计算需要等待recv完成
        assert isinstance(packed_recv_x, torch.Tensor)

        # N ranks wait for received data to build the payload for local experts.
        expert_tokens_meta = ExpertTokensMetadata(
            expert_num_tokens=packed_recv_count, expert_num_tokens_cpu=None
        )
        assert isinstance(packed_recv_x, torch.Tensor)
        payload = ExpertForwardPayload(
            expert_x=packed_recv_x,
            expert_x_scale=None,
            expert_tokens_meta=expert_tokens_meta,
        )

        self.comm_handle = handle

        return payload

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        assert (
            self.comm_handle is not None
        ), "Communication handle is missing for finalize()."

        handle = self.comm_handle
        self.comm_handle = None
        last_send_wait = False

        if self.count < self.max_count:
            self.prefetch_payload = self.fetch()
        else:
            self.count = 0
            last_send_wait = True

        _, num_topk = topk_ids.size()

        _, event, _ = self.buffer.low_latency_combine_send(
            fused_expert_output,
            handle,
            num_topk,
            zero_copy=False,
            async_finish=True,
            return_recv_hook=False,
        )
        # print(f"combine send done, count: {self.count}", flush=True)
        event.current_stream_wait()
        # self.last_send_event = event

        if last_send_wait:
            event.current_stream_wait()

        _, hidden_dim = extra_finalize_args["a1_shape"]
        return torch.empty(0, hidden_dim)
