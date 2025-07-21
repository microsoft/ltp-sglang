import asyncio
import fastapi
import copy
import logging 
from functools import wraps
from typing import Optional, Union, List, Any
from sglang.srt.managers.io_struct import GenerateReqInput, EmbeddingReqInput, TokenizedGenerateReqInput
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager
from sglang.srt.managers.scheduler import Scheduler
from sglang.utils import TypeBasedDispatcher

logger = logging.getLogger(__name__)

def apply_list_checking():
    original_call = TypeBasedDispatcher.__call__
    def __call__(self, obj: Any):
        for ty, fn in self._mapping:
            if isinstance(obj, ty.__origin__ if hasattr(ty, "__origin__") else ty):
                return fn(obj)
        raise ValueError(f"Invalid object: {obj}")
    TypeBasedDispatcher.__call__ = __call__

# def apply_real_batch_to_handling():
#     async def _handle_batch_request(
#         self,
#         obj: Union[GenerateReqInput, EmbeddingReqInput],
#         request: Optional[fastapi.Request] = None,
#         created_time: Optional[float] = None,
#     ):
#         batch_size = obj.batch_size

#         generators = []
#         rids = []
#         if getattr(obj, "parallel_sample_num", 1) == 1:
#             # Send all requests
#             objs = [obj[i] for i in range(batch_size)]
#             tokenized_objs = await asyncio.gather(
#                 *(self._tokenize_one_request(obj) for obj in objs)
#             )
#             states = [ReqState([], False, asyncio.Event(), obj, created_time=created_time) for obj in objs]
#             for i, obj in enumerate(objs):
#                 self.rid_to_state[obj.rid] = states[i]
#                 generators.append(self._wait_one_response(obj, states[i], request))
#                 rids.append(obj.rid)
#             self.send_to_scheduler.send_pyobj(tokenized_objs)
#         else:
#             # FIXME: When using batch and parallel_sample_num together, the perf is not optimal.
#             if batch_size > 128:
#                 logger.warning(
#                     "Sending a single large batch with parallel sampling (n > 1) has not been well optimized. "
#                     "The performance might be better if you just duplicate the requests n times or use "
#                     "many threads to send them one by one with parallel sampling (n > 1)."
#                 )

#             # Tokenize all requests
#             objs = [obj[i] for i in range(batch_size)]
#             tokenized_objs = await asyncio.gather(
#                 *(self._tokenize_one_request(obj) for obj in objs)
#             )

#             # Cache the common prefix for parallel sampling
#             for i in range(batch_size):
#                 tmp_obj = copy.copy(objs[i])
#                 tokenized_obj = copy.copy(tokenized_objs[i])
#                 tokenized_obj.rid = tmp_obj.regenerate_rid()
#                 tokenized_obj.sampling_params = copy.copy(tokenized_obj.sampling_params)
#                 tokenized_obj.sampling_params.max_new_tokens = 0
#                 tokenized_obj.stream = False
#                 state = self._send_one_request(tmp_obj, tokenized_obj, created_time)
#                 await self._wait_one_response(tmp_obj, state, request).__anext__()

#             # Expand requests, assign new rids for them, and send them
#             for i in range(batch_size):
#                 for _ in range(obj.parallel_sample_num):
#                     tmp_obj = copy.copy(objs[i])
#                     tokenized_obj = copy.copy(tokenized_objs[i])
#                     tokenized_obj.rid = tmp_obj.regenerate_rid()
#                     state = self._send_one_request(tmp_obj, tokenized_obj, created_time)
#                     generators.append(self._wait_one_response(tmp_obj, state, request))
#                     rids.append(tmp_obj.rid)

#         # Wait for all requests
#         is_stream = hasattr(obj, "stream") and obj.stream
#         if not is_stream:
#             outputs = await asyncio.gather(*(gen.__anext__() for gen in generators))
#             yield outputs
#         else:
#             rid_to_index = {rid: i for i, rid in enumerate(rids)}
#             task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}
#             while task_map:
#                 done, _ = await asyncio.wait(
#                     task_map.keys(), return_when=asyncio.FIRST_COMPLETED
#                 )

#                 for task in done:
#                     gen = task_map.pop(task)
#                     try:
#                         result = task.result()
#                         result["index"] = rid_to_index[result["meta_info"]["id"]]
#                         yield result
#                         new_task = asyncio.create_task(gen.__anext__())
#                         task_map[new_task] = gen
#                     except StopAsyncIteration:
#                         pass
#     original_handle_batch_request = TokenizerManager._handle_batch_request
#     TokenizerManager._handle_batch_request = _handle_batch_request

def apply_real_batch_to_handling():
    async def _handle_batch_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
        created_time: Optional[float] = None,
    ):
        batch_size = obj.batch_size

        generators = []
        rids = []
        if getattr(obj, "parallel_sample_num", 1) == 1:
            # Send all requests
            objs = [obj[i] for i in range(batch_size)]
            tokenized_objs = await asyncio.gather(
                *(self._tokenize_one_request(obj) for obj in objs)
            )
            states = [ReqState([], False, asyncio.Event(), obj, created_time=created_time) for obj in objs]
            for i, obj in enumerate(objs):
                self.rid_to_state[obj.rid] = states[i]
                generators.append(self._wait_one_response(obj, request))
                rids.append(obj.rid)
            self.send_to_scheduler.send_pyobj(tokenized_objs)
        else:
            # FIXME: When using batch and parallel_sample_num together, the perf is not optimal.
            if batch_size > 128:
                logger.warning(
                    "Sending a single large batch with parallel sampling (n > 1) has not been well optimized. "
                    "The performance might be better if you just duplicate the requests n times or use "
                    "many threads to send them one by one with parallel sampling (n > 1)."
                )

            # Tokenize all requests
            objs = [obj[i] for i in range(batch_size)]
            tokenized_objs = await asyncio.gather(
                *(self._tokenize_one_request(obj) for obj in objs)
            )

            # Cache the common prefix for parallel sampling
            for i in range(batch_size):
                tmp_obj = copy.copy(objs[i])
                tokenized_obj = copy.copy(tokenized_objs[i])
                tokenized_obj.rid = tmp_obj.regenerate_rid()
                tokenized_obj.sampling_params = copy.copy(tokenized_obj.sampling_params)
                tokenized_obj.sampling_params.max_new_tokens = 0
                tokenized_obj.stream = False
                self._send_one_request(tmp_obj, tokenized_obj, created_time)
                await self._wait_one_response(tmp_obj, request).__anext__()

            # Expand requests, assign new rids for them, and send them
            for i in range(batch_size):
                for _ in range(obj.parallel_sample_num):
                    tmp_obj = copy.copy(objs[i])
                    tokenized_obj = copy.copy(tokenized_objs[i])
                    tokenized_obj.rid = tmp_obj.regenerate_rid()
                    self._send_one_request(tmp_obj, tokenized_obj, created_time)
                    generators.append(self._wait_one_response(tmp_obj, request))
                    rids.append(tmp_obj.rid)

        # Wait for all requests
        is_stream = hasattr(obj, "stream") and obj.stream
        if not is_stream:
            outputs = await asyncio.gather(*(gen.__anext__() for gen in generators))
            yield outputs
        else:
            rid_to_index = {rid: i for i, rid in enumerate(rids)}
            task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}
            while task_map:
                done, _ = await asyncio.wait(
                    task_map.keys(), return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    gen = task_map.pop(task)
                    try:
                        result = task.result()
                        result["index"] = rid_to_index[result["meta_info"]["id"]]
                        yield result
                        new_task = asyncio.create_task(gen.__anext__())
                        task_map[new_task] = gen
                    except StopAsyncIteration:
                        pass
    original_handle_batch_request = TokenizerManager._handle_batch_request
    TokenizerManager._handle_batch_request = _handle_batch_request

def add_batch_handle_generate_request():
    def batch_handle_generate_request(
        self,
        recv_reqs: List[TokenizedGenerateReqInput],
    ):
        for req in recv_reqs:
            self.handle_generate_request(req)
    Scheduler.batch_handle_generate_request = batch_handle_generate_request

def add_new_dispatcher():
    original_init = Scheduler.__init__
    @wraps(original_init)
    def init_with_new_dispatcher(self, *args, **kwargs):
        """
        Initialize the Scheduler with a new dispatcher.
        """
        original_init(self, *args, **kwargs)
        self._request_dispatcher._mapping.append((List[TokenizedGenerateReqInput], self.batch_handle_generate_request))
    Scheduler.__init__ = init_with_new_dispatcher
    
apply_list_checking()
add_batch_handle_generate_request()
add_new_dispatcher()
apply_real_batch_to_handling()
