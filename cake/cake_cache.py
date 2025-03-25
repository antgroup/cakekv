import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import DynamicCache, Cache, HybridCache
from typing import Any, Dict, List, Optional, Tuple, Union
from cake.utils import adjust_budgets

class CakeCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.pref_scores = []
        self.evict_scores = []
        self.layer_budget = []
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def to(self, device):
        """
        Move the cache to the specified device.
        """
        cache = CakeCache()
        cache._seen_tokens = self._seen_tokens
        
        cache.key_cache = [k.to(device) if isinstance(k, torch.Tensor) else k for k in self.key_cache]
        cache.value_cache = [v.to(device) if isinstance(v, torch.Tensor) else v for v in self.value_cache]
        
        cache.pref_scores = [p.to(device) if isinstance(p, torch.Tensor) else p for p in self.pref_scores]
        cache.evict_scores = [e.to(device) if isinstance(e, torch.Tensor) else e for e in self.evict_scores]
        cache.layer_budget = self.layer_budget.copy() if hasattr(self.layer_budget, 'copy') else self.layer_budget
        
        return cache
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `CakeCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_score(
        self,
        pref_score: torch.Tensor,
        evict_score: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ):
        self.pref_scores.append(pref_score)
        self.evict_scores.append(evict_score)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. CakeCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `CakeCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "CakeCache":
        """Converts a cache in the legacy cache format into an equivalent `CakeCache`. Used for
        backward compatibility."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
    @classmethod
    def from_dynamic_cache(cls, past_key_values: Optional[DynamicCache] = None) -> "CakeCache":
        cache = cls()
        if past_key_values is not None:
            cache.key_cache = past_key_values.key_cache
            cache.value_cache = past_key_values.value_cache

        return cache
    @classmethod
    def from_hybrid_cache(cls, past_key_values: Optional[HybridCache] = None) -> "CakeCache":
        cache = cls()
        if past_key_values is not None:
            cache.key_cache = past_key_values.key_cache
            cache.value_cache = past_key_values.value_cache

        return cache
    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""

        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
            self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    def batch_split(self, full_batch_size: int, split_size: int) -> List["CakeCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = CakeCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            current_split.pref_scores = self.pref_scores
            current_split.evict_scores = self.evict_scores
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["CakeCache"]) -> "CakeCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls()
        for idx in range(len(splits[0])):
            layer_keys = torch.cat([current.key_cache[idx] for current in splits], dim=0)
            layer_values = torch.cat([current.value_cache[idx] for current in splits], dim=0)
            cache.update(layer_keys, layer_values, idx)
            cache.pref_scores = splits[0].pref_scores
            cache.evict_scores = splits[0].evict_scores
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]



class CakeprefillKVCache:
    def __init__(
        self,
        cache_size=512,
        window_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        num_heads = 32, 
        num_layers = 32,
        use_cascading = False
    ):

        self.window_size = window_size
        self.total_size = (cache_size-window_size) * num_layers
        self.cache_size = cache_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_cascading = use_cascading  # If true, ensure high attention precision

        # print(f"CakeprefillKVCache: {self.total_size}, {self.window_size}")
        
    def __call__(self, past_key_values, seq_len):
        if seq_len<=self.cache_size+self.window_size:
            return past_key_values

        pref_scores = past_key_values.pref_scores
  
        layer_budgets = [pref_score/sum(pref_scores)*self.total_size for pref_score in pref_scores]
    
        layer_budgets = adjust_budgets(layer_budgets, self.total_size, seq_len-self.window_size,  self.num_layers)

        if self.use_cascading:
            layer_idx = 0
            print(layer_budgets)
            for budget in layer_budgets:
                if budget>= seq_len-self.window_size:
                    budget = seq_len-self.window_size
                past_key_values = self.evcit_layer_kvcache(past_key_values, layer_idx, budget)
                past_key_values.layer_budget[layer_idx]=budget
                layer_idx +=1
        else:
            layer_idx = 0
            if len(layer_budgets) ==self.num_layers:
                for budget in layer_budgets:
                    if budget>= seq_len-self.window_size:
                        budget = seq_len-self.window_size
                    past_key_values = self.evcit_layer_kvcache(past_key_values, layer_idx, budget)
                    past_key_values.layer_budget[layer_idx]=budget
                    layer_idx +=1

        return past_key_values

    def evcit_layer_kvcache(self, past_key_values, layer_idx, budget):

        bsz, num_key_value_heads, seq_len, head_dim = past_key_values.key_cache[layer_idx].shape

        num_key_value_groups = self.num_heads // num_key_value_heads
        hh_score = past_key_values.evict_scores[layer_idx]

        if budget> hh_score.shape[-1]:
            budget=hh_score.shape[-1]
  
        indices = hh_score.topk(budget, dim=-1).indices
        hh_score_compress = hh_score.gather(dim=2, index=indices)
        past_key_values.evict_scores[layer_idx] = hh_score_compress

        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past_compress = past_key_values.key_cache[layer_idx][:, :, :-self.window_size, :].gather(dim=2, index=indices)
        v_past_compress = past_key_values.value_cache[layer_idx][:, :, :-self.window_size, :].gather(dim=2, index=indices)
        k_cur = past_key_values.key_cache[layer_idx][:, :, -self.window_size:, :]
        v_cur = past_key_values.value_cache[layer_idx][:, :, -self.window_size:, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)
        
        past_key_values.key_cache[layer_idx] = key_states
        past_key_values.value_cache[layer_idx] = value_states

        return past_key_values

class CakeDecodingKVCache_LayerWise:
    def __init__(
        self,
        hh_size=128,
        window_size=32,
        k_seq_dim=2,
        v_seq_dim=2,

    ):
        # print(f"CakeDecodingKVCache_LayerWise: {hh_size}, {window_size}")
        self.hh_size = hh_size
        self.window_size = window_size
        self.cache_size = hh_size + window_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None

    def __call__(self, past_key_values, attn_score_cache, layer_idx):
        num_heads = attn_score_cache.shape[1]
        bsz, num_key_value_heads, seq_len, head_dim = past_key_values.key_cache[layer_idx].shape
        num_key_value_groups = num_heads // num_key_value_heads

        seq_len = past_key_values.key_cache[layer_idx].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values

        attn_cache = attn_score_cache[:, :, :, :-self.window_size].mean(dim = -2)

        attn_cache = F.avg_pool1d(attn_cache, kernel_size = 5, padding=5//2, stride=1)
        attn_cache = attn_cache.reshape(bsz, num_key_value_heads, num_key_value_groups, -1)

        attn_cache = attn_cache.mean(dim=-2)

        indices = attn_cache.topk(self.hh_size, dim=-1).indices
        # indices = indices.sort().values
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past_compress = past_key_values.key_cache[layer_idx][:, :, :-self.window_size, :].gather(dim=2, index=indices)
        v_past_compress = past_key_values.value_cache[layer_idx][:, :, :-self.window_size, :].gather(dim=2, index=indices)
        k_cur = past_key_values.key_cache[layer_idx][:, :, -self.window_size:, :]
        v_cur = past_key_values.value_cache[layer_idx][:, :, -self.window_size:, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)

        past_key_values.key_cache[layer_idx] = key_states
        past_key_values.value_cache[layer_idx] = value_states

        return past_key_values

    def _update_hh_score(self, attn_score_cache, num_key_value_heads):

        bsz,num_heads, num_new_tokens,_ = attn_score_cache.shape
        num_key_value_groups = num_heads //  num_key_value_heads
        if self.hh_score is None:
            self.hh_score = attn_score_cache.sum(2) #bsz, total num_heads, seq_len
            self.hh_score = self.hh_score.reshape(bsz, num_key_value_heads, num_key_value_groups, -1)
            self.hh_score = self.hh_score.mean(dim=-2)
        
        else:
            # print(self.hh_score.shape)
            attn_score_cache = attn_score_cache.sum(2)
            attn_score_cache = attn_score_cache.reshape(bsz, num_key_value_heads, num_key_value_groups, -1)
            attn_score_cache = attn_score_cache.mean(dim=-2)
            attn_score_cache[:, :, :-num_new_tokens] += self.hh_score
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None
