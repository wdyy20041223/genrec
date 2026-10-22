"""A fixed-length catalog SID Trie; decoder start/PAD is not part of a SID."""
from dataclasses import dataclass, field

import torch


@dataclass
class SIDTrie:
    children: torch.Tensor
    sid_length: int
    vocab_size: int
    item_count: int
    unique_sid_count: int
    prefix_counts: list[int]
    _device_children: torch.Tensor | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_semantic_ids(cls, semantic_ids, codebook_size=256):
        """Build from raw codes, applying token[d] = code[d] + d*C + 1."""
        raw = torch.as_tensor(semantic_ids, device="cpu")
        if raw.ndim != 2 or not raw.shape[0] or not raw.shape[1]:
            raise ValueError("Expected a nonempty (items, SID length) array")
        if raw.is_floating_point() or raw.is_complex() or raw.dtype == torch.bool:
            raise ValueError("SID codes must be integers")
        if codebook_size < 1 or (raw < 0).any() or (raw >= codebook_size).any():
            raise ValueError("SID code is outside its codebook")
        length = raw.shape[1]
        tokens = raw.long() + torch.arange(length) * codebook_size + 1
        unique = torch.unique(tokens, dim=0)
        paths = unique.tolist()
        # Node 0 is the root; node 1 is a shared terminal with no children.
        nodes = {(): 0}
        for path in paths:
            for depth in range(1, length):
                prefix = tuple(path[:depth])
                if prefix not in nodes:
                    nodes[prefix] = len(nodes) + 1
        vocab_size = length * codebook_size + 1
        children = torch.full((len(nodes) + 1, vocab_size), -1, dtype=torch.int32)
        for path in paths:
            parent = 0
            for depth, token in enumerate(path, 1):
                child = 1 if depth == length else nodes[tuple(path[:depth])]
                children[parent, token] = child
                parent = child
        return cls(children, length, vocab_size, len(raw), len(unique),
                   [len(torch.unique(tokens[:, :d], dim=0)) for d in range(1, length + 1)])

    def children_on(self, device):
        """Cache one device table; generation then uses vectorized node lookups."""
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if self._device_children is None or self._device_children.device != device:
            self._device_children = self.children.to(device)
        return self._device_children

    def allowed_tokens(self, prefix):
        node = 0
        for token in prefix:
            if not 0 <= token < self.vocab_size:
                return []
            node = int(self.children[node, token])
            if node < 0:
                return []
        return (self.children[node] >= 0).nonzero(as_tuple=True)[0].tolist()

    def save(self, path):
        torch.save({name: getattr(self, name) for name in (
            "children", "sid_length", "vocab_size", "item_count", "unique_sid_count", "prefix_counts")}, path)

    @classmethod
    def load(cls, path):
        return cls(**torch.load(path, map_location="cpu", weights_only=True))
