"""Exact engineering alternatives for the fixed history-calibrated decoder."""
import numpy as np
import torch
import torch.nn.functional as F

from .inference_priors import PrefixPrior
from .inference_rerank import sid_keys


def reorder_self_cache(cache, parent):
    """Constant-width beams stay within users; encoder cross KV is user-constant."""
    return tuple((tuple(x.index_select(0, parent) for x in layer[0]), layer[1]) for layer in cache)


class CompactPrior:
    """Compact legal-prefix indexes; unknown full-vocabulary candidates get zero."""
    def __init__(self, reference, device="cuda"):
        if reference.strengths[0] != 0:
            raise ValueError("This fast prior preserves the fixed alpha1=0 protocol")
        self.reference = reference
        self.device = torch.device(device)
        self.catalog = reference.catalog_keys
        self.prefixes = np.unique(self.catalog // 769)
        self.sizes = (len(self.prefixes) + 1, len(self.catalog) + 1)
        # Zero is an unmapped-prefix sentinel, never a valid SID catalog index.
        map2 = np.zeros((769, 769), np.int32)
        map2[self.prefixes // 769, self.prefixes % 769] = np.arange(1, len(self.prefixes) + 1)
        map3 = np.zeros((len(self.prefixes) + 1, 769), np.int32)
        map3[np.searchsorted(self.prefixes, self.catalog // 769) + 1, self.catalog % 769] = np.arange(1, len(self.catalog) + 1)
        self.map2 = torch.from_numpy(map2).to(self.device)
        self.map3 = torch.from_numpy(map3).to(self.device)

    def __call__(self, depth, parents, probabilities):
        if depth == 1:
            return 0.
        rows = torch.arange(len(parents), device=parents.device)[:, None, None]
        if depth == 2:
            columns = self.map2[parents[:, :, 1]]
        else:
            prefix = self.map2[parents[:, :, 1], parents[:, :, 2]]
            columns = self.map3[prefix]
        return self.tables[depth - 2][rows, columns]


class CPUCompactPrior(CompactPrior):
    """Merge each user's three sparse histories once, transfer two compact tables."""
    def __init__(self, reference, device="cuda"):
        super().__init__(reference, device)
        self.by_depth = []
        for depth, keys in [(2, self.prefixes), (3, self.catalog)]:
            rows = {}
            for context, (tokens, lifts) in reference.by_depth[depth - 1].items():
                code = tokens[:, 0] * 769 + tokens[:, 1]
                if depth == 3:
                    code = code * 769 + tokens[:, 2]
                rows[context] = (np.searchsorted(keys, code) + 1, lifts)
            self.by_depth.append(rows)

    def prepare(self, ids, mask):
        self.reference.prepare(ids, mask)
        tables = []
        for offset, size in enumerate(self.sizes):
            table = np.zeros((len(ids), size), np.float32)
            alpha = self.reference.strengths[offset + 1]
            for row, contexts in enumerate(self.reference.contexts):
                for context, weight in contexts:
                    if alpha and context in self.by_depth[offset]:
                        columns, lifts = self.by_depth[offset][context]
                        table[row, columns] += alpha * weight * lifts
                if offset == 1 and self.reference.suppress_seen:
                    columns = np.searchsorted(self.catalog, sid_keys(self.reference.seen[row])) + 1
                    table[row, columns] -= 1000.
            tables.append(torch.from_numpy(table).to(self.device))
        self.tables = tables


class CSRGPUPrior(CompactPrior):
    """GPU CSR gathers with sequential history additions preserving float32 order."""
    def __init__(self, reference, device="cuda"):
        super().__init__(reference, device)
        self.catalog_gpu = torch.from_numpy(self.catalog).to(self.device)
        self.singleton = torch.from_numpy(reference.group_sizes == 1).to(self.device)
        self.csr = []
        for depth, targets in [(2, self.prefixes), (3, self.catalog)]:
            contexts = np.array(sorted(reference.by_depth[depth - 1]), np.int64)
            counts = [0] + [len(reference.by_depth[depth - 1][int(c)][0]) for c in contexts]
            indptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
            columns, values = [], []
            for context in contexts:
                tokens, lift = reference.by_depth[depth - 1][int(context)]
                code = tokens[:, 0] * 769 + tokens[:, 1]
                if depth == 3:
                    code = code * 769 + tokens[:, 2]
                columns.append(np.searchsorted(targets, code) + 1)
                values.append(lift)
            arrays = [contexts, indptr, np.r_[np.concatenate(columns), 0].astype(np.int64),
                      np.r_[np.concatenate(values), 0].astype(np.float32)]
            self.csr.append(tuple(torch.from_numpy(x).to(self.device) for x in arrays) +
                            (torch.arange(max(counts), device=self.device),))
        self.strengths = tuple(reference.strengths)
        self.history_items = reference.history_items
        self.suppress_seen = reference.suppress_seen
        # Keep the compact CSR, not a duplicate Python dictionary of all rows.
        del self.reference

    def prepare(self, ids, mask):
        ids, mask = ids.to(self.device), mask.to(self.device)
        # Left padding need not be a multiple of three: align groups from the end.
        padding = (-ids.shape[1]) % 3
        history = F.pad(ids, (padding, 0)).reshape(len(ids), -1, 3)
        active = F.pad(mask, (padding, 0)).reshape(len(ids), -1, 3).bool().all(-1)
        keys = (history[:, :, 0] * 769 + history[:, :, 1]) * 769 + history[:, :, 2]
        recent = F.pad(keys, (max(0, 3 - keys.shape[1]), 0))[:, -3:].flip(1)
        valid = F.pad(active, (max(0, 3 - active.shape[1]), 0))[:, -3:].flip(1)
        for age in (1, 2):
            valid[:, age] &= (recent[:, age:age + 1] != recent[:, :age]).all(-1)
        # Both depths have the same full-SID contexts; resolve the three once.
        contexts = self.csr[0][0]
        position = torch.searchsorted(contexts, recent.contiguous())
        safe = position.clamp_max(len(contexts) - 1)
        context_rows = torch.where(valid & (contexts[safe] == recent), position + 1, 0)
        self.tables = []
        for offset, (contexts, ptr, columns, values, steps) in enumerate(self.csr):
            table = torch.zeros((len(ids), self.sizes[offset]), device=self.device)
            alpha = self.strengths[offset + 1]
            for age in range(self.history_items):
                row = context_rows[:, age]
                start, end = ptr[row], ptr[row + 1]
                source = (start[:, None] + steps).clamp_max(len(values) - 1)
                keep = steps < (end - start)[:, None]
                dest = torch.where(keep, columns[source], 0)
                contribution = values[source] * (alpha * .5 ** age)
                table.scatter_add_(1, dest, torch.where(keep, contribution, 0.))
            if offset == 1 and self.suppress_seen:
                position = torch.searchsorted(self.catalog_gpu, keys)
                safe = position.clamp_max(len(self.catalog) - 1)
                keep = active & (self.catalog_gpu[safe] == keys) & self.singleton[safe]
                dest = torch.where(keep, position + 1, 0)
                seen = torch.zeros_like(table, dtype=torch.bool)
                seen.scatter_(1, dest, keep)
                # The sentinel can receive padding writes; it must remain zero.
                seen[:, 0] = False
                table.sub_(seen.float() * 1000.)
            self.tables.append(table)


@torch.inference_mode()
def fast_search(tiger, ids, mask, widths=(15, 15, 15), adjuster=None, trace=None):
    """Same scores/Top-K; constant width, reuse encoder and user-constant cross KV."""
    if tiger.training or len(widths) != 3 or len(set(widths)) != 1 or min(widths) < 1:
        raise ValueError("Exact fast path requires eval mode and constant Beam width")
    model = tiger.model
    encoded = model.encoder(input_ids=ids, attention_mask=mask)
    b, k, v = len(ids), widths[0], model.config.vocab_size
    paths = torch.full((b * k, 1), model.config.decoder_start_token_id, device=ids.device, dtype=torch.long)
    raw = torch.zeros((b, k), device=ids.device)
    raw[:, 1:] = -1e9
    cache, hidden, expanded_mask = None, None, None
    rows = torch.arange(b, device=ids.device)[:, None]
    for step in range(3):
        first = step == 0 and model.config.first_beam_only
        if not first and hidden is None:
            hidden = encoded[0].unsqueeze(1).expand(-1, k, -1, -1).reshape(b * k, encoded[0].shape[1], -1)
            expanded_mask = mask.unsqueeze(1).expand(-1, k, -1).reshape(b * k, -1)
        out = model(encoder_outputs=encoded if first else (hidden,), attention_mask=mask if first else expanded_mask,
                    decoder_input_ids=paths.view(b, k, -1)[:, 0] if first else paths if cache is None else paths[:, -1:],
                    use_cache=True, past_key_values=cache)
        cache = out.past_key_values
        if first:
            cache = tuple(tuple(None if pair is None else tuple(x.repeat_interleave(k, 0) for x in pair)
                                for pair in layer) for layer in cache)
        lp = out.logits[:, -1].float().log_softmax(-1).view(b, 1 if first else k, v)
        candidate = raw[:, :, None] + lp
        selection = candidate if adjuster is None else candidate + adjuster(step + 1, paths.view(b, k, -1), lp)
        _, flat = selection.view(b, -1).topk(k, -1, sorted=True)
        parent, token = flat // v, flat % v
        raw = candidate.view(b, -1).gather(1, flat)
        global_parent = (rows * k + parent).reshape(-1)
        paths = torch.cat((paths[global_parent], token.reshape(-1, 1)), -1)
        if trace is not None:
            trace(step + 1, paths.view(b, k, -1), raw)
        if step < 2 and not first:
            # Within one user all beams have identical encoder cross-attention KV.
            # Self-attention history differs, so it must follow the selected parent.
            cache = reorder_self_cache(cache, global_parent)
    return paths


@torch.inference_mode()
def generate_fast(tiger, input_ids, attention_mask, prior, widths=(15, 15, 15)):
    prior.prepare(input_ids, attention_mask)
    return fast_search(tiger, input_ids, attention_mask, widths, prior)
