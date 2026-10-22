"""Train-count prefix calibration, conditioned only on observed recent SIDs."""
import numpy as np
import torch

from .inference_rerank import PREFIX_SPACE, RADIX, SID_SPACE, lookup, sid_keys


def decode(keys):
    return np.stack((keys // RADIX ** 2, keys // RADIX % RADIX, keys % RADIX), -1)


def aggregate(keys, counts):
    unique, inverse = np.unique(keys, return_inverse=True)
    return unique, np.bincount(inverse, weights=counts).astype(np.int64)


class PrefixPrior:
    """S(prefix)=LM cumulative score + alpha[d]*weighted train transition lift.

    Prefix bonuses REPLACE the previous depth's bonus; they never accumulate.
    The constant depending only on the user context is omitted from PMI.
    """
    def __init__(self, stats, catalog_tokens, strengths, suppress_seen=False, history_items=1):
        if history_items not in (1, 3):
            raise ValueError("history_items must be 1 or 3")
        self.strengths = strengths
        self.suppress_seen = suppress_seen
        self.history_items = history_items
        self.catalog_keys, self.group_sizes = np.unique(sid_keys(catalog_tokens), return_counts=True)
        contexts = stats["sid_pair_keys"] // SID_SPACE
        full = stats["sid_pair_keys"] % SID_SPACE
        pair_counts = stats["sid_pair_counts"]
        n = int(stats["samples"])
        self.by_depth = []
        for depth, space, target, global_keys, global_counts, catalog_size in (
            (1, RADIX, full // RADIX ** 2, np.arange(RADIX), stats["code1_counts"], len(np.unique(catalog_tokens[:, 0]))),
            (2, PREFIX_SPACE, full // RADIX, stats["prefix_target_keys"], stats["prefix_target_counts"], len(np.unique(self.catalog_keys // RADIX))),
            (3, SID_SPACE, full, stats["sid_target_keys"], stats["sid_target_counts"], len(self.catalog_keys)),
        ):
            pair, count = aggregate(contexts * space + target, pair_counts)
            context, candidate = pair // space, pair % space
            marginal = (lookup(global_keys, global_counts, candidate) + 1) / (n + catalog_size)
            lift = np.log1p(count / (20 * marginal)).astype(np.float32)
            changes = np.r_[0, np.flatnonzero(context[1:] != context[:-1]) + 1, len(context)]
            grouped = {}
            for start, end in zip(changes[:-1], changes[1:]):
                codes = candidate[start:end]
                if depth == 1:
                    tokens = codes[:, None]
                elif depth == 2:
                    tokens = np.stack((codes // RADIX, codes % RADIX), -1)
                else:
                    tokens = decode(codes)
                grouped[int(context[start])] = (tokens, lift[start:end])
            self.by_depth.append(grouped)

    def prepare(self, input_ids, attention_mask):
        """Observed encoder input only; no target/label argument exists."""
        self.contexts, self.seen = [], []
        for ids, mask in zip(input_ids.cpu().numpy(), attention_mask.cpu().numpy()):
            history = ids[mask.astype(bool)].reshape(-1, 3)
            contexts, used = [], set()
            for age, context in enumerate(sid_keys(history[-self.history_items:])[::-1]):
                if int(context) not in used:
                    contexts.append((int(context), .5 ** age))
                    used.add(int(context))
            self.contexts.append(contexts)
            keys = np.unique(sid_keys(history))
            sizes = lookup(self.catalog_keys, self.group_sizes, keys)
            self.seen.append(decode(keys[sizes == 1]))

    def __call__(self, depth, parent_paths, probabilities):
        if not self.strengths[depth - 1] and not (depth == 3 and self.suppress_seen):
            return 0.0
        b, k, _ = parent_paths.shape
        adjustment = np.zeros((b, k, probabilities.shape[-1]), np.float32)
        parents = parent_paths[:, :, 1:].detach().cpu().numpy()
        alpha = self.strengths[depth - 1]
        for row, contexts in enumerate(self.contexts):
            for context, weight in contexts:
                if alpha and context in self.by_depth[depth - 1]:
                    tokens, lifts = self.by_depth[depth - 1][context]
                    match = (parents[row, :, None, :] == tokens[None, :, :depth - 1]).all(-1)
                    beam, column = np.nonzero(match)
                    adjustment[row, beam, tokens[column, -1]] += alpha * weight * lifts[column]
            if depth == 3 and self.suppress_seen:
                tokens = self.seen[row]
                match = (parents[row, :, None, :] == tokens[None, :, :2]).all(-1)
                beam, column = np.nonzero(match)
                adjustment[row, beam, tokens[column, -1]] -= 1000.
        return torch.from_numpy(adjustment).to(probabilities.device)
