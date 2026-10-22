"""CPU-only audit of cached CoTIGER scores; no model forward or source edits."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from cotiger.ids import RADIX, SID_SPACE, lookup, sid_keys
from cotiger.prior import PrefixPrior


from .paths import ROOT, STUDY


def distribution(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    assert len(values) and np.isfinite(values).all()
    return dict(count=len(values), mean=float(values.mean()),
                p10=float(np.quantile(values, .1)), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), p95=float(np.quantile(values, .95)),
                min=float(values.min()), max=float(values.max()))


class Rewards:
    def __init__(self, prior):
        self.prior, self.indexes = prior, []
        for depth, grouped in enumerate(prior.by_depth, 1):
            keys, values = [], []
            for context, (tokens, lifts) in grouped.items():
                codes = np.zeros(len(tokens), np.int64)
                for position in range(depth):
                    codes = codes * RADIX + tokens[:, position]
                keys.append(context * RADIX**depth + codes)
                values.append(lifts)
            keys, values = np.concatenate(keys), np.concatenate(values)
            order = keys.argsort()
            self.indexes.append((keys[order], values[order]))

    def get(self, tokens, history, depth):
        candidates = np.zeros(tokens.shape[:2], np.int64)
        for position in range(depth):
            candidates = candidates * RADIX + tokens[:, :, position]
        keys, values = self.indexes[depth-1]
        bonus, weighted = np.zeros(candidates.shape, np.float32), np.zeros(candidates.shape, np.float32)
        alpha = self.prior.strengths[depth-1]
        for age in range(3):
            context = history[:, -age-1]
            active = context != 0
            if age:
                active &= (context[:, None] != history[:, -age:]).all(-1)
            lift = lookup(keys, values, context[:, None] * RADIX**depth + candidates)
            lift = np.where(active[:, None], lift, np.float32(0))
            bonus += .5**age * lift
            weighted += alpha * .5**age * lift
        return bonus, weighted

    def seen(self, tokens, history):
        keys = sid_keys(tokens)
        singleton = lookup(self.prior.catalog_keys, self.prior.group_sizes, keys) == 1
        return singleton & ((keys[:, :, None] == history[:, None, :]) &
                            (history[:, None, :] != 0)).any(-1)


def components(lm, bonus, reward):
    return dict(lm=distribution(lm), bonus=distribution(bonus),
                weighted_bonus=distribution(reward),
                nonzero_reward_fraction=float((reward > 0).mean()),
                reward_exceeds_abs_lm_fraction=float((reward > np.abs(lm)).mean()),
                mean_reward_over_mean_abs_lm=float(reward.mean(dtype=np.float64)/np.abs(lm).mean(dtype=np.float64)))


def analyze(folder, beam, rewards):
    meta = json.loads((folder / 'summary.json').read_text())
    assert meta['samples'] == 22363 and meta['widths'] == [beam]*3
    assert meta['prior_strengths'] == [0., .4, .4] and meta['prior_history'] == 3
    with np.load(folder / 'candidates.npz', allow_pickle=False) as cache:
        tokens, logps, raw, score = [cache[key] for key in ('tokens', 'logps', 'raw_scores', 'search_scores')]
    with np.load(folder / 'contexts.npz', allow_pickle=False) as cache:
        histories, ids, targets = [cache[key] for key in ('histories', 'sample_ids', 'targets')]
    assert np.array_equal(logps.sum(-1), raw) and (logps <= 0).all()
    with np.load(STUDY / f'production_benchmark_valid/prior{beam}_predictions.npz', allow_pickle=False) as old:
        assert np.array_equal(tokens[:, :beam], old['tokens'])
    history = sid_keys(histories)
    result = dict(beam=beam, samples=len(ids), cohort='Final Top10 SID paths, including their own D1/D2 prefixes; repeated prefixes remain path-weighted',
                  top10_prefixes={}, source=str(folder))
    for depth in (1, 2, 3):
        bonus, reward = rewards.get(tokens[:, :10], history, depth)
        result['top10_prefixes'][str(depth)] = components(logps[:, :10, :depth].sum(-1), bonus, reward)
        # Independent check against the unmodified reference reward implementation.
        n, k = 32, 10
        input_ids = torch.from_numpy(histories[:n].astype(np.int64).reshape(n, -1))
        rewards.prior.prepare(input_ids, input_ids.ne(0).long())
        parent = np.concatenate((np.zeros((n,k,1), np.int64), tokens[:n,:k,:depth-1]), -1)
        expected = rewards.prior(depth, torch.from_numpy(parent), torch.zeros(n,k,769))
        actual = reward[:n].copy()
        if depth == 3:
            actual -= np.float32(1000) * rewards.seen(tokens[:n,:k], history[:n])
        if torch.is_tensor(expected):
            expected = expected.gather(-1, torch.from_numpy(tokens[:n,:k,depth-1:depth].astype(np.int64))).squeeze(-1).numpy()
            assert np.array_equal(expected, actual), (beam, depth, abs(expected-actual).max())
        else:
            assert np.all(actual == expected)
    reward = np.empty_like(raw)
    seen = np.empty(raw.shape, bool)
    for start in range(0, len(ids), 256):
        sl = slice(start, start+256)
        _, reward[sl] = rewards.get(tokens[sl], history[sl], 3)
        seen[sl] = rewards.seen(tokens[sl], history[sl])
    reconstructed = raw + (reward - np.float32(1000)*seen)
    assert np.array_equal(reconstructed, score), float(abs(reconstructed-score).max())
    result['all_256_pool_scores_bitwise_reconstructed'] = True
    result['seen_top10_paths'] = int(seen[:, :10].sum())
    result['final_top1'] = components(raw[:, :1], reward[:, :1]/.4, reward[:, :1])
    result['all_saved_256_final_expansions'] = components(raw, reward/.4, reward)
    lm, r = raw[:, :10], reward[:, :10]
    i, j = np.triu_indices(10, 1)
    lm_gap = lm[:, i] - lm[:, j]
    score_gap = score[:, i] - score[:, j]
    comparable = (np.abs(lm_gap) > 1e-5) & (score_gap > 1e-5)
    result['top10_order'] = dict(lm_span=distribution(np.ptp(lm, axis=1)),
                                 reward_span=distribution(np.ptp(r, axis=1)),
                                 reward_span_exceeds_lm_span_fraction=float((np.ptp(r, axis=1)>np.ptp(lm,axis=1)).mean()),
                                 unambiguous_pairs=int(comparable.sum()),
                                 reversed_pairs=int(((lm_gap < -1e-5)&comparable).sum()),
                                 reversed_pair_fraction=float(((lm_gap < -1e-5)&comparable).sum()/comparable.sum()))
    # Remove D3 reward only, retain identical D2 parents and seen-item filtering.
    base = raw - np.float32(1000)*seen
    order = np.argsort(-base, axis=1, kind='stable')
    rows = np.arange(len(ids))
    best, tenth = base[rows,order[:,0]], base[rows,order[:,9]]
    cutoff = score[:, -1]
    exact1, exact10 = best > cutoff+1e-4, tenth > cutoff+1e-4
    changed1 = best - base[:,0] > 1e-5
    changed_set = ~(np.sort(order[:,:10],axis=1) == np.arange(10)).all(1)
    strict_set_change = base[:,10:].max(1) - base[:,:10].min(1) > 1e-5
    result['remove_d3_reward_same_parents'] = dict(
        guaranteed_exact_top1_rows=int(exact1.sum()), guaranteed_exact_top10_rows=int(exact10.sum()),
        clear_top1_changed_rows=int((changed1&exact1).sum()),
        clear_top1_changed_fraction=float(changed1[exact1].mean()),
        stable_tie_top10_set_changed_rows=int((changed_set&exact10).sum()),
        clear_top10_set_changed_rows=int((strict_set_change&exact10).sum()),
        clear_top10_set_changed_fraction=float(strict_set_change[exact10].mean()),
        definition='Raw LM minus seen-singleton penalty. Saved-pool completeness proved against omitted-candidate selection-score upper bound; fixed actual D2 parents, not full TIGER-only decoding.')
    is_gold = (tokens[:,:10] == targets[:,None]).all(-1)
    result['final_top10_gold_paths'] = components(lm[is_gold], r[is_gold]/.4, r[is_gold])
    cases = np.flatnonzero(changed1 & exact1 & ~seen[:,0])[:3]
    result['examples'] = []
    for row in cases:
        candidates = []
        for col in (0, int(order[row,0])):
            candidates.append(dict(tokens=tokens[row,col].tolist(), lm=float(raw[row,col]),
                                   bonus=float(reward[row,col]/.4), weighted_bonus=float(reward[row,col]),
                                   score=float(score[row,col]), seen=bool(seen[row,col]),
                                   observed_target=bool(np.array_equal(tokens[row,col],targets[row]))))
        result['examples'].append(dict(sample_id=str(ids[row]), calibrated_winner=candidates[0],
                                       lm_winner_with_same_d2_parents=candidates[1]))
    print(json.dumps(dict(beam=beam, top10_prefixes=result['top10_prefixes'],
                          top10_order=result['top10_order'], counterfactual=result['remove_d3_reward_same_parents'])), flush=True)
    return result


def main():
    global STUDY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, default=STUDY)
    parser.add_argument('--sids', type=Path, default=ROOT/'artifacts/beauty/item_semantic_ids.npy')
    parser.add_argument('--output', type=Path, default=ROOT/'out/tiger_prefix_ft/beauty_score_components_20261017')
    args = parser.parse_args()
    STUDY = args.study_root.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    with np.load(STUDY/'train_statistics.npz', allow_pickle=False) as archive:
        statistics = {key:archive[key] for key in archive.files}
    sids = np.load(args.sids, allow_pickle=False)
    prior = PrefixPrior(statistics, sids+[1,257,513], [0.,.4,.4], True, 3)
    rewards = Rewards(prior)
    results = [analyze(STUDY/'valid'/name, beam, rewards) for name,beam in
               [('prior15_h3_d2_04_d3_04',15), ('prior_h3_d2_04_d3_04',30)]]
    summary = dict(state='passed', split='valid', samples=22363, no_model_execution=True,
                   no_training=True, strengths=[0,.4,.4], history_items=3,
                   filter_term='-1000 for seen singleton SID; separate from nonnegative historical bonus', results=results)
    with (args.output/'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary,stream,ensure_ascii=False,indent=2)
    with (args.output/'prefix_statistics.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['beam','depth','paths','lm_mean','lm_p10','lm_median','lm_p90','bonus_mean',
                         'weighted_bonus_mean','weighted_bonus_p10','weighted_bonus_median','weighted_bonus_p90',
                         'nonzero_reward_fraction','mean_reward_over_mean_abs_lm'])
        for result in results:
            for depth,values in result['top10_prefixes'].items():
                a,b,c=values['lm'],values['bonus'],values['weighted_bonus']
                writer.writerow([result['beam'],depth,a['count'],a['mean'],a['p10'],a['median'],a['p90'],b['mean'],
                                 c['mean'],c['p10'],c['median'],c['p90'],values['nonzero_reward_fraction'],
                                 values['mean_reward_over_mean_abs_lm']])


if __name__ == '__main__':
    main()
