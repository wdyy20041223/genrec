"""Real-data bitwise checks of prior tensors, selected paths and FP32 scores."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import numpy as np
import torch

from cotiger.core.models.tiger import Tiger
from .beam_diagnosis import BASE, TaggedValidation, collate_tagged
from .finetune import file_sha256, make_loader
from .inference_fast import CPUCompactPrior, CSRGPUPrior, fast_search
from .inference_lab import CHECKPOINT_HASH, alternative_search, new_json
from .inference_priors import PrefixPrior
from .inference_weight_cache import FrozenLinearCastCache
from .profile_prior_speed import STUDY


@torch.inference_mode()
def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    checkpoint = BASE/'run/tiger/best_model.pt'
    assert file_sha256(checkpoint) == CHECKPOINT_HASH
    tiger = Tiger(json.loads((BASE/'analysis/model_config.json').read_text())).eval().requires_grad_(False).cuda()
    tiger.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'), strict=True)
    sids = np.load(BASE/'analysis/item_semantic_ids.npy', allow_pickle=False)
    with np.load(STUDY/'train_statistics.npz', allow_pickle=False) as f:
        stats = {k:f[k] for k in f.files}
    old = PrefixPrior(stats, sids+[1,257,513], [0.,.4,.4], True, 3)
    new = (CPUCompactPrior if args.backend == 'cpu' else CSRGPUPrior)(old)
    original_parameters = {name:value.detach().cpu().clone() for name,value in tiger.state_dict().items()}
    dataset = TaggedValidation(BASE/'analysis/valid_samples.parquet', sids, 256)
    loader = make_loader(dataset, 64, collate_tagged, False, 0, 42, True)
    checked_adjustments, checked_scores, total = 0, 0, 0
    with torch.autocast('cuda', dtype=torch.float16):
        for batch in loader:
            ids, mask = batch['input_ids'].cuda(), batch['attention_mask'].cuda()
            old.prepare(ids, mask)
            new.prepare(ids, mask)
            steps = []
            def record(**t):
                nonlocal checked_adjustments
                depth, b, k = t['depth'], t['batch_size'], t['num_beams']
                parents = t['parent_tokens'].view(b,k,depth)
                flat = t['selected_flat_indices']
                parent, token = flat//769, flat%769
                prefix = parents.gather(1,parent[:,:,None].expand(-1,-1,depth))
                paths = torch.cat((prefix,token[:,:,None]),-1)
                if depth > 1:
                    expected = old(depth,parents,t['next_token_logprobs'])
                    actual = new(depth,parents,t['next_token_logprobs'])
                    assert torch.equal(actual,expected), 'Prior changed a full-vocabulary candidate score'
                    checked_adjustments += actual.numel()
                steps.append((paths.clone(),t['selected_cumulative_scores'].clone()))
            expected = alternative_search(tiger,ids,mask,[args.beam]*3,recorder=record,adjuster=old)[0]
            def check(depth, paths, scores):
                nonlocal checked_scores
                assert torch.equal(paths,steps[depth-1][0]), 'Selected parent/path order changed'
                assert torch.equal(scores,steps[depth-1][1]), 'Raw FP32 cumulative score changed'
                checked_scores += scores.numel()
            with FrozenLinearCastCache(tiger) if args.cache_linear else nullcontext():
                actual = fast_search(tiger,ids,mask,[args.beam]*3,new,trace=check)
            assert torch.equal(actual,expected)
            total += len(ids)
    assert total == 256 and file_sha256(checkpoint) == CHECKPOINT_HASH
    assert all(torch.equal(value.detach().cpu(),original_parameters[name]) for name,value in tiger.state_dict().items())
    new_json(args.output,dict(state='passed',samples=total,beam=args.beam,batch_size=64,
        candidate_prior_values_checked=checked_adjustments,raw_fp32_scores_checked=checked_scores,
        prior_bitwise_equal=True,raw_cumulative_scores_bitwise_equal=True,
        all_three_depth_paths_and_order_bitwise_equal=True,log_softmax_dtype='float32',
        linear_cast_cache=args.cache_linear,prior_backend=args.backend,original_fp32_parameters_bitwise_unchanged=True,
        no_training=True,checkpoint_sha256=CHECKPOINT_HASH))
    print(json.dumps(dict(state='passed',beam=args.beam,checked_adjustments=checked_adjustments,checked_scores=checked_scores)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--beam',type=int,choices=(15,30),default=15)
    p.add_argument('--cache-linear',action='store_true')
    p.add_argument('--backend',choices=('cpu','gpu'),default='gpu')
    run(p.parse_args())
