"""Audit existing Beauty RQ-VAE assignments on CPU; no inference or training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


from .paths import ROOT, BASE as BEAUTY_BASE
BASE = BEAUTY_BASE / 'analysis'
OUTPUT = ROOT / 'out/tiger_prefix_ft/beauty_codebook_usage_20261020'
DOCS = ROOT / 'docs/generated/tiger_prefix_ft_beauty_20261013'
REPORT = DOCS / 'codebook_usage_20261020.md'
FIGURE = DOCS / 'codebook_usage_20261020.png'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(counts):
    counts = np.asarray(counts, dtype=np.int64)
    p = counts / counts.sum()
    positive = p[p > 0]
    entropy = float(-(positive * np.log2(positive)).sum())
    ordered = np.sort(counts)
    gini = float((2 * (np.arange(1,257) * ordered).sum()) / (256 * counts.sum()) - 257/256)
    return dict(total_assignments=int(counts.sum()), used_codes=int((counts>0).sum()),
                unused_codes=np.flatnonzero(counts==0).tolist(),
                usage_fraction=float((counts>0).mean()), entropy_bits=entropy,
                normalized_entropy=entropy/8, effective_codes=float(2**entropy),
                effective_fraction=float(2**entropy/256), gini=gini,
                min_count=int(counts.min()), min_nonzero_count=int(counts[counts>0].min()),
                max_count=int(counts.max()), mean_count=float(counts.mean()),
                count_cv=float(counts.std()/counts.mean()), top_code=int(counts.argmax()),
                top_code_fraction=float(p.max()), top10_fraction=float(ordered[-10:].sum()/counts.sum()))


def main():
    global BASE, OUTPUT, REPORT, FIGURE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', type=Path, default=BASE,
                        help='Original export, including codebook vectors and checkpoint SID snapshots')
    parser.add_argument('--sids', type=Path, default=ROOT/'artifacts/beauty/item_semantic_ids.npy')
    parser.add_argument('--train', type=Path, default=ROOT/'artifacts/beauty/train_samples.parquet')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    BASE, OUTPUT, REPORT = args.analysis_dir, args.output_dir, args.report
    FIGURE = args.figure or REPORT.with_suffix('.png')
    if OUTPUT.exists() or REPORT.exists() or FIGURE.exists():
        raise FileExistsError('Choose new output paths; existing analysis is preserved')
    inputs = dict(sids=args.sids,
                  original_sids=BASE/'item_semantic_ids.npy',
                  train=args.train,
                  epochs=BASE/'rqvae_checkpoint_epochs.npy',
                  checkpoint_sids=BASE/'rqvae_checkpoint_semantic_ids.npy',
                  codebooks=BASE/'rqvae_codebooks.npy')
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Full codebook audit requires original RQ-VAE exports; see reproduction.md. Missing: ' + ', '.join(missing))
    hashes = {name:sha(path) for name,path in inputs.items()}
    sids = np.load(inputs['sids'], allow_pickle=False)
    snapshots = np.load(inputs['checkpoint_sids'], allow_pickle=False)
    epochs = np.load(inputs['epochs'], allow_pickle=False)
    assert sids.shape == (12101,3) and sids.min() >= 0 and sids.max() < 256
    assert np.array_equal(sids, np.load(inputs['original_sids'], allow_pickle=False))
    assert np.array_equal(sids, snapshots[-1]) and epochs[-1] == 2000
    train = pq.read_table(inputs['train'], columns=['sample_id','target_sid','target_tokens'])
    assert len(train)==131413 and all(x.startswith('train_') for x in train['sample_id'].to_pylist())
    targets = np.asarray(train['target_sid'].to_pylist())
    assert np.array_equal(targets+[1,257,513], np.asarray(train['target_tokens'].to_pylist()))
    codebooks = np.load(inputs['codebooks'], allow_pickle=False).astype(np.float64)
    assert codebooks.shape == (3,256,32) and np.isfinite(codebooks).all()
    counts = {name:[np.bincount(ids[:,d], minlength=256) for d in range(3)]
              for name,ids in [('catalog',sids),('train_targets',targets)]}
    summary = dict(state='passed', sid_length=3, codebook_size=256, catalog_items=12101,
                   train_targets=131413, rqvae_epoch=2000, no_model_execution=True, no_training=True,
                   raw_code_zero_is_valid=True, source_sha256=hashes,
                   scopes={name:[summarize(c) for c in layers] for name,layers in counts.items()})
    timeline=[]
    for epoch,ids in zip(epochs,snapshots):
        for d in range(3):
            row=dict(epoch=int(epoch), layer=d+1, **summarize(np.bincount(ids[:,d],minlength=256)))
            timeline.append(row)
    geometry=[]
    for d,vectors in enumerate(codebooks):
        distances=np.linalg.norm(vectors[:,None]-vectors[None,:],axis=-1)
        pairs=distances[np.triu_indices(256,1)]
        geometry.append(dict(layer=d+1,unique_vectors=len(np.unique(vectors,axis=0)),
                             centered_matrix_rank=int(np.linalg.matrix_rank(vectors-vectors.mean(0))),
                             min_pair_distance=float(pairs.min()), median_pair_distance=float(np.median(pairs))))
    summary['codebook_geometry']=geometry
    _,prefix_counts=np.unique(sids,axis=0,return_counts=True)
    summary['sid_collisions']=dict(unique_sids=len(prefix_counts), groups=int((prefix_counts>1).sum()),
        colliding_items=int(prefix_counts[prefix_counts>1].sum()),
        colliding_item_fraction=float(prefix_counts[prefix_counts>1].sum()/len(sids)),
        excess_assignments=int(len(sids)-len(prefix_counts)), max_group=int(prefix_counts.max()))
    summary['checkpoint_usage']=[dict(epoch=r['epoch'],layer=r['layer'],used_codes=r['used_codes'],
                                    effective_codes=r['effective_codes']) for r in timeline]
    OUTPUT.mkdir(parents=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/'summary.json').open('x',encoding='utf-8') as f:
        json.dump(summary,f,indent=2,ensure_ascii=False)
    with (OUTPUT/'code_counts.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['scope','layer','raw_code','offset_token','count','share'])
        for scope,layers in counts.items():
            for d,c in enumerate(layers):
                for code,count in enumerate(c):
                    writer.writerow([scope,d+1,code,code+256*d+1,int(count),float(count/c.sum())])
    with (OUTPUT/'checkpoint_usage.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['rqvae_epoch','layer','used_codes','effective_codes','entropy_bits'])
        for row in timeline:
            writer.writerow([row['epoch'],row['layer'],row['used_codes'],row['effective_codes'],row['entropy_bits']])
    fig,axes=plt.subplots(2,3,figsize=(14.5,7.2),sharex=True,layout='constrained')
    for row,(scope,layers) in enumerate(counts.items()):
        limit=max(float(c.max()/c.sum()*100) for c in layers)*1.23
        for d,c in enumerate(layers):
            ax=axes[row,d];v=summary['scopes'][scope][d]
            ax.bar(np.arange(256),c/c.sum()*100,width=1,color=['#cc624f','#477eac','#378c80'][d])
            ax.axhline(100/256,color='#555555',linestyle='--',linewidth=1,label='Uniform: 0.391%')
            if (c==0).any():
                ax.scatter(np.flatnonzero(c==0),np.zeros((c==0).sum()),marker='|',color='#a82929',s=15)
            ax.set(title=f"{'Catalog (one vote per item)' if row==0 else 'Train targets (popularity weighted)'} - D{d+1}",
                   ylim=(0,limit),xlim=(-2,257),ylabel='Assignment share (%)')
            ax.text(.98,.96,f"Used: {v['used_codes']}/256\nEffective: {v['effective_codes']:.1f}",
                    transform=ax.transAxes,ha='right',va='top',fontsize=10,
                    bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=3))
            ax.set_xticks([0,64,128,192,255]);ax.grid(axis='y',alpha=.15)
            if row==1:ax.set_xlabel('Raw code (0-255; zero is valid)')
            if row==0 and d==0:ax.legend(loc='upper left',fontsize=8)
    fig.suptitle('Beauty RQ-VAE code usage | checkpoint epoch 2000 | 12,101 catalog items',fontsize=15)
    fig.savefig(FIGURE,dpi=160);plt.close(fig)
    lines=['# Beauty 当前 Codebook 使用诊断（2026-10-20）','',
        '**第一层存在明显码字闲置/部分使用坍缩，第二层有一定频次偏斜，第三层整体接近均匀。**','',
        '## 数据与口径','',
        '- 当前实际用于TIGER/CoTIGER的三位SID，共12101个商品；每层256个码字，原始ID为0～255，原始0是合法码字。',
        '- SID与原云端导出及RQ-VAE第2000 epoch快照逐项一致，输入SHA256保存在summary.json。这里的epoch均指RQ-VAE，不是TIGER的100 epoch。',
        '- 主统计按商品计数，每个商品一票；另按131413条Train目标交互加权，后者会额外反映商品流行度。',
        '- 完全离线CPU统计，未加载模型、重新量化、训练或使用Validation/Test标签调整模型。','',
        '## 按商品统计（诊断码本利用率的主口径）','',
        '| 层 | 活跃码字 | 使用率 | 未用码字 | 熵（满值8bit） | 有效码字数（满值256） | 最高单码占比 | Top10码字占比 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for d,v in enumerate(summary['scopes']['catalog'],1):
        lines.append(f"| D{d} | {v['used_codes']} | {v['usage_fraction']*100:.2f}% | {len(v['unused_codes'])} | {v['entropy_bits']:.4f} | {v['effective_codes']:.2f} | {v['top_code_fraction']*100:.2f}% | {v['top10_fraction']*100:.2f}% |")
    lines+=['','有效码字数=2^H，其中H=-sum(p*log2(p))。均匀使用m个码字时有效数为m；它同时反映活跃数和频次均衡程度。完美均匀基准：单码0.390625%、Top10码3.90625%、每码约47.27件商品。','',
        '- 第一层只覆盖96个码字，160个未命中；有效数78.45，主要问题是大面积闲置，并非所有商品只挤到一两个码字。活跃码字中每码20～362件商品。',
        '- 第二层无未用码字，有效数235.81；每码8～127件商品，仍有一定不均衡。',
        '- 第三层无未用码字，有效数252.98；每码33～138件商品，整体接近均匀，但code147仍是热点（1.14%），不能称严格均匀。','',
        '## 训练目标交互加权','',
        '| 层 | 活跃码字 | 有效码字数 | 最高单码占比 | Top10码字占比 |','| --- | ---: | ---: | ---: | ---: |']
    for d,v in enumerate(summary['scopes']['train_targets'],1):
        lines.append(f"| D{d} | {v['used_codes']} | {v['effective_codes']:.2f} | {v['top_code_fraction']*100:.2f}% | {v['top10_fraction']*100:.2f}% |")
    lines+=['','交互加权后更偏斜，这包含用户行为/商品流行度效应，不应全部解释为RQ-VAE坍缩。','',
        '## 训练快照','',
        '| RQ-VAE epoch | D1活跃/有效数 | D2活跃/有效数 | D3活跃/有效数 |','| --- | ---: | ---: | ---: |']
    for epoch in epochs:
        rows=[r for r in timeline if r['epoch']==epoch]
        lines.append(f"| {epoch} | "+' | '.join(f"{r['used_codes']} / {r['effective_codes']:.2f}" for r in rows)+' |')
    lines+=['','第一层第600～2000 epoch始终只有96个活跃码字；本轮只能确认这种利用不足持续存在，不能据此认定更长训练、重启死码或修改损失一定有效。','',
        '## 几何与碰撞核查','',
        '- 三层各有256个数值不同的32维向量，中心化矩阵秩均为32。因此没有出现全部码字向量完全相同或整体降到低秩的情况；这不能排除局部近重复，也不能消除第一层使用坍缩。',
        '- 完整SID有11401种，525个碰撞组，1225件商品处于碰撞组（10.12%），最大组21件。不同层分别均匀不保证组合SID无碰撞，不能把全部碰撞单独归因于第一层。',
        '- 本报告中的“未用/死码”只表示当前完整商品目录eval分配未命中，不等于已证明训练时永远没有梯度。',
        '- 当前仓库RQ-VAE配置为前两层STE、末层Sinkhorn；源码训练分支为Sinkhorn指定均匀列边际，而导出的SID使用eval最近邻argmin。该设置与末层较均衡的现象相符，本轮没有进行因果消融。','',
        '## 图与结果','',
        '![Codebook usage](codebook_usage_20261020.png)','',
        '- 上排：商品目录；下排：Train目标交互；虚线为256码完全均匀的0.391%。',
        '- 统计与全部未用码字ID：`out/tiger_prefix_ft/beauty_codebook_usage_20261020/summary.json`。',
        '- 每码频次：同目录`code_counts.csv`；训练快照：`checkpoint_usage.csv`。',
        '- 复现脚本：`tiger_prefix_ft/analyze_codebook_usage.py`（运行前选择不存在的新输出路径，保留既有结果）。']
    with REPORT.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    assert hashes == {name:sha(path) for name,path in inputs.items()}
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
