"""Create a small, self-contained results package after audited evaluation."""
import csv
import gzip
import hashlib
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    audit = json.loads((ROOT / 'results/audit.json').read_text())
    assert audit['status'] == 'PASS_COMPLETE_TWO_MODELS_FULL3905_N20_R20_K20'
    independent = json.loads((ROOT / 'voting/final_results_independent_review.json').read_text())
    assert independent['status'] == 'PASS_INDEPENDENT_FINAL_RESULTS_REVIEW'
    metrics = json.loads((ROOT / 'results/metrics.json').read_text())
    voting = json.loads((ROOT / 'voting/audit.json').read_text())
    properties = json.loads((ROOT / 'properties/reports/final_audit.json').read_text())
    by_model = {model: {} for model in ['gellmo_p6_mistral', 'gellmo_p6_llama']}
    for row in metrics['macro']:
        if row['scope'] == 'Macro10':
            by_model[row['model_id']][row['readout']] = row
    paired = {model: Counter() for model in by_model}
    with gzip.open(str(ROOT / 'results/paired_outcomes.jsonl.gz'), 'rt') as f:
        for line in f:
            r = json.loads(line)
            a, b, c = r['N20_K1_success'], r['N20_K20_success'], r['N400_K1_success']
            assert not a or b
            out = paired[r['model_id']]
            out['inputs'] += 1
            out['N20_K20_vs_N20_K1_gains'] += int(b and not a)
            out['N20_K20_vs_N20_K1_losses'] += int(a and not b)
            out['N20_K20_vs_N400_K1_gains'] += int(b and not c)
            out['N20_K20_vs_N400_K1_losses'] += int(c and not b)
            out['N20_K1_vs_N400_K1_gains'] += int(a and not c)
            out['N20_K1_vs_N400_K1_losses'] += int(c and not a)
    assert all(r['inputs'] == 3905 for r in paired.values())
    (ROOT / 'results/paired_summary.json').write_text(json.dumps(paired, indent=2) + '\n')
    views = [('N20_R1_K1', '20票，一轮，K1（固定第一组）'),
             ('N20_R20_K20', '20票，20轮，K20'),
             ('N400_R1_K1_reference', '400票，一轮，K1（旧配置参考）')]
    lines = ['Exact Voting：两模型、完整3,905输入、N20/R20实验结果',
             '生成时间UTC：' + datetime.now(timezone.utc).isoformat(), '',
             '1. 已完成范围',
             '两个模型各3,905个输入；每个输入20个实际投票赢家，共156,200个终端槽位。',
             '原有400条提案按原始20次生成调用分组，每组20票，独立计票。未新增语言模型生成。',
             '完整检查了3,124,000条原始提案的缓存哈希、输入、顺序、分组与种子身份。',
             '性质请求去重后共{}个；复用{}个，CPU补评{}个，未知性质缺失0个。'.format(
                 properties['unique_requests'], properties['reused_requests'], properties['new_requests']), '',
             '2. 成功率（10个任务等权平均，单位%）',
             '配置\tMistral\tLlama']
    for view, label in views:
        lines.append('{}\t{:.4f}\t{:.4f}'.format(label, by_model['gellmo_p6_mistral'][view]['SR'], by_model['gellmo_p6_llama'][view]['SR']))
    lines += ['', '这里N20/K1与N20/K20来自同一配置，K1固定取第一组赢家。',
              'N400/K1保留为不同内部预算的参考，不是N20/K20的同配置K1。',
              'K20成功指20个完整投票输出中至少一个成功；不代表部署时已选出了成功分子。', '',
              '3. 逐输入配对检查（单位：输入数，不是Macro10百分点）']
    for model in by_model:
        r = paired[model]
        short = model.rsplit('_', 1)[1]
        v = voting['results'][short]
        lines += [model + ':',
                  'N20/K20 对同配置N20/K1：新增成功{}，丢失成功{}。'.format(r['N20_K20_vs_N20_K1_gains'], r['N20_K20_vs_N20_K1_losses']),
                  'N20/K20 对旧N400/K1：新增成功{}，丢失成功{}。'.format(r['N20_K20_vs_N400_K1_gains'], r['N20_K20_vs_N400_K1_losses']),
                  '20组赢家均同一分子的输入：{}/3905；平均不同赢家身份：{:.4f}。'.format(v['all20_winners_same_identity_inputs'], v['mean_distinct_winner_identities']),
                  '平票组：{}/78100；所有票身份均不同的组：{}；无效赢家组：{}（保留，不替换）。'.format(v['tied_max_groups'], v['all_unique_groups'], v['invalid_winner_groups'])]
    lines += ['', '4. 方法和评价边界',
              '保持既有分子版Exact Voting的分子身份归并、频次投票及最早出现平票规则。性质不参与投票。',
              '无效、空输出、重复赢家均保留，不补采样，不按性质筛选或重分组。',
              '这是N=20的预算设置，不能称作400票投票重复20次。',
              '原Vote论文允许可变采样数；本文没有复刻其从400缓存有放回抽样的缩放曲线估计。',
              '保持原GeLLMO任务、严格方向改善规则及原候选性质两位小数比较精度。',
              'Sim/SAS/RI/Novelty/APS只在成功输入上计算；K20使用原代码选出的同一个成功代表。',
              'RI采用原代码（包括源性质为0的分支）；Sim/SAS使用未四舍五入值。',
              'Novelty使用旧RDKit规范化输出与原参考字符串集合比较。APS按性质分列，不混合单位。',
              '同时保留旧/新RDKit的输入级有效率；有效率不等于所有输出槽位中有效分子所占比例。',
              '一份预先固定的分组实现；没有选择成绩较好的分组或种子，不声称多随机种子均值。', '',
              '5. 文件',
              'results/per_task.csv：60行，两模型×三种读出×十任务。',
              'results/macro.csv：IND、OOD、Macro10汇总，以及原GeLLMO补充指标。',
              'results/aps_by_task.csv：逐任务、逐性质APS。',
              'results/per_input.jsonl.gz：23,430条逐输入评价记录及代表分子。',
              'results/paired_summary.json：新旧配置逐输入得失。',
              'voting/grouped_inputs_*.jsonl.gz：固定赢家及原提案位置，可审计。',
              'properties/results/winner_properties.jsonl.gz：赢家性质与来源证据。',
              'results/audit.json、voting/audit.json、properties/reports/final_audit.json：完成校验。', '',
              '该包只代表P10的双模型N20/R20已完成，不代表九种方法的全部K20实验均已完成。',
              '现有K1论文结果及Overleaf文件未被本次修改。']
    (ROOT / '结果说明.txt').write_text('\n'.join(lines) + '\n')
    files = [ROOT / name for name in ['protocol.json', 'original_input_hashes.json', '结果说明.txt',
                                      'evaluate_votes.py', 'package_results.py', 'run_evaluation.sbatch',
                                      'voting/extract.py', 'voting/audit.json', 'voting/output_manifest.json',
                                      'voting/source_provenance.json', 'voting/tie_diagnostics.json',
                                      'voting/evaluator_independent_review.json',
                                      'voting/final_results_independent_review.json', 'voting/verify_results.py',
                                      'properties/reports/inventory.json', 'properties/reports/final_audit.json',
                                      'properties/reports/raw_cache_difference_audit.json',
                                      'properties/results/winner_properties.jsonl.gz']]
    files += sorted((ROOT / 'results').glob('*'))
    files += sorted((ROOT / 'voting').glob('grouped_inputs_*.jsonl.gz'))
    files += sorted((ROOT / 'properties/code').glob('*.py'))
    files += sorted((ROOT / 'properties/config').glob('*.json'))
    files = sorted(set(files))
    assert all(p.is_file() for p in files), [str(p) for p in files if not p.is_file()]
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    files += [ROOT / 'artifact_manifest.json']
    target = ROOT / 'p10_n20_r20_results.zip'
    with zipfile.ZipFile(str(target), 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(str(p), str(p.relative_to(ROOT)))
    with zipfile.ZipFile(str(target)) as z:
        assert z.testzip() is None
    receipt = dict(status='PASS', path=str(target), files=len(files), size_bytes=target.stat().st_size,
                   sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (ROOT / 'package_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
