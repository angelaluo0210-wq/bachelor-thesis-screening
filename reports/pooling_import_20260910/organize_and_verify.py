"""One-off, non-destructive import and repeatable audit of the September GPU batch.

Run from any directory with Python and the project's analysis dependencies.
No model download, inference, training, or test-split access is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.run_training_strategy_comparison import read_result

SOURCE = ROOT.parent / 'pooling实验结果' / '第一次实验结果-比较4种poolin'
DEST = ROOT / 'outputs/pooling_comparison/formal/20260905_122936'
REPORT = Path(__file__).resolve().parent
POOLINGS = ('cls', 'mean', 'attention', 'concat')
REQUIRED = ('config.json', 'run_summary.json', 'metrics.json', 'training_history.csv',
            'run.log', 'best_pr_auc_classifier_head.pt', 'validation_predictions.csv',
            'per_review_validation_metrics.csv')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory(base):
    return {p.relative_to(base).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
            for p in sorted(base.rglob('*')) if p.is_file()}


def close(actual, expected, context):
    # Existing reporting intentionally emits NaN for undefined ratios (e.g. zero positives predicted).
    if math.isnan(float(actual)) and math.isnan(float(expected)):
        return
    require(math.isclose(float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-10),
            f'{context}: {actual} != {expected}')


def verify_runs(base):
    manifests = [p for p in base.glob('*manifest*.json')
                 if read_json(p).get('status') == 'completed']
    require(len(manifests) == 1, 'Exactly one original completed manifest is required')
    manifest = read_json(manifests[0])
    require(set(manifest['runs']) == set(POOLINGS), 'Four completed pooling runs required')
    comparison = pd.read_csv(base / 'final_comparison.csv').set_index('Pooling')
    # Read validation identities only; test.csv is never opened.
    validation = pd.read_csv(ROOT / 'data/processed/splits/validation.csv',
                             keep_default_na=False, dtype=str)
    reports, reference_config, reference_predictions = {}, None, None
    shared = ('learning_rate', 'batch_size', 'epochs', 'max_length', 'dropout', 'seed',
              'threshold', 'optimizer', 'loss', 'pos_weight', 'scheduler', 'gradient_clipping',
              'model_name_or_path', 'tokenizer_name_or_path', 'split_dir', 'criteria_path', 'data_qc')
    for pooling in POOLINGS:
        entry = manifest['runs'][pooling]
        run = base / pooling / PurePosixPath(entry['run_directory']).name
        require(entry['completed'] is True, f'{pooling}: incomplete manifest entry')
        require(all((run / name).is_file() and (run / name).stat().st_size for name in REQUIRED),
                f'{pooling}: missing/empty required files')
        config, summary, metrics = [read_json(run / name) for name in REQUIRED[:3]]
        require(config['pooling'] == summary['pooling'] == metrics['pooling'] == pooling,
                f'{pooling}: identity mismatch')
        for key in ('smoke_test', 'dry_run', 'evaluate_test', 'test_evaluated'):
            require(config.get(key) is False, f'{pooling}: {key} must be false')
        require(summary['test_evaluated'] is False, 'Test must remain unevaluated')
        require(config['max_train_samples'] is None and config['max_validation_samples'] is None,
                'Formal runs must not be subsampled')
        require(config['epochs'] == 5 and config['seed'] == 42 and config['backbone_frozen'],
                'Formal protocol mismatch')
        require(config['environment']['device'] == 'cuda', 'Expected remote GPU experiment')
        require(config['checkpoint_selection_metric'] == 'validation_pr_auc', 'Wrong selection metric')
        current = {k: config[k] for k in shared}
        if reference_config is None:
            reference_config = current
        require(reference_config == current, 'Shared configuration differs across pooling runs')
        history = pd.read_csv(run / 'training_history.csv')
        require(history['epoch'].tolist() == [1, 2, 3, 4, 5], f'{pooling}: incomplete history')
        best = history.loc[history['pr_auc'].idxmax()]
        require(int(best['epoch']) == summary['best_pr_auc_epoch'] == metrics['best_epoch'],
                f'{pooling}: best epoch mismatch')
        saved = summary['best_checkpoint_metrics']
        for key, value in saved.items():
            close(metrics['overall_validation'][key], value, f'{pooling}/metrics/{key}')
            close(best[key], value, f'{pooling}/history/{key}')
        checkpoint = torch.load(run / 'best_pr_auc_classifier_head.pt',
                                map_location='cpu', weights_only=True)
        require(checkpoint['model_config']['pooling'] == pooling, 'Checkpoint pooling mismatch')
        require(checkpoint['epoch'] == summary['best_pr_auc_epoch'], 'Checkpoint epoch mismatch')
        require(checkpoint['training_arguments'] == {k: config[k] for k in checkpoint['training_arguments']},
                'Checkpoint/config arguments differ')
        for key, value in saved.items():
            close(checkpoint['validation_metrics'][key], value, f'{pooling}/checkpoint/{key}')
        dim = 2304 if pooling == 'concat' else 768
        head = torch.nn.Linear(dim, 1)
        head.load_state_dict(checkpoint['classifier_head_state_dict'], strict=True)
        parameters = sum(t.numel() for t in head.parameters())
        if pooling == 'attention':
            scorer = torch.nn.Linear(768, 1, bias=False)
            scorer.load_state_dict(checkpoint['attention_pooling_state_dict'], strict=True)
            parameters += sum(t.numel() for t in scorer.parameters())
        else:
            require(checkpoint['attention_pooling_state_dict'] is None, 'Unexpected attention state')
        require(parameters == config['trainable_parameters'], 'Checkpoint parameter count mismatch')
        require(all(torch.isfinite(t).all().item() for t in head.parameters()), 'Nonfinite head weights')
        pred = pd.read_csv(run / 'validation_predictions.csv', keep_default_na=False, dtype=str)
        identities = pred.drop(columns=['predicted_probability', 'predicted_label'])
        if reference_predictions is None:
            reference_predictions = identities
        pd.testing.assert_frame_equal(identities, reference_predictions)
        mapped = validation.rename(columns={'review_key': 'review_name', 'label_included': 'true_label'})
        pd.testing.assert_frame_equal(identities, mapped[identities.columns], check_names=True)
        labels = pred['true_label'].astype(int).to_numpy()
        probabilities = pred['predicted_probability'].astype(float).to_numpy()
        require(np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all(),
                'Invalid prediction probabilities')
        predicted = probabilities >= config['threshold']
        require(np.array_equal(predicted, pred['predicted_label'].astype(int)), 'Threshold mismatch')
        recalculated = {'pr_auc': average_precision_score(labels, probabilities),
                        'roc_auc': roc_auc_score(labels, probabilities),
                        'tp': int(((labels == 1) & predicted).sum()),
                        'tn': int(((labels == 0) & ~predicted).sum()),
                        'fp': int(((labels == 0) & predicted).sum()),
                        'fn': int(((labels == 1) & ~predicted).sum())}
        for key, value in recalculated.items():
            close(value, saved[key], f'{pooling}/predictions/{key}')
        per_review = pd.read_csv(run / 'per_review_validation_metrics.csv').set_index('review_name')
        require(set(per_review.index) == set(pred['review_name']), 'Review identities differ')
        for item in metrics['per_review_validation']:
            group = pred[pred['review_name'] == item['review_name']]
            for key, value in item.items():
                if key != 'review_name':
                    close(per_review.loc[item['review_name'], key], value, f'{pooling}/per-review/{key}')
            close(average_precision_score(group['true_label'].astype(int),
                                          group['predicted_probability'].astype(float)),
                  item['pr_auc'], f'{pooling}/per-review/AP')
        require('Training complete' in (run / 'run.log').read_text(encoding='utf-8'), 'No completion log')
        row = comparison.loc['CLS' if pooling == 'cls' else pooling.title()]
        for label, key in [('PR-AUC', 'pr_auc'), ('ROC-AUC', 'roc_auc'), ('Recall', 'recall'),
                           ('FNR', 'fnr'), ('Precision', 'precision'), ('F1', 'f1')]:
            close(row[label], saved[key], f'{pooling}/comparison/{key}')
        reports[pooling] = {'run_directory': run.relative_to(base).as_posix(),
                            'best_epoch': int(best['epoch']), 'validation_rows': len(pred),
                            'metrics': saved, 'checkpoint_tensor_load': 'passed',
                            'validation_identity_matches_local_split': True}
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.verify_only:
        require(not DEST.exists(), f'Refusing to overwrite existing destination: {DEST}')
        source_files = inventory(SOURCE)
        existing = inventory(ROOT / 'outputs')
        source_runs = verify_runs(SOURCE / 'pooling_comparison')
        print('Source audit passed for all four pooling runs', flush=True)
        # Save the previous project outputs inventory before any result mutation.
        write_json(REPORT / 'existing_outputs_before.json', existing)
        by_hash = defaultdict(list)
        for path, info in existing.items():
            by_hash[info['sha256']].append(path)
        mapping, duplicates = [], defaultdict(list)
        for relative, info in source_files.items():
            original = SOURCE / relative
            path = Path(relative)
            if '.ipynb_checkpoints' in path.parts:
                target = Path('archive/notebook_autosaves') / path.relative_to('pooling_comparison')
                category = 'notebook_autosave'
            elif original.suffix == '.json' and 'manifest' in original.name:
                status = read_json(original)['status']
                target = Path(original.name) if status == 'completed' else Path('archive/failed_attempts') / original.name
                category = 'completed_manifest' if status == 'completed' else 'failed_attempt'
            elif relative == 'pooling_comparison.log':
                target, category = Path('logs/pooling_comparison.log'), 'batch_log'
            else:
                target, category = path.relative_to('pooling_comparison'), 'formal_result'
            mapping.append({'source': relative, 'destination': target.as_posix(), 'category': category,
                            **info, 'source_mtime_ns': original.stat().st_mtime_ns,
                            'existing_project_duplicates': by_hash.get(info['sha256'], [])})
            duplicates[info['sha256']].append(relative)
        require(len({item['destination'] for item in mapping}) == len(mapping), 'Destination collision')
        DEST.mkdir(parents=True, exist_ok=False)
        for item in mapping:
            target = DEST / item['destination']
            target.parent.mkdir(parents=True, exist_ok=True)
            require(not target.exists(), f'Refusing overwrite: {target}')
            shutil.copy2(SOURCE / item['source'], target)
        write_json(DEST / 'import_manifest.json', {
            'imported_utc': datetime.now(timezone.utc).isoformat(), 'source': str(SOURCE),
            'destination': str(DEST), 'operation': 'copy; originals retained; no overwrite',
            'files': mapping, 'source_duplicate_groups': [v for v in duplicates.values() if len(v) > 1]})
        local_index = {'schema_version': 1, 'path_base': '.', 'original_paths_preserved': True,
                       'runs': {p: {k: f"{r['run_directory']}/{v}" for k, v in
                                   [('config', 'config.json'), ('summary', 'run_summary.json'),
                                    ('checkpoint', 'best_pr_auc_classifier_head.pt'),
                                    ('metrics', 'metrics.json'), ('predictions', 'validation_predictions.csv')]}
                                for p, r in source_runs.items()}}
        write_json(DEST / 'LOCAL_INDEX.json', local_index)
    provenance = read_json(DEST / 'import_manifest.json')
    for item in provenance['files']:
        require(sha(DEST / item['destination']) == item['sha256'], f"Copied file changed: {item['destination']}")
        require(sha(SOURCE / item['source']) == item['sha256'], f"Source file changed: {item['source']}")
    require(set(inventory(SOURCE)) == {item['source'] for item in provenance['files']}, 'Source file list changed')
    before = read_json(REPORT / 'existing_outputs_before.json')
    for relative, info in before.items():
        require(sha(ROOT / 'outputs' / relative) == info['sha256'], f'Previous output changed: {relative}')
    runs = verify_runs(DEST)
    index = read_json(DEST / 'LOCAL_INDEX.json')
    for entry in index['runs'].values():
        require(all((DEST / path).is_file() for path in entry.values()), 'Broken local index')
    baseline, _ = read_result(DEST / index['runs']['cls']['summary'])
    require(Path(baseline['checkpoint_path']).is_file(), 'Existing baseline loader failed')
    audit = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
             'original_files_preserved_and_verified': len(provenance['files']),
             'previous_output_files_unchanged': len(before), 'runs': runs,
             'existing_cls_baseline_loader': 'passed', 'test_split_opened': False,
             'full_encoder_inference': 'not run; remote base weights/tokenizer not in download',
             'data_identity_limit': 'Validation row identities/labels match locally. Historical train contents, '
                                    'criteria and base-model identity lack remote hashes; counts/paths are not hashes.'}
    write_json(DEST / 'verification.json', audit)
    write_json(REPORT / 'verification.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
