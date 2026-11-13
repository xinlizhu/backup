"""Resumable three-head OPRD experiment: shared S2 cache, train, then evaluate."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
OUTPUT = PROJECT / 'outputs/head_ablation_oprd_20261109'
HEADS = ['mmret_add', 'clip4cir_combiner', 'magiclens']
CACHE = PROJECT / 'outputs/on_policy_centered/cache/bge_mllm_s2_cirr_train'


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def status(phase, **kwargs):
    payload = dict(phase=phase, pid=os.getpid(), updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kwargs)
    write(OUTPUT / 'pipeline_status.json', payload)
    print(json.dumps(payload), flush=True)


def run(script, args, log, env=None):
    command = [sys.executable, '-u', str(PROJECT / 'scripts' / script), *map(str, args)]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as handle:
        handle.write('\n' + json.dumps({'command': command, 'started': time.time()}) + '\n')
        handle.flush()
        subprocess.run(command, cwd=PROJECT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main():
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / '.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert set(json.loads((OUTPUT / 'architecture_equivalence.json').read_text())) == set(HEADS)
        smoke = json.loads((OUTPUT / 'gpu_smoke.json').read_text())
        assert set(smoke) == set(HEADS)
        assert all(v['missing_gradient_count'] == v['nonfinite_gradient_count'] == 0 for v in smoke.values())
        env = os.environ.copy()
        env.update(PYTHONPATH=str(PROJECT / 'src'), OMP_NUM_THREADS='8',
                   HF_HOME=str(WORKSPACE / 'models/hf_cache'),
                   HF_MODULES_CACHE=str(WORKSPACE / 'models/hf_modules'),
                   HF_HUB_DISABLE_PROGRESS_BARS='1', TOKENIZERS_PARALLELISM='false')
        source_files = list((PROJECT / 'src/cir_qdiag_train').glob('*.py'))
        source_files += [PROJECT / 'scripts' / f for f in ['run_head_ablation_oprd.py',
            'train_on_policy_distillation.py', 'evaluate_external_qdiag.py', 'evaluate_external_retrieval.py',
            'oprd_head_adapter.py', 'prepare_s2_head_ablation_cache.py', 'validate_official_heads.py',
            'export_magiclens_head_oracle.py', 'collect_head_ablation_results.py']]
        source_files += [PROJECT / f'configs/head_{h}_oprd_s2_e4.yaml' for h in HEADS]
        hashes = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
        manifest = OUTPUT / 'source_manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != hashes:
            raise RuntimeError('Source changed since pipeline launch; audit before resuming')
        write(manifest, hashes)
        for p in source_files:
            dest = OUTPUT / 'source_snapshot' / p.relative_to(PROJECT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(p, dest)
        status('waiting_for_shared_teacher_cache', cache=str(CACHE))
        CACHE.mkdir(parents=True, exist_ok=True)
        # The already-running builder holds this lock. Resume/verify after it exits.
        with (CACHE / '.build.lock').open('a') as teacher_lock:
            fcntl.flock(teacher_lock, fcntl.LOCK_EX)
            fcntl.flock(teacher_lock, fcntl.LOCK_UN)
        teacher_env = {**env, 'PYTHONPATH': str(WORKSPACE / 'models/oprd_teacher_compat') + ':' + env['PYTHONPATH']}
        run('prepare_s2_head_ablation_cache.py', ['--config', PROJECT / 'configs/head_mmret_add_oprd_s2_e4.yaml',
            '--teacher-model', WORKSPACE / 'models/BGE-VL-MLLM-S2', '--batch-size', 16],
            OUTPUT / 'teacher_cache_verify.log', teacher_env)
        metadata = json.loads((CACHE / 'metadata.json').read_text())
        assert metadata['complete'] and metadata['query_count'] == 28225 and metadata['gallery_count'] == 16939
        write(OUTPUT / 'teacher_cache_verified.json', metadata)
        for head in HEADS:
            folder = OUTPUT / head
            config = PROJECT / f'configs/head_{head}_oprd_s2_e4.yaml'
            final = folder / 'checkpoint_epoch_004.pt'
            if not final.exists():
                status('training', head=head, config=str(config), log=str(folder / 'train.log'))
                args = ['--config', config]
                if (folder / 'last.pt').exists():
                    args += ['--resume-from', folder / 'last.pt']
                run('train_on_policy_distillation.py', args, folder / 'train.log', env)
            for label, checkpoint in [('initial', folder / 'initial.pt'), ('final', final)]:
                status('evaluating_qdiag', head=head, checkpoint=label)
                run('evaluate_external_qdiag.py', ['--model', 'oprd_head', '--checkpoint', checkpoint,
                    '--output', folder / f'qdiag_{label}', '--image-batch-size', 512, '--query-batch-size', 512],
                    folder / f'qdiag_{label}.log', env)
            status('evaluating_retrieval', head=head)
            run('evaluate_external_retrieval.py', ['--model', 'oprd_head', '--model-name', f'CLIP-L-OPRD-{head}',
                '--checkpoint', final, '--output', folder / 'retrieval', '--image-batch-size', 512,
                '--query-batch-size', 512], folder / 'retrieval.log', env)
            run('collect_head_ablation_results.py', [], OUTPUT / 'collect.log', env)
        run('collect_head_ablation_results.py', ['--require-complete'], OUTPUT / 'collect.log', env)
        status('complete', result=str(OUTPUT / 'comparison.csv'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=str(exc), traceback=traceback.format_exc())
        raise
