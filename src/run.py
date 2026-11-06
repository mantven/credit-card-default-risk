"""Run from the repository root: python -m src.run."""
import argparse
import hashlib
import importlib.metadata
import platform
from pathlib import Path
import warnings
from sklearn.exceptions import ConvergenceWarning
from .data import load_data, make_splits, write_json
from .modeling import evaluate
from .plots import make_plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-only', action='store_true', help='Download and validate official XLS only')
    parser.add_argument('--output-root', type=Path, help='Separate output root for reproduction audit (contains its own raw cache)')
    args = parser.parse_args()
    root = args.output_root.resolve() if args.output_root else Path(__file__).resolve().parents[1]
    for directory in ['data/raw', 'data/generated', 'exports', 'images', 'report']:
        (root / directory).mkdir(parents=True, exist_ok=True)
    df = load_data(root)
    print(f'Official data validated: {df.shape[0]} rows, {df.shape[1]-2} features', flush=True)
    if args.download_only:
        return
    splits = make_splits(df, root)
    from .eda import run_eda
    run_eda(df, df.iloc[splits['train']].copy(), root)
    print('EDA complete on train; whole-data quality audit is descriptive.', flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        results = evaluate(df, splits, root)
    make_plots(results, root)
    packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'xlrd',
                'joblib', 'threadpoolctl', 'contourpy', 'cycler', 'fonttools', 'kiwisolver',
                'packaging', 'pillow', 'pyparsing', 'python-dateutil', 'pytz', 'tzdata', 'cloudpickle']
    write_json(root / 'exports/environment.json', {'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in packages}})
    from .reporting import build_report
    build_report(root)
    from .readme import build_readme
    build_readme(root)
    manifest = {}
    for folder in ['exports', 'images', 'report', 'data/generated']:
        for path in sorted((root / folder).glob('*')):
            if path.is_file() and path.name not in ['manifest.json', 'review.md', 'reproduction_check.json']:
                manifest[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['README.md'] = hashlib.sha256((root / 'README.md').read_bytes()).hexdigest()
    write_json(root / 'exports/manifest.json', manifest)
    print(results['metrics'].to_string(index=False), flush=True)
    print(f'Complete. Results: {root}', flush=True)


if __name__ == '__main__':
    main()
