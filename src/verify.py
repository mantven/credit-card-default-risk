"""Compare clean-run outputs and validate images: python -m src.verify OTHER_ROOT."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from .data import write_json


def compare(root, other):
    checks = []
    reference = json.loads((root / 'exports/manifest.json').read_text(encoding='utf-8'))
    reproduced = json.loads((other / 'exports/manifest.json').read_text(encoding='utf-8'))
    if set(reference) != set(reproduced):
        raise AssertionError(f'Output sets differ: {set(reference) ^ set(reproduced)}')
    for relative, checksum in reference.items():
        a, b = root / relative, other / relative
        if a.suffix == '.csv':
            pd.testing.assert_frame_equal(pd.read_csv(a), pd.read_csv(b), check_exact=False, atol=1e-12, rtol=1e-10)
        elif a.suffix == '.png':
            with Image.open(a) as image:
                image.verify()
            with Image.open(a) as ia, Image.open(b) as ib:
                assert ia.size == ib.size, relative
                np.testing.assert_array_equal(np.asarray(ia), np.asarray(ib), err_msg=relative)
        else:
            assert checksum == reproduced[relative], f'File hash mismatch: {relative}'
        checks.append(relative)
    result = {'status': 'passed', 'compared_artifacts': len(checks),
              'csv_atol': 1e-12, 'csv_rtol': 1e-10, 'images': 'decoded pixel equality and PNG integrity',
              'other_files': 'SHA-256 equality', 'files': checks}
    write_json(root / 'exports/reproduction_check.json', result)
    print(f'Reproduction verified: {len(checks)} artifacts.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('other_root', type=Path)
    args = parser.parse_args()
    compare(Path(__file__).resolve().parents[1], args.other_root.resolve())


if __name__ == '__main__':
    main()
