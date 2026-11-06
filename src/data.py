"""Official UCI XLS retrieval, strict schema checks and immutable row splits."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen, Request
from zipfile import ZipFile
import io

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
SOURCE = 'https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients'
DOWNLOAD = 'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip'
EXPECTED_SHA256 = '30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933'
STATUS = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
CATEGORICAL = ['SEX', 'EDUCATION', 'MARRIAGE'] + STATUS
NUMERIC = ['LIMIT_BAL', 'AGE'] + [f'BILL_AMT{i}' for i in range(1, 7)] + [f'PAY_AMT{i}' for i in range(1, 7)]
FEATURES = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE'] + STATUS + [f'BILL_AMT{i}' for i in range(1, 7)] + [f'PAY_AMT{i}' for i in range(1, 7)]
TARGET_ORIGINAL = 'default payment next month'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def load_data(root):
    raw = root / 'data/raw/default_of_credit_card_clients.xls'
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        request = Request(DOWNLOAD, headers={'User-Agent': 'credit-card-default-risk/1.0'})
        try:
            with urlopen(request, timeout=90) as response:
                payload = response.read()
            with ZipFile(io.BytesIO(payload)) as archive:
                names = [n for n in archive.namelist() if n.lower().endswith('.xls')]
                if len(names) != 1:
                    raise ValueError(f'Expected one XLS, found {names}')
                raw.write_bytes(archive.read(names[0]))
        except Exception as exc:
            raise RuntimeError(f'Cannot obtain official UCI file. Retry with network or put the official XLS at {raw}. Source: {DOWNLOAD}') from exc
    checksum = hashlib.sha256(raw.read_bytes()).hexdigest()
    if checksum != EXPECTED_SHA256:
        raise ValueError(f'Official XLS checksum changed: {checksum}. Investigate source changes before updating EXPECTED_SHA256.')
    df = pd.read_excel(raw, header=1, engine='xlrd')
    df.columns = df.columns.astype(str).str.strip()
    expected = ['ID'] + FEATURES + [TARGET_ORIGINAL]
    if list(df.columns) != expected or df.shape != (30000, 25):
        raise ValueError(f'Unexpected UCI schema: {df.shape}, {df.columns.tolist()}')
    df = df.rename(columns={TARGET_ORIGINAL: 'default'})
    if set(df['default'].unique()) != {0, 1} or df.ID.duplicated().any():
        raise ValueError('Invalid target or duplicate IDs; investigate before splitting.')
    write_json(root / 'exports/dataset_metadata.json', {
        'source_url': SOURCE, 'download_url': DOWNLOAD,
        'doi': 'https://doi.org/10.24432/C55S3H', 'creator': 'I-Cheng Yeh',
        'license': 'CC BY 4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/',
        'sha256': checksum,
        'rows': len(df), 'features': len(FEATURES), 'original_target': TARGET_ORIGINAL,
        'original_columns': expected, 'header_zero_based': 1,
    })
    return df


def make_splits(df, root):
    positions = np.arange(len(df))
    development, test = train_test_split(positions, test_size=.2, stratify=df['default'], random_state=SEED)
    train, validation = train_test_split(development, test_size=.25, stratify=df.iloc[development]['default'], random_state=SEED)
    splits = {'train': train, 'validation': validation, 'test': test}
    assert not (set(train) & set(test) or set(train) & set(validation) or set(validation) & set(test))
    assert len(set(np.concatenate(list(splits.values())))) == len(df)
    membership = pd.DataFrame({'row_position': positions, 'split': ''})
    summary = []
    for name, rows in splits.items():
        membership.loc[rows, 'split'] = name
        y = df.iloc[rows]['default']
        summary.append({'split': name, 'n': len(y), 'positives': int(y.sum()), 'default_rate': float(y.mean())})
    membership.to_csv(root / 'data/generated/splits.csv', index=False, lineterminator='\n')
    pd.DataFrame(summary).to_csv(root / 'exports/split_summary.csv', index=False, lineterminator='\n')
    return splits
