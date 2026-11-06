"""Generate a Russian report from the computed aggregate results."""
from pathlib import Path
import json
import pandas as pd


def table(frame, columns):
    """Render Markdown without an optional tabulate dependency."""
    def cell(value):
        if pd.isna(value):
            return '—'
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    subset = frame[list(columns)]
    lines = ['| ' + ' | '.join(columns.values()) + ' |',
             '| ' + ' | '.join('---' for _ in columns) + ' |']
    lines.extend('| ' + ' | '.join(cell(v) for v in row) + ' |'
                 for row in subset.itertuples(index=False, name=None))
    return '\n'.join(lines)


def build_report(root):
    root = Path(root)
    exports = root / 'exports'
    read = lambda name: pd.read_csv(exports / name)
    metrics = read('metrics_test.csv')
    model = metrics.loc[metrics.model == 'logistic_regression'].iloc[0]
    splits = read('split_summary.csv')
    validation = read('thresholds_validation.csv')
    thresholds = read('thresholds_test.csv')
    calibration = read('calibration_test.csv')
    groups = read('group_errors_test.csv')
    coefficients = read('coefficients.csv')
    policy = json.loads((exports / 'policy.json').read_text(encoding='utf-8'))
    metadata = json.loads((exports / 'dataset_metadata.json').read_text(encoding='utf-8'))
    threshold = policy['selected_threshold']
    if float(model.threshold) != float(threshold):
        raise ValueError('Report threshold differs from frozen policy')
    threshold_columns = {'threshold':'Порог','tp':'TP','fp':'FP','tn':'TN','fn':'FN',
                         'precision':'Precision','recall':'Recall','fpr':'FPR','review_rate':'Доля проверок'}
    top = coefficients.loc[coefficients.coefficient.abs().sort_values(ascending=False).index].head(12)
    report = f'''# Прогнозирование просрочки по кредитным картам

Отчёт сформирован из рассчитанных таблиц `exports/`. Числа относятся к одному
зафиксированному историческому разбиению и учебной модели.

## Источник и задача

[UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients),
I-Cheng Yeh (2009), [DOI 10.24432/C55S3H](https://doi.org/10.24432/C55S3H),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
В наборе 30 000 действующих клиентов Тайваня, 23 признака и идентификатор ID.
История охватывает апрель–сентябрь 2005 года. Цель `default payment next month`
переименована в `default`: 1 — дефолт по платежу в следующем месяце, 0 — его отсутствие
по определению набора. Это не заявки на выдачу кредита и не фактические кредитные убытки.
Суммы заданы в NT$, возраст — в годах. SHA-256 исходного файла:
`{metadata['sha256']}`. См. [метаданные](../exports/dataset_metadata.json)
и [получение данных](../data/README.md).

## Качество данных и EDA

[EDA-заметки](eda_notes.md) и таблицы `exports/eda_*` содержат проверки схемы,
типов, пропусков, ID, полных дубликатов, необычных кодов и числа строк до/после обработки.
Общая сводка набора — описательная; связи признаков с целью изучены только на train
после фиксации разбиения. Необычные категории и отрицательные суммы не удаляются
автоматически. Статусам −2 и 0 не приписывается не подтверждённый UCI смысл.

## Методика

Стратифицированный test выделен первым: 20% набора. Из остатка 25% отведены
под validation. Оба вызова `train_test_split` используют `random_state=42`.

{table(splits, {'split':'Выборка','n':'Клиенты','positives':'Дефолты','default_rate':'Доля дефолтов'})}

Принадлежность строк сохранена локально в `data/generated/splits.csv`.
ID и цель исключены из X. Вся обучаемая предобработка находится внутри
`ColumnTransformer` / `Pipeline`, обучаемого только на train. Числовые признаки:
медианная импутация и StandardScaler. Категориальные SEX, EDUCATION, MARRIAGE
и шесть PAY-статусов: импутация и one-hot с первой категорией в качестве опорной,
`handle_unknown="ignore"`. Так не предполагается равное расстояние между всеми кодами статусов.

Логистическая регрессия: `C=1`, `solver="lbfgs"`, `max_iter=3000`, без class weights,
подбора гиперпараметров и дополнительной калибровки. Baseline —
`DummyClassifier(strategy="prior")`: константная вероятность равна частоте дефолтов на train.
Дисбаланс учитывается выбором метрик и порога; accuracy не является главным критерием.
Положительный прогноз задаётся правилом `probability >= threshold`.
Даты признаков по документации UCI предшествуют целевому месяцу, однако индивидуальных
дат событий и новых периодов нет. Случайное разбиение не заменяет проверку на будущем.

## Выбор порога на validation

До test заданы кандидаты 0,2 / 0,3 / 0,5. Правило: наибольший recall при доле проверок
не выше 30%; при равенстве — больший precision, затем больший порог. Если допустимых
порогов нет, выбирается минимальная доля проверок, затем больший порог.
30% — условная операционная ёмкость учебного сценария.

{table(validation, threshold_columns)}

Выбранный порог **{threshold}** сохранён в `policy.json` до расчёта тестовых прогнозов.

## Финальные метрики test

Тест: **{int(model.n)}** клиентов, **{int(model.positive_support)}** с дефолтом,
**{int(model.negative_support)}** без дефолта.

{table(metrics, {'model':'Модель','threshold':'Порог','roc_auc':'ROC-AUC','average_precision':'Average precision','brier':'Brier ↓','precision':'Precision','recall':'Recall','f1':'F1','review_rate':'Доля проверок'})}

Average precision — средняя точность по precision–recall, не трапецеидальная PR-AUC.
Brier score — средний квадрат ошибки вероятности; меньше — лучше.
Baseline не ранжирует клиентов (ROC-AUC 0,5). При пороге 0,5 он не выдаёт
положительных прогнозов; precision в этом случае неопределён математически
и возвращается как 0 по соглашению `zero_division=0`.

![ROC и precision–recall](../images/roc_pr.png)

## Учебный сценарий дополнительной проверки

Положительный прогноз означает направление действующего клиента на дополнительную проверку.
TP — клиент с дефолтом на проверке; FP — клиент без дефолта, ошибочно направленный на неё;
FN — клиент с дефолтом, пропущенный моделью; TN — клиент без дефолта вне проверки.
Отказы в выдаче кредита в данных не наблюдаются.

{table(thresholds, threshold_columns)}

Знаменатели: recall = TP/(TP+FN), precision = TP/(TP+FP), FPR = FP/(FP+TN),
доля проверок = (TP+FP)/n. Сумма четырёх ячеек равна n.
Заранее заданные тестовые пороги показывают чувствительность сценария;
порог по test повторно не выбирается.

При выбранном пороге направлено **{model.review_rate:.2%}** клиентов:
**{int(model.tp)} TP**, **{int(model.fp)} FP**; пропущено **{int(model.fn)} FN**.
Recall **{model.recall:.3f}**, precision **{model.precision:.3f}**.
Это компромисс нагрузки и выявления метки, а не доказанный эффект вмешательства.
Денежный эффект не рассчитывается: нет фактических потерь, стоимости проверки
и оценки предотвращаемых дефолтов. LIMIT_BAL не приравнивается к убытку.

![Матрица ошибок](../images/confusion_matrix.png)

## Проверка вероятностей

Reliability diagram сопоставляет средний прогноз и наблюдаемую частоту дефолтов.
Ниже приведено наполнение интервалов; редкие интервалы дают нестабильную оценку.

{table(calibration, {'model':'Модель','bin_left':'От','bin_right':'До','n':'n','mean_probability':'Средний прогноз','observed_rate':'Частота дефолта'})}

![Калибровка и наполнение интервалов](../images/calibration.png)

Brier основной модели **{model.brier:.4f}**; сравнение с baseline приведено выше.
Этот показатель отражает качество вероятностей в целом, а не только калибровку.
Дополнительная калибровка не обучалась; улучшение после калибровочной процедуры не заявляется.
Историческая оценка не гарантирует корректность вероятностей на новых клиентах и периодах.

## Факторы и интерпретация

12 наибольших по абсолютному значению коэффициентов; все коэффициенты,
единицы сравнения и опорные категории — в [coefficients.csv](../exports/coefficients.csv).

{table(top, {'feature':'Признак','coefficient':'Коэффициент','odds_ratio':'Odds ratio','interpretation':'Единица / сравнение'})}

![Коэффициенты](../images/coefficients.png)

Коэффициент — изменение log-odds; odds ratio = exp(coefficient). Для числовых признаков
сравнение соответствует одной train SD, для категорий — переходу от опорной категории,
при прочих фиксированных признаках. Odds ratio не является вероятностью.
Соседние месячные признаки коррелируют, поэтому знаки и величины условных коэффициентов
могут быть нестабильны. Это ассоциации модели, а не причинное влияние.

## Ошибки по чувствительным группам

Только test, логистическая модель, порог {threshold}. Возрастные границы заданы в коде заранее.
SEX и MARRIAGE показаны исходными кодами. Размеры групп и число дефолтов приведены
для оценки устойчивости; метрики редких групп следует интерпретировать осторожно.

{table(groups, {'group':'Признак','category':'Группа','n':'n','positives':'Дефолты','tp':'TP','fp':'FP','tn':'TN','fn':'FN','precision':'Precision','recall':'Recall','fpr':'FPR','review_rate':'Доля проверок'})}

Чувствительные поля входят в учебную модель. Описательное сравнение ошибок не доказывает
справедливость, дискриминацию или юридическую допустимость. Причины различий,
качество разметки и переносимость на современных клиентов отдельно не установлены.

## Выводы и ограничения

Получено воспроизводимое ранжирование метки дефолта и явный расчёт нагрузки,
ложных направлений и пропущенных дефолтов в учебном сценарии. Порог выбран на validation,
итоговые метрики получены на test после фиксации решений. Повторный запуск проверяет
воспроизводимость, а не создаёт новый независимый тест и не служит подбором по test.
Для внедрения потребовались бы современные временные выборки, оценка устойчивости,
реальных потерь и эффекта проверки, правовая и этическая экспертиза.
Модель предназначена для обучения и портфолио и не должна применяться для реальных
решений о кредитовании.
'''
    destination = root / 'report' / 'report.md'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding='utf-8', newline='\n')
    return destination


if __name__ == '__main__':
    build_report(Path(__file__).resolve().parents[1])
