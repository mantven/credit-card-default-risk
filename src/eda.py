"""Quality checks on full data; associations and plots on frozen train only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data import CATEGORICAL, STATUS, NUMERIC, FEATURES


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _groups(train, groups, feature):
    t = train.groupby(groups, observed=True, dropna=False)['default'].agg(n='size', defaults='sum', default_rate='mean').reset_index()
    t = t.rename(columns={t.columns[0]: 'group'})
    t.insert(0, 'feature', feature)
    n, p, z = t.n, t.default_rate, 1.959963984540054
    center = (p + z*z/(2*n))/(1+z*z/n)
    width = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    t['ci95_low'], t['ci95_high'] = center-width, center+width
    t['rare_n_lt_100'] = n < 100
    return t


def run_eda(df, train, root):
    root = Path(root)
    ex, im = root/'exports', root/'images'
    for p in (ex, im, root/'report'):
        p.mkdir(parents=True, exist_ok=True)
    quality = {
        'scope': 'full dataset, descriptive only',
        'rows_before': len(df), 'rows_after': len(df), 'rows_removed': 0,
        'feature_count': len(FEATURES), 'missing_cells': int(df.isna().sum().sum()),
        'duplicate_ids': int(df.ID.duplicated().sum()),
        'full_duplicates': int(df.duplicated().sum()),
        'duplicate_feature_vectors': int(df[FEATURES].duplicated().sum()),
        'duplicates_without_id': int(df.drop(columns='ID').duplicated().sum()),
        'target_counts': {str(k): int(v) for k,v in df['default'].value_counts().sort_index().items()},
        'default_rate': float(df['default'].mean()),
        'decision': 'No deletion or recoding; equal feature vectors do not establish client identity.'
    }
    (ex/'eda_quality.json').write_text(json.dumps(quality, indent=2, ensure_ascii=False)+'\n', encoding='utf-8', newline='\n')
    pd.DataFrame([{'column': c, 'dtype': str(df[c].dtype), 'missing_n': int(df[c].isna().sum()), 'unique_n': df[c].nunique(), 'minimum': df[c].min(), 'maximum': df[c].max()} for c in df]).to_csv(ex/'eda_schema.csv', index=False, lineterminator='\n')
    documented = {'SEX': {1,2}, 'EDUCATION': {1,2,3,4}, 'MARRIAGE': {1,2,3}}
    documented.update({c: {-1, *range(1,10)} for c in STATUS})
    categories = pd.DataFrame([{'feature': c, 'code': k, 'n': int(n), 'documented_on_uci_page': k in documented[c], 'scope': 'full descriptive'} for c in CATEGORICAL for k,n in df[c].value_counts().sort_index().items()])
    categories.to_csv(ex/'eda_category_audit.csv', index=False, lineterminator='\n')
    amounts = ['LIMIT_BAL'] + [f'{prefix}{i}' for prefix in ['BILL_AMT','PAY_AMT'] for i in range(1,7)]
    signs = pd.DataFrame([{'feature': c, 'negative_n': int((df[c]<0).sum()), 'zero_n': int((df[c]==0).sum()), 'positive_n': int((df[c]>0).sum()), 'scope': 'full descriptive'} for c in amounts])
    signs.to_csv(ex/'eda_amount_signs.csv', index=False, lineterminator='\n')
    train[NUMERIC].describe(percentiles=[.01,.25,.5,.75,.99]).T.to_csv(ex/'eda_train_numeric_summary.csv', index_label='feature', lineterminator='\n')
    association = pd.concat([_groups(train,c,c) for c in CATEGORICAL], ignore_index=True)
    association.to_csv(ex/'eda_train_category_associations.csv', index=False, lineterminator='\n')
    buckets = pd.cut(train.PAY_AMT1, [-np.inf,-.001,0,1000,5000,10000,50000,np.inf], labels=['negative','zero','(0,1k]','(1k,5k]','(5k,10k]','(10k,50k]','>50k'])
    payments = _groups(train,buckets.rename('payment_bucket'),'PAY_AMT1_fixed_ntd_buckets')
    payments.to_csv(ex/'eda_train_payment_associations.csv', index=False, lineterminator='\n')
    corr = train[NUMERIC].corr()
    corr.to_csv(ex/'eda_train_correlations.csv', index_label='feature', lineterminator='\n')
    adjacent = pd.DataFrame([{'family': prefix, 'feature_a': f'{prefix}{i}', 'feature_b': f'{prefix}{i+1}', 'pearson_r': train[f'{prefix}{i}'].corr(train[f'{prefix}{i+1}']), 'spearman_rho': train[f'{prefix}{i}'].corr(train[f'{prefix}{i+1}'],method='spearman'), 'n': len(train)} for prefix in ['BILL_AMT','PAY_AMT'] for i in range(1,6)])
    adjacent.to_csv(ex/'eda_train_adjacent_correlations.csv', index=False, lineterminator='\n')
    _plots(train, association, payments, corr, im)
    _notes(df, train, quality, categories, signs, adjacent, root)
    return quality


def _association_plot(table, title, xlabel, path):
    fig, (ax, counts) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]}, constrained_layout=True)
    x = np.arange(len(table))
    p = table.default_rate.to_numpy()
    colors = np.where(table.rare_n_lt_100, '#db8b39', '#31738e')
    ax.bar(x, p, color=colors)
    ax.errorbar(x, p, yerr=np.array([p-table.ci95_low, table.ci95_high-p]), fmt='none', color='#263746', capsize=3)
    ax.set(title=title+' — train only', ylabel='Observed default rate', ylim=(0,min(1.03,float(table.ci95_high.max())+.1)))
    ax.grid(axis='y',alpha=.2)
    counts.bar(x,table.n,color=colors)
    counts.set(ylabel='Clients (n)',xlabel=xlabel)
    counts.set_xticks(x,[str(v) for v in table.group])
    counts.tick_params(axis='x',rotation=25)
    fig.text(.01,-.025,'95% Wilson intervals; orange: n < 100. Associations, not causal effects.',fontsize=9)
    _save(fig,path)


def _plots(train, association, payments, corr, im):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    classes = train['default'].value_counts().sort_index()
    pd.DataFrame({'target':classes.index,'n':classes.values,'fraction':classes.values/len(train)}).to_csv(im.parent/'exports/eda_train_classes.csv',index=False,lineterminator='\n')
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    axs[0].bar([str(c) for c in classes.index],classes.values,color=['#31738e','#db8b39'])
    axs[0].set(xlabel='Default next month (0 / 1)',ylabel='Clients (n)',title='Target — train',ylim=(0,classes.max()*1.15))
    for x,n in enumerate(classes):
        axs[0].text(x,n,f'{n:,}\n{n/len(train):.1%}',ha='center',va='bottom')
    axs[1].hist(train.LIMIT_BAL,bins=40,color='#31738e')
    axs[1].set(xlabel='Credit limit (NT dollars)',ylabel='Clients (n)',title='Credit limit — train')
    axs[1].ticklabel_format(axis='x',style='plain')
    axs[1].tick_params(axis='x',rotation=25)
    axs[2].hist(train.AGE,bins=np.arange(20,83,2),color='#31738e')
    axs[2].set(xlabel='Age (years)',ylabel='Clients (n)',title='Age — train')
    _save(fig,im/'eda_population.png')
    status=pd.DataFrame({c:train[c].value_counts() for c in STATUS}).fillna(0).sort_index().astype(int)
    status.rename_axis('status_code').to_csv(im.parent/'exports/eda_train_status_counts.csv',lineterminator='\n')
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    status.set_axis(['Sep','Aug','Jul','Jun','May','Apr'],axis=1).plot.bar(ax=ax,width=.85,colormap='viridis')
    ax.set(xlabel='Raw repayment status code (categorical)',ylabel='Clients (n)',title='Monthly repayment status, Apr–Sep 2005 — train only')
    ax.tick_params(axis='x',rotation=0)
    ax.legend(title='Month',ncols=3)
    _save(fig,im/'eda_status_distributions.png')
    fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,prefix,title in zip(axs,['BILL_AMT','PAY_AMT'],['Bill statements','Previous payments']):
        for i,month in enumerate(['Sep','Aug','Jul','Jun','May','Apr'],1):
            values=np.sort(train[f'{prefix}{i}'].to_numpy())
            ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=month,linewidth=1.5)
        ax.set_xscale('symlog',linthresh=1000)
        if prefix == 'BILL_AMT':
            ax.set_xticks([-100000,-10000,-1000,0,1000,10000,100000,1000000],
                          ['−100k','−10k','−1k','0','1k','10k','100k','1M'])
        else:
            ax.set_xticks([0,1000,10000,100000,1000000], ['0','1k','10k','100k','1M'])
        ax.set(xlabel='Amount (NT dollars; symmetric log scale)',ylabel='Cumulative share of clients',title=title+' — train only',ylim=(0,1.02))
        ax.grid(alpha=.2)
        ax.legend(ncols=2)
    _save(fig,im/'eda_amount_distributions.png')
    _association_plot(association.loc[association.feature=='PAY_0'],'September repayment status and default','Raw PAY_0 code',im/'eda_recent_status_default.png')
    _association_plot(payments,'September payment amount and default','Previous payment (NT dollars; fixed buckets)',im/'eda_recent_payment_default.png')
    fig,ax=plt.subplots(figsize=(10,9),constrained_layout=True)
    mesh=ax.imshow(corr,cmap='RdBu_r',vmin=-1,vmax=1)
    ax.set_xticks(np.arange(len(corr)),corr.columns,rotation=65,ha='right')
    ax.set_yticks(np.arange(len(corr)),corr.index)
    ax.set_title('Continuous-feature Pearson correlation — train only')
    fig.colorbar(mesh,ax=ax,label='Pearson r',shrink=.8)
    _save(fig,im/'eda_correlations.png')


def _notes(df, train, quality, categories, signs, adjacent, root):
    undocumented=categories.loc[~categories.documented_on_uci_page]
    codes='; '.join(f"{c}: {', '.join(str(int(v)) for v in g.code)}" for c,g in undocumented.groupby('feature',sort=False))
    billneg=int(signs.loc[signs.feature.str.startswith('BILL_AMT'),'negative_n'].sum())
    payneg=int(signs.loc[signs.feature.str.startswith('PAY_AMT'),'negative_n'].sum())
    r=adjacent.loc[adjacent.family=='BILL_AMT','pearson_r']
    assoc = pd.read_csv(root/'exports/eda_train_category_associations.csv')
    recent = assoc.loc[assoc.feature == 'PAY_0'].set_index('group')
    pay = pd.read_csv(root/'exports/eda_train_payment_associations.csv').set_index('group')
    note=f'''# Наблюдения EDA

Схема, качество и состав классов проверены на всём наборе **только описательно**. Все графики, связи с целевой переменной и корреляции рассчитаны только на train ({len(train):,} записей), после фиксации разбиения. Эти результаты не используются для настройки по test.

## Проверка данных

- Строк до / после: {len(df):,} / {len(df):,}; ничего не удалено, категории не объединены.
- Признаков без ID и цели: {quality['feature_count']}; пропущенных ячеек: {quality['missing_cells']}.
- Повторов ID: {quality['duplicate_ids']}; полных дублей: {quality['full_duplicates']}; повторных векторов признаков без ID и цели: {quality['duplicate_feature_vectors']}. Число повторов считается после первого вхождения, это не число групп. Одинаковые признаки не доказывают, что записи принадлежат одному клиенту; они сохранены. Случайное разбиение по строкам не исключает совпадения векторов между выборками.
- Доля метки 1 по всему набору (описательно): {quality['default_rate']:.2%}; на train: {train['default'].mean():.2%}.
- Наблюдаемые коды, не объяснённые таблицей значений на странице UCI: {codes}. В частности, статусы −2 и 0 нельзя автоматически называть «без просрочки» или «не использовал карту» либо удалять. Модель кодирует статусы как категории.
- Отрицательных ячеек счетов за шесть месяцев: {billneg}; отрицательных ячеек платежей: {payneg}. Это число ячеек, а не уникальных клиентов. Знак сохранён без неподтверждённой экономической трактовки. Детали по столбцам — в `exports/eda_amount_signs.csv`.

## Связи на train

В `eda_train_category_associations.csv` и `eda_train_payment_associations.csv` для каждой группы приведены n, число дефолтов, частота, 95% интервал Уилсона и индикатор редкой группы (n < 100). Интервалы описывают неопределённость при предположении независимых наблюдений; это не доказательство причинности и не поправка на множественные сравнения.

Недавний статус PAY_0 и размер PAY_AMT1 показаны отдельно. Платежи разбиты на фиксированные диапазоны в NT dollars; нулевые выделены в свою группу. Сравнения описывают ассоциации и зависят от состава клиентов.

При PAY_0=0 доля дефолтов на train составляет {recent.loc[0, 'default_rate']:.2%} (n={int(recent.loc[0, 'n']):,}), при PAY_0=2 — {recent.loc[2, 'default_rate']:.2%} (n={int(recent.loc[2, 'n']):,}). Для нулевого PAY_AMT1 — {pay.loc['zero', 'default_rate']:.2%} (n={int(pay.loc['zero', 'n']):,}). Смысл кода 0 не доопределяется; это сравнение наблюдаемых групп. Для PAY_0=4..8 размеры групп менее 100, поэтому хвосты нельзя считать устойчивой монотонной зависимостью.

![Классы, лимит и возраст](../images/eda_population.png)

![Распределения статусов](../images/eda_status_distributions.png)

![Счета и платежи](../images/eda_amount_distributions.png)

![Связь недавнего статуса с меткой](../images/eda_recent_status_default.png)

![Связь платежей с меткой](../images/eda_recent_payment_default.png)

Корреляция Пирсона соседних месячных счетов на train: {r.min():.3f}–{r.max():.3f}. Детали и ранговые корреляции — в `eda_train_adjacent_correlations.csv`. Коррелирующие признаки затрудняют отдельную интерпретацию коэффициентов: их знаки и величины могут быть нестабильны. Для необъяснённых кодов статусов числовое расстояние не интерпретируется.

![Корреляции числовых признаков](../images/eda_correlations.png)

Данные относятся к действующим клиентам Тайваня и истории апреля–сентября 2005 года. Проверки на будущих периодах и новых заявителях нет. Источник: [UCI, I-Cheng Yeh, CC BY 4.0](https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients).
'''
    (root/'report/eda_notes.md').write_text(note,encoding='utf-8',newline='\n')
