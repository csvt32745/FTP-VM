# %%
import pandas as pd
import seaborn as sns
import os
import glob
import tqdm
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--root', help='target root', default='', type=str)
parser.add_argument('--skip_smoke', help='skip smoke class in vm108', action='store_true')
parser.add_argument('--first_mem_only', help='Just agg model with first frame memory', action='store_true')
# parser.add_argument('--mem_freq_only', help='Just agg model with mem freq comparison', action='store_true')
args = parser.parse_args()

assert args.skip_smoke + args.first_mem_only <= 1

root = args.root
dataset = os.path.basename(root)

if skip_smoke := ((args.skip_smoke) and ('vm108' in dataset)):
    print("skip smokes")
if first_only := args.first_mem_only:
    print("First given memory only")

# root = '/home/csvt32745/matte/MaskPropagation/vm240k_val_midtri_512x288'
models = sorted(os.listdir(root))
excel_root = './'
db_root = './exp_db'
if skip_smoke:
    suffix = '_skipsmoke'
elif first_only:
    suffix = '_first'
else:
    suffix = ''
db_path = os.path.join(db_root, dataset+suffix+'.json')

if os.path.isfile(db_path):
    print("Load db: ", db_path)
    with open(db_path) as f:
        db = json.load(f)
    total_metrics_clip = db['avg_clip']
    total_metrics_frame = db['avg_frame']
    if len(total_metrics_clip) > 0:
        metrics = set(total_metrics_clip[next(iter(total_metrics_clip))].keys())
    else:
        metrics = set()
    print(f"{len(total_metrics_clip)} models, {len(metrics)} metrics")
    
    exist_models = set(total_metrics_clip.keys())
    # print(exist_models)
    last_time = os.path.getmtime(db_path)
else:
    print("No db exists")
    total_metrics_clip = {}
    total_metrics_frame = {}
    metrics = set()
    exist_models = set()
    last_time = -1

# %%
print("Load model metrics...")
is_updated = False

model_paths = []
for m in models:
    if m == 'GT':
        continue
    if not os.path.isfile(p:=os.path.join(root, m, m+'.xlsx')):
        print("Metric file not found: " + m)
        continue
    if m in exist_models and os.path.getmtime(p) < last_time:
        print("Models exists: " + m)
        continue

    if first_only and ('_mem' in  m) and (os.path.splitext(m)[0][-1] == 'f'):
        print('First only, skip ', m)
        continue
    model_paths.append((m, p))

for m, p in tqdm.tqdm(model_paths):
    print('Read excel: ', m)
    dfs = pd.read_excel(p, sheet_name=None)
    metric_clip = {}
    metric_frame = {}
    keys = set(dfs.keys())
    keys.remove('summary')

    for k in keys:
        df = dfs[k].iloc[1:]
        if skip_smoke:
            df = df.loc[~df[df.columns[1]].str.contains('smokes_')]
        df = df[df.columns[3:]]

        metric_clip[k] = df.mean(axis=1).mean()
        metric_frame[k] = df.mean(axis=0).mean()
    total_metrics_clip[m] = metric_clip
    total_metrics_frame[m] = metric_frame
    metrics = metrics | keys
    
    is_updated = True

# %%
# if not is_updated:
#     print("No file updated, exit")
#     exit()

# %%
sort_dict = lambda d: d
# sort_dict = lambda d: dict(sorted(d.items()))
print("Save db: ", db_path)
print(f"{len(total_metrics_clip)} models, {len(metrics)} metrics")
total_dict = {'avg_clip': sort_dict(total_metrics_clip), 'avg_frame': sort_dict(total_metrics_frame)}
with open(db_path, 'w') as f:
    json.dump(total_dict, f, indent='\t')

# %%
excel_path = os.path.join(excel_root, dataset+suffix+".xlsx")
print("Save excel: ", excel_path)
with pd.ExcelWriter(excel_path) as writer:  
    for k, df in total_dict.items():
        df = pd.DataFrame(df).sort_index(ascending=False)
        df.to_excel(writer, sheet_name=k, float_format='%0.4f')
        workbook = writer.book
        worksheet = writer.sheets[k]
        # num_format = workbook.add_format({
        #     'num_format': '0.0000'
        # })
        # worksheet.set_row(1, 30, cell_format=num_format)

        header_format = workbook.add_format({
            'text_wrap': True,
            'bold': True,
            'align' : 'center',
            'valign' : 'vcenter',
        })
        worksheet.set_column(0, 1000, width=20)
        for col_num, value in enumerate(df.columns.values):
            print(value)
            worksheet.write(0, col_num+1, value, header_format)


