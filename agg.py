# %%
import pandas as pd
import seaborn as sns
import os
import glob
import tqdm
import json

# %%
import sys
root = sys.argv[1]
# root = '/home/csvt32745/matte/MaskPropagation/vm240k_val_midtri_512x288'
db_root = './exp_db'
excel_root = './'
dataset = os.path.basename(root)
models = sorted(os.listdir(root))

# %%
db_path = os.path.join(db_root, dataset+'.json')
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
for m in tqdm.tqdm(models):
    if m == 'GT':
        continue
    if not os.path.isfile(p:=os.path.join(root, m, m+'.xlsx')):
        print("Metric file not found: " + m)
        continue
    if m in exist_models and os.path.getmtime(p) < last_time:
        print("Models exists: " + m)
        continue

    dfs = pd.read_excel(p, sheet_name=None)
    metric_clip = {}
    metric_frame = {}
    keys = set(dfs.keys())
    keys.remove('summary')

    for k in keys:
        df = dfs[k].iloc[1:]
        df = df[df.columns[3:]]

        metric_clip[k] = df.mean(axis=1).mean()
        metric_frame[k] = df.mean(axis=0).mean()
    total_metrics_clip[m] = metric_clip
    total_metrics_frame[m] = metric_frame
    metrics = metrics | keys
    
    is_updated = True

# %%
if not is_updated:
    print("No file updated, exit")
    exit()

# %%
print("Save db: ", db_path)
print(f"{len(total_metrics_clip)} models, {len(metrics)} metrics")
total_dict = {'avg_clip': total_metrics_clip, 'avg_frame': total_metrics_frame}
with open(db_path, 'w') as f:
    json.dump(total_dict, f, indent='\t')

# %%
excel_path = os.path.join(excel_root, dataset+".xlsx")
print("Save excel: ", excel_path)
with pd.ExcelWriter(excel_path) as writer:  
    for k, df in total_dict.items():
        pd.DataFrame(df).sort_index(ascending=False).to_excel(writer, sheet_name=k)


