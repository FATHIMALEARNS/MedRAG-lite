import os
from ml.predict import predict

val_abn = 'c:/Projects/MedRAG-lite/ml/data/val/abnormal'
val_nor = 'c:/Projects/MedRAG-lite/ml/data/val/normal'

print('--- ABNORMAL ---')
for f in os.listdir(val_abn)[:5]:
    if f.endswith('.png'):
        res = predict(os.path.join(val_abn, f))
        lbl = res['label']
        n_p = res['normal_probability']
        a_p = res['abnormal_probability']
        print(f'{f}: {lbl} (N:{n_p:.3f}, A:{a_p:.3f})')

print('--- NORMAL ---')
for f in os.listdir(val_nor)[:5]:
    if f.endswith('.png'):
        res = predict(os.path.join(val_nor, f))
        lbl = res['label']
        n_p = res['normal_probability']
        a_p = res['abnormal_probability']
        print(f'{f}: {lbl} (N:{n_p:.3f}, A:{a_p:.3f})')
