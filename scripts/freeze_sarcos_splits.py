"""Generate SARCOS A3 indices only when explicitly requested with --write."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data/raw/sarcos_inv.mat'
z=loadmat(source)['sarcos_inv']
assert z.shape==(44484,28),z.shape
n=len(z); p=int(.6*n);q=int(.8*n); gap=25
blocked={'train':np.arange(0,p-gap),'val':np.arange(p+gap,q-gap),'test':np.arange(q+gap,n)}
perm=np.random.default_rng(42).permutation(n)
random={'train':perm[:p],'val':perm[p:q],'test':perm[q:]}
for name,s in [('blocked',blocked),('random',random)]:
 ids=np.concatenate(list(s.values()))
 assert len(ids)==len(np.unique(ids))
 assert np.all((ids>=0)&(ids<n))
 if name=='random': assert len(ids)==n
indices_path=ROOT/'data/splits/sarcos_split_indices.npz'
manifest_path=ROOT/'data/splits/sarcos_split_manifest.json'
summary={'data_file_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'n_rows':n,'target_column_zero_based':21,'feature_columns_zero_based':'0..20','blocked_cutpoints':[p,q],'blocked_purge_each_side':gap,'random_seed':42,'splits':{method:{part:{'count':len(ind),'first_row_0based':int(ind[0]),'last_row_0based':int(ind[-1])} for part,ind in s.items()} for method,s in [('blocked',blocked),('random',random)]}}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--write',action='store_true',help='write generated indices and manifest')
parser.add_argument('--overwrite',action='store_true',help='allow replacing existing frozen artifacts')
args=parser.parse_args()
if args.write:
 if (indices_path.exists() or manifest_path.exists()) and not args.overwrite:
  raise SystemExit('Frozen artifacts already exist; inspect them and pass --overwrite only if replacement is intended.')
 np.savez_compressed(indices_path,**{f'{method}_{part}':ind for method,split in [('blocked',blocked),('random',random)] for part,ind in split.items()})
 manifest_path.write_text(json.dumps(summary,indent=2))
else:
 print('Dry run only. Existing frozen split files were not modified. Pass --write to save.')
print(json.dumps(summary,indent=2))
