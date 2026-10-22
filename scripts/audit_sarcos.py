import numpy as np, pandas as pd, hashlib, json
from pathlib import Path
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
ROOT = Path(__file__).resolve().parents[1]
paths={'train':ROOT/'data/raw/sarcos_inv.mat','test':ROOT/'data/raw/sarcos_inv_test.mat'}
a=loadmat(paths['train'])['sarcos_inv']; b=loadmat(paths['test'])['sarcos_inv_test']; x=np.vstack([a,b]);
res={}
for k,z in [('train',a),('test',b),('all',x)]:
 uniq,counts=np.unique(z,axis=0,return_counts=True)
 ux,cx=np.unique(z[:,:21],axis=0,return_counts=True)
 res[k]={'shape':list(z.shape),'missing':int(np.isnan(z).sum()),'inf':int(np.isinf(z).sum()),'constant_columns':np.where(np.ptp(z,axis=0)==0)[0].tolist(),'duplicate_rows':int(len(z)-len(uniq)),'duplicate_feature_rows':int(len(z)-len(ux)),'min':z.min(axis=0).tolist(),'max':z.max(axis=0).tolist(),'mean':z.mean(axis=0).tolist(),'std':z.std(axis=0).tolist(),'target_quantiles':np.quantile(z[:,21],[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist(),'file_sha256':hashlib.sha256(Path(paths[k]).read_bytes()).hexdigest() if k!='all' else None}
# exact overlap
at={row.tobytes() for row in a}; ax={row.tobytes() for row in a[:,:21]}
res['overlap']={'test_exact_row_in_train':sum(row.tobytes() in at for row in b),'test_exact_features_in_train':sum(row.tobytes() in ax for row in b[:,:21])}
# adjacent euclidean distance standardized by train spread
sc=StandardScaler().fit(a[:,:21]); aa=sc.transform(a[:,:21]);bb=sc.transform(b[:,:21]);
for k,z in [('train',aa),('test',bb)]:
 d=np.linalg.norm(np.diff(z,axis=0),axis=1)
 # random nonadjacent comparable pairs
 rng=np.random.default_rng(42); i=rng.integers(0,len(z),size=20000);j=rng.integers(0,len(z),size=20000);dr=np.linalg.norm(z[i]-z[j],axis=1)
 res[k]['adjacent_dist_quantiles']=np.quantile(d,[0,.1,.5,.9,.99,1]).tolist()
 res[k]['random_pair_dist_quantiles']=np.quantile(dr,[0,.1,.5,.9,.99,1]).tolist()
 res[k]['adjacent_target_corr']=float(np.corrcoef((a if k=='train' else b)[:-1,21],(a if k=='train' else b)[1:,21])[0,1])
# near neighbors sampled query, approximate with brute matrix over candidate subset, reproducible, no claim full exact NN
rng=np.random.default_rng(42)
qi=rng.choice(len(b),min(500,len(b)),replace=False)
ci=rng.choice(len(a),min(10000,len(a)),replace=False)
D=cdist(bb[qi],aa[ci],metric='euclidean')
mins=D.min(axis=1)
res['sampled_nearest_test_to_train']={'n_test_queries':len(qi),'n_train_candidates':len(ci),'distance_quantiles':np.quantile(mins,[0,.01,.05,.25,.5,.75,.95,1]).tolist(),'count_distance_below_0_1':int((mins<.1).sum()),'count_distance_below_0_5':int((mins<.5).sum())}
# baseline random split versus row segment distance (sampled)
tr,te=train_test_split(np.arange(len(a)),test_size=.2,random_state=42)
for k,ii,jj in [('random',tr,te),('contiguous',np.arange(int(.8*len(a))),np.arange(int(.8*len(a)),len(a)))]:
 i=rng.choice(ii,min(500,len(ii)),replace=False); j=rng.choice(jj,min(10000,len(jj)),replace=False)
 ds=cdist(aa[i],aa[j]).min(axis=1)
 res['sampled_'+k+'_split_nn']={'train_candidates':len(j),'test_queries':len(i),'quantiles':np.quantile(ds,[0,.05,.5,.95,1]).tolist()}
print(json.dumps(res,indent=2)[:16000]);(ROOT/'reports/sarcos_audit_metrics.json').write_text(json.dumps(res,indent=2))
