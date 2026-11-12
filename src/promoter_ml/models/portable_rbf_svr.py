"""NumPy-only inference for exported StandardScaler + RBF SVR models."""
from pathlib import Path
import numpy as np

class PortableRBFSVR:
    """Inference-only parameters; never loads Python pickle objects."""
    schema = 'position_1mer_2mer_global_1to3mer_gc'
    def __init__(self, mean, scale, support_vectors, dual_coef, intercept, gamma, schema):
        self.mean=np.asarray(mean,dtype=np.float64); self.scale=np.asarray(scale,dtype=np.float64)
        self.support_vectors=np.asarray(support_vectors,dtype=np.float64); self.dual_coef=np.asarray(dual_coef,dtype=np.float64)
        self.intercept=float(intercept);self.gamma=float(gamma);self.feature_schema=str(schema)
        if self.feature_schema!=self.schema or self.mean.ndim!=1 or self.scale.shape!=self.mean.shape or self.support_vectors.ndim!=2 or self.support_vectors.shape[1]!=len(self.mean) or self.support_vectors.shape[0]==0 or self.dual_coef.shape!=(self.support_vectors.shape[0],):raise ValueError('Invalid SVR checkpoint shapes/schema')
        if not all(np.isfinite(x).all() for x in (self.mean,self.scale,self.support_vectors,self.dual_coef)) or not np.isfinite(self.intercept) or not np.isfinite(self.gamma) or np.any(self.scale<=0) or self.gamma<=0:raise ValueError('Invalid SVR checkpoint values')
        self.support_norm=np.einsum('ij,ij->i',self.support_vectors,self.support_vectors)
    @classmethod
    def from_pipeline(cls,pipeline):
        scaler,svr=pipeline.steps[0][1],pipeline.steps[1][1]
        if svr.kernel!='rbf' or np.asarray(svr.dual_coef_).shape[0]!=1:raise ValueError('Only single-output RBF SVR supported')
        return cls(scaler.mean_,scaler.scale_,svr.support_vectors_,svr.dual_coef_[0],svr.intercept_[0],svr._gamma,cls.schema)
    def predict(self,features,batch_size=128):
        x=np.asarray(features,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=len(self.mean) or not np.isfinite(x).all() or batch_size<1:raise ValueError('Invalid features or batch size')
        result=np.empty(len(x),dtype=np.float64)
        for start in range(0,len(x),batch_size):
            v=(x[start:start+batch_size]-self.mean)/self.scale
            distance=np.maximum(np.einsum('ij,ij->i',v,v)[:,None]+self.support_norm[None,:]-2*(v@self.support_vectors.T),0)
            result[start:start+len(v)]=np.exp(-self.gamma*distance)@self.dual_coef+self.intercept
        return result
    def save(self,path):
        np.savez_compressed(Path(path),format_version=np.array(1),feature_schema=np.array(self.feature_schema),mean=self.mean,scale=self.scale,support_vectors=self.support_vectors,dual_coef=self.dual_coef,intercept=np.array(self.intercept),gamma=np.array(self.gamma))
    @classmethod
    def load(cls,path):
        with np.load(Path(path),allow_pickle=False) as f:
            if int(f['format_version'])!=1:raise ValueError('Unsupported SVR checkpoint version')
            return cls(f['mean'],f['scale'],f['support_vectors'],f['dual_coef'],f['intercept'],f['gamma'],str(f['feature_schema']))
