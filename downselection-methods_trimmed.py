import numpy as np, matplotlib.pyplot as plt, polars as pl, pickle
from scipy.stats import pearsonr
from CarpetBagging_viv import CarpetBaggingRegressor
from dppy.finite_dpps import FiniteDPP
from scipy.integrate import simpson
from scipy.optimize import brenth
from time import time


rng = np.random.default_rng()
cl = pl.col


centers   = np.array([[ 0, 0, 0],
                    [ 5, 5, 5],
                    [-2, 0, 1],
                    [-5, 1, 2]])
stds      = np.array([[0.5, 0.5, 0.5],
                      [1.0, 1.0, 1.0],
                      [0.8, 0.8, 0.8],
                      [0.1, 0.1, 0.5]])
counts    = [5000, 1000, 2500, 15]

counts_val = [200,200,200,200]

df = pl.read_csv('../datasets/quarter_sphere_weather.csv')

feature_labels = [f"x{i}" for i in range(10)]
target = ['y0']

X = df.select(feature_labels).to_numpy()
y = df.select(target).to_numpy().flatten()

feature_mask = X.std(0)!=0
X = X[:,feature_mask] 

n_tr = int(len(X)*0.05)

Xtr = X[:n_tr]
ytr = y[:n_tr]

Xval = X[n_tr:]
yval = y[n_tr:]

nest = 100
cbr = CarpetBaggingRegressor(n_estimators=nest,random_state=1234)
cbr.model.fit(X,y)

cbr_preds = np.array([est.predict(Xtr) for est in cbr.model.estimators_]).T

def logpdet(A):
    # only works for hermitian A
    loglams = np.log(abs(np.linalg.eigvalsh(A)))
    logpdet = loglams[loglams>-15].sum()
    return logpdet

def loss_coreset(preds,y,N=30,Ns=None):
    ''' Loss Based Scores:
        |- score points based on how bad the model predicts this point
    '''
    losses = (y-preds.mean(1))**2
    scores = losses/losses.sum()
    if Ns is not None:
        samples_ns, weights_ns = [], []
        for n in Ns:
            samples = rng.choice(np.arange(len(scores)),size=n,p=scores/scores.sum(),replace=False)
            weights = scores[samples]/N
            samples_ns.append(samples)
            weights_ns.append(weights)
        return samples_ns, weights_ns
    else:
        samples = rng.choice(np.arange(len(scores)),size=N,p=scores/scores.sum(),replace=False)
        weights = scores[samples]/N
        return samples, weights

def dpp_loss_coreset(X,preds,y,tau=2,N=300,Ns=None):
    norm_X  = X.copy()
    norm_X -= np.mean(norm_X,axis=0)
    norm_X /= np.std(norm_X,axis=0)
    q = abs(y-preds.mean(1))
    if Ns is not None:
        samples_ns, scores_ns = [], []
        for n in Ns:
            samples,scores = DPP_sampler(q/q.sum(),X,tau=tau,N=n)
            samples_ns.append(samples)
            scores_ns.append(scores)
        return samples_ns,scores_ns
    else:
        samples,scores = DPP_sampler(q/q.sum(),X,tau=tau,N=N)
        return samples,scores


def dpp_dom_coreset(X,diff_of_means,tau=2,N=300,Ns=None):
    norm_X  = X.copy()
    norm_X -= np.mean(norm_X,axis=0)
    norm_X /= np.std(norm_X,axis=0)
    q = abs(diff_of_means)
    if Ns is not None:
        samples_ns, scores_ns = [], []
        for n in Ns:
            samples,scores = DPP_sampler(q/q.sum(),X,tau=tau,N=n)
            samples_ns.append(samples)
            scores_ns.append(scores)
        return samples_ns,scores_ns
    else:
        samples,scores = DPP_sampler(q/q.sum(),X,tau=tau,N=N)
        return samples,scores


def DPP_sampler(q,X,tau=2,N=300):
    # for internal use in methods
    sftpls  = lambda s: np.log1p(np.exp(-np.abs(s))) + np.maximum(s, 0)
    R = 2000
    while R<N*2:
        R += 100
    w = rng.normal(0,1,size=(R,X.shape[-1]))
    b = rng.uniform(0,2*np.pi,size=R)
    B = np.sqrt(2/R)*np.cos(X@w.T+b)
    B *= q[:,None]
    evals,evecs = np.linalg.eigh(B.T@B)
    samples = []
    N_target = N
    while len(samples)<N_target:
        print((sftpls(-1e7)*evals.clip(0)/(1+sftpls(-1e7)*evals.clip(0))).sum(),
              (sftpls(1e70)*evals.clip(0)/(1+sftpls(1e70)*evals.clip(0))).sum())
        s = brenth(lambda s: (sftpls(s)*evals.clip(0)/(1+sftpls(s)*evals.clip(0))).sum()-N,-1e7,1e70,maxiter=20000)
        L_evecs = np.nan_to_num(B@evecs/np.sqrt(sftpls(s)*evals),0)
        p = (sftpls(s)*evals.clip(0))/(1+sftpls(s)*evals.clip(0))
        sel_inds = np.where(rng.binomial(1,p))[0]
        V = L_evecs[:,sel_inds]
        Q = np.linalg.qr(V)[0]
        pdpp = FiniteDPP('correlation',K=Q@Q.T,projection=True)
        samples = pdpp.sample_exact()
        if len(samples)<N_target:
            print(f"failed: {len(samples)} samples less than target {N_target}")
            N += 50
    scores = np.diag(pdpp.K).copy()
    scores /= scores.sum()
    return samples[:N], scores[samples[:N]]/N


def KL(cbr,Xtr,Xstar,tau=2,N=300,Ns=None):
    oob_inds, inb_inds = cbr.get_oob_samples(Xtr)
    all_inds           = np.arange(len(Xtr))
    inb_mask           = np.array([np.isin(all_inds,i) for i in inb_inds]).T
    #sig                = 1e-5
    cbr_preds_Xstar    = np.array([est.predict(Xstar) for est in cbr.model.estimators_]).T
    KLs                = np.zeros(len(Xtr))
    for ind in range(len(Xtr)):
        print('kl',ind)
        #n_ib,n_ob      = inb_mask[ind].sum(), (~inb_mask[ind]).sum()
        preds_ib       = cbr_preds_Xstar[:, inb_mask[ind]]
        preds_ob       = cbr_preds_Xstar[:,~inb_mask[ind]]
        mu_ib          = preds_ib.mean(1)
        mu_ob          = preds_ob.mean(1)
        sig_ib         = np.cov(preds_ib)
        sig_ob         = np.cov(preds_ob)
        sig_ob_inv     = np.linalg.pinv(sig_ob,hermitian=True)
        KL             = (sig_ob_inv@sig_ib).trace()
        KL            += (mu_ob-mu_ib)@(sig_ob_inv@(mu_ob-mu_ib))
        KL            -= len(Xstar)
        KL            += logpdet(sig_ib) - logpdet(sig_ob)
        KL            /= 2
        KLs[ind]       = KL
    q                  = KLs-KLs.min()
    if Ns is not None:
        samples_ns, scores_ns = [], []
        for n in Ns:
            samples,scores = DPP_sampler(q/q.sum(),Xtr,tau=tau,N=n)
            samples_ns.append(samples)
            scores_ns.append(scores)
        return samples_ns,scores_ns
    else:
        samples,scores = DPP_sampler(q/q.sum(),Xtr,tau=tau,N=N)
        return samples,scores

name = '5% Quarter Sphere Weather'
reps = 20

labels = ['rand','retrain diff of means','Loss Coreset','Loss-DPP','DoM-DPP','KL-DPP']
nkeeps = np.arange(10,400,50)
r2s = np.zeros((len(nkeeps),len(labels),reps))
maes = np.zeros((len(nkeeps),len(labels),reps))
mses = np.zeros((len(nkeeps),len(labels),reps))
r2s_noweight = np.zeros_like(r2s)
full_r2s = []
full_maes = []
full_mses = []
for rep in range(reps):
    reorder = rng.permutation(np.arange(len(X)))
    Xtr,ytr = X[reorder][:n_tr], y[reorder][:n_tr]
    Xva,yva = X[reorder][n_tr:], y[reorder][n_tr:]
    nest = 100
    cbr = CarpetBaggingRegressor(n_estimators=nest,random_state=1234)
    cbr.model.fit(Xtr,ytr)
    cbr_preds = np.array([est.predict(Xtr) for est in cbr.model.estimators_]).T
    oob_inds, inb_inds = cbr.get_oob_samples(Xtr)
    all_inds           = np.arange(len(Xtr))
    inb_mask           = np.array([np.isin(all_inds,i) for i in inb_inds]).T
    inb_preds = []
    oob_preds = []
    for i,pred in enumerate(cbr_preds):
        inb_preds.append(pred[ inb_mask[i]])
        oob_preds.append(pred[~inb_mask[i]])
    inb_preds_mean = np.array([i.mean() for i in inb_preds])
    inb_preds_std  = np.array([i.std()  for i in inb_preds])
    oob_preds_mean = np.array([i.mean() for i in oob_preds])
    oob_preds_std  = np.array([i.std()  for i in oob_preds])
    diff_of_means  = inb_preds_mean - oob_preds_mean
    diff_of_stds   = np.sqrt(np.abs(inb_preds_std**2 - oob_preds_std**2))
    D = np.vstack((inb_preds_mean, inb_preds_std, oob_preds_mean, 
                   oob_preds_std, diff_of_means, diff_of_stds)).T
    full_r2s.append(pearsonr(cbr.model.predict(Xva),yva)[0]**2)
    full_maes.append(abs(yva-cbr.model.predict(Xva)).mean())
    full_mses.append(((yva-cbr.model.predict(Xva))**2).mean())
    #
    N = nkeeps.max()
    ranks = [(rng.permutation(np.arange(len(Xtr))),np.ones(len(Xtr)))]

    t0 = time()
    keep_inds_ns = []
    cur_rem_inds = np.arange(len(Xtr))
    for n in nkeeps[::-1]:
        prune_itt = 5
        prune_amt = (len(Xtr[cur_rem_inds])-n)//prune_itt
        down_X,down_y,err,keep_inds = cbr.down_selection(Xtr[cur_rem_inds],ytr[cur_rem_inds],
                                                         prune_itt=prune_itt,prune_amt=prune_amt)
        keep_inds_ns.append(cur_rem_inds[keep_inds])
        cur_rem_inds = cur_rem_inds[keep_inds]
    ranks.append([keep_inds_ns[::-1],[np.ones(len(keep_inds_ns[::-1][i])) for i in range(len(keep_inds_ns))]])
    print(f"t retrain diff of means: {time()-t0:0.2f}")

    t0 = time()
    ranks.append(loss_coreset(cbr_preds,ytr,Ns=nkeeps))
    print(f"t loss i.i.d. coreset: {time()-t0:0.2f}")

    t0 = time()
    ranks.append(dpp_loss_coreset(Xtr,cbr_preds,ytr,Ns=nkeeps,tau=1))
    print(f"t dpp loss coreset: {time()-t0:0.2f}")
    t0 = time()
    ranks.append(dpp_dom_coreset(Xtr,diff_of_means,Ns=nkeeps,tau=1))
    print(f"t dpp dom coreset: {time()-t0:0.2f}")

    t0 = time()
    ranks.append(KL(cbr,Xtr,Xval[rng.choice(len(Xval),300)],Ns=nkeeps))
    print(f"t kl: {time()-t0:0.2f}")

    for i,nkeep in enumerate(nkeeps):
        for nmod in range(len(labels)):
            if len(ranks[nmod][0])==len(nkeeps):
                nkeep_inds = ranks[nmod][0][i]
                weights    = ranks[nmod][1][i]
            else:
                nkeep_inds = ranks[nmod][0][:nkeep]
                weights    = ranks[nmod][1][:nkeep]
            cbr_mod = CarpetBaggingRegressor(n_estimators=nest,random_state=1234)
            cbr_mod.model.fit(Xtr[nkeep_inds],ytr[nkeep_inds],sample_weight=weights)
            cbr_mod_preds = cbr_mod.model.predict(Xva)
            r2s[i,nmod,rep] = pearsonr(cbr_mod_preds,yva)[0]**2
            maes[i,nmod,rep] = abs(yva-cbr_mod_preds).mean()
            mses[i,nmod,rep] = ((yva-cbr_mod_preds)**2).mean()
    print(rep,r2s[-1,:,rep])
    print(rep,maes[-1,:,rep])
    print(rep,mses[-1,:,rep])
    np.save('r2s.npy',r2s)
    np.save('maes.npy',maes)
    np.save('mses.npy',mses)
    np.save('full_r2s.npy',np.array(full_r2s))
    np.save('full_maes.npy',np.array(full_maes))
    np.save('full_mses.npy',np.array(full_maes))
    np.save(f'reorder-{rep}.npy',reorder)
    with open(f'ranks-{rep}.pickle','wb') as f:
        pickle.dump(ranks,f)

from matplotlib.colors import LinearSegmentedColormap
#plt.rcParams['font.family'] = 'Lato'
cmap = LinearSegmentedColormap.from_list('name',['dodgerblue','crimson','seagreen'])

f,ax = plt.subplots()
violins = ax.violinplot([i.flatten() for i in r2s],positions=nkeeps,widths=np.diff(nkeeps)[0]/2,
                        showextrema=False)
[i.set_fc(('dodgerblue',0.25)) for i in violins['bodies']]
order = simpson(r2s.mean(-1),axis=0).argsort()[::-1]
for i in order:
    ax.scatter(nkeeps,np.mean(r2s[:,i],-1),label=labels[i],lw=1,c=cmap(i/(len(order)-1)),marker='o')

ax.legend(frameon=False)
plt.savefig(f'{name}-violin.png',dpi=600,bbox_inches='tight')
#plt.show()
plt.close()

f,ax = plt.subplots(2,1,figsize=(8,12))
order = simpson(r2s.mean(-1),axis=0).argsort()[::-1]
for i in order:
    ax[0].plot(nkeeps,np.log(np.mean(maes[:,i],-1)),label=labels[i],lw=1,c=cmap(i/(len(order)-1)))
    ax[1].plot(nkeeps,np.mean(r2s[:,i],-1),label=labels[i],lw=1,c=cmap(i/(len(order)-1)))

ax[0].axhline(np.log(np.mean(full_maes)),lw=1.2,ls='dotted',c='crimson')
ax[1].axhline(np.mean(full_r2s),lw=1.2,ls='dotted',c='crimson')
ax[0].legend(frameon=False)
ax[0].set_xlabel('coreset size')
ax[1].set_xlabel('coreset size')
ax[0].set_ylabel('log mae on validation')
ax[1].set_ylabel('r$^2$ on validation')
ax[0].set_title('MAE',weight='bold')
ax[1].set_title('R$^2$',weight='bold')
plt.tight_layout()
#name = 'sm bs'
plt.savefig(f'{name}.png',dpi=600,bbox_inches='tight')
#plt.show()
