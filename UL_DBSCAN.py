print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import rtpipe
from scipy.stats import gaussian_kde
from scipy import linalg as la

#Our data
# fig0 = "cands_14A-425_sb29612394_1.56903.3271372338_sc24seg0.pkl"
fig1 = "cands_14A-425_14jun21_sc175.pkl"
# fig2 = "cands_14A-425_14jun21_sc172.pkl"
fig3 = "cands_14A-425_14jun21_sc174.pkl"
fig4 = "cands_14A-425_14jun21_sc173.pkl"

fig5 = "cands_12A-336_sb9667618_1b.56040.87127945602_sc5.pkl"
# fig6 = "cands_12A-336_sb9667618_1b.56040.87127945602_sc6.pkl"
# fig7 = "cands_12A-336_sb9667618_1b.56040.87127945602_sc7.pkl"
# fig8 = "cands_12A-336_sb9667618_1b.56040.87127945602_sc8.pkl"
# fig9 = "cands_12A-336_sb9667618_1b.56040.87127945602_merge.pkl"

# fig10 = "cands_15B-305_sb31388707_1.57331.269053298616_merge.pkl"

candsfile = fig1
loc, prop, d = rtpipe.parsecands.read_candidates(candsfile, returnstate=True)

times = rtpipe.parsecands.int2mjd(d, loc)
dms = np.array(d['dmarr'])[loc[:,3]]
snrs = prop[:,0]

X0 = np.vstack((times,dms)).transpose()
X = StandardScaler().fit_transform(X0)

#------------------------Compute DBSCAN-----------------------------------
#DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, p=None, random_state = None)
#euclidean, cityblock 

candsfile = "cands_12A-336_sb9667618_1b.56040.87127945602_sc8.pkl"
db = DBSCAN(eps=.09, min_samples=2, metric='cityblock').fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)



#-------------Black removed and is used for noise instead-----------------

unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14, alpha=.4)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6, alpha=.8)


clusters = [X0[labels == i] for i in xrange(n_clusters_)]




plt.xlabel('Time(s)')
plt.ylabel('DM(pc/cc)')
plt.title('Estimated Number of Clusters: %d' % n_clusters_)
plt.show()
