import GMM
import numpy as np

gmm = GMM.GMM()

gmm.fit(np.array([[1,2,3,4],[2,3,4,5]]))
