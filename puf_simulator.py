from pypuf.simulation import *
import pypuf.metrics as pm
from pypuf.io import random_inputs
import pypuf.io, pypuf.simulation

class PUFs():
    def __init__(self, stages=64, similarity=2, PUF_seed=666):
        self.pufs = []
        self.responses = []
        self.stages = stages
        self.seed = PUF_seed
        self.similarity = similarity
        self.challenges = None
        self.n_pufs = None
        self.hereroFF_PUF = []
        ## Phenotype
        self.phenotype_seed = None

    def add_XOR_PUF(self,k,num,noise=0):
        for _ in range(num):
            # puf = XORArbiterPUF(n=self.stages,k=k,seed=self.seed,scale2=self.similarity)
            puf = XORArbiterPUF(n=self.stages,k=k,seed=self.seed, noisiness=noise)
            # seed increase
            self.seed += 1000
            self.pufs.append(puf)

    def add_FF_PUF(self,ff,num):
        for _ in range(num):
            puf = FeedForwardArbiterPUF(n=self.stages,ff=ff,seed=self.seed)
            self.seed +=10
            self.pufs.append(puf)

    def add_XORFF_PUF(self,k,ff,num):
         for _ in range(num):
            puf = XORFeedForwardArbiterPUF(n=self.stages,k=k,ff=ff,seed=self.seed)
            self.seed +=10
            self.pufs.append(puf)

    def add_herero_XORFF_PUFs(self,k,ff,num):
        for i in range(num):
            FF_PUF = []
            for j in range(k):
                puf = FeedForwardArbiterPUF(n=self.stages,ff=ff[j],seed=self.seed)
                self.seed += 10
                FF_PUF.append(puf)
            self.hereroFF_PUF.append(FF_PUF)

    def add_interpose_PUFs(self, ks_up, ks_down, num):
        for i in range(num):
            k_up,k_down = ks_up, ks_down
            puf = InterposePUF(self.stages,k_down=k_down,k_up=k_up)
            self.seed += 10
            self.pufs.append(puf)

    def generate_crps(self, c_seed=2,N=[3000]):
        self.challenges = random_inputs(n=self.stages,N=N,seed=c_seed)
        self.n_pufs = len(self.pufs)+len(self.hereroFF_PUF)
        for puf in self.pufs:
            r = puf.eval(self.challenges)
            r = (1-r) // 2
            self.responses.append(r)
        for ff_pufs in self.hereroFF_PUF:
            rs = []
            for ff_puf in ff_pufs:
                r = ff_puf.eval(self.challenges)
                r = (1-r) // 2
                rs.append(r)
            rs = np.array(rs)
            rs = np.bitwise_xor.reduce(rs,axis=0)
            self.responses.append(rs)

    # def mask(self,mask)

        return self.challenges, self.responses
    
    def generate_crps_loop(self,):
        pass