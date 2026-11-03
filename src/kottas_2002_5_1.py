#! /usr/bin/env python3

from dataclasses import dataclass, field, InitVar
import numpy as np
from scipy.stats import lognorm, binom, gamma
import matplotlib.pyplot as plt


@dataclass
class TwoLognormalMix():

    mu_1: float=0
    mu_2: float=0
    sigma_1: float=1
    sigma_2: float=1
    p: float=0.5
    n: int=100
    seed: int=250118

    p1_n: int=field(init=False)
    p2_n: int=field(init=False)
    lgn_mix: np.ndarray=field(init=False)

    def __post_init__(self):

        self.p1_n = round(self.p * self.n)
        self.p2_n = self.n - self.p1_n

        lgn_1 = lognorm.rvs(
            s=np.sqrt(self.sigma_1), 
            scale=np.exp(self.mu_1), 
            size=self.p1_n, 
            random_state=self.seed)
        lgn_2 = lognorm.rvs(
            s=np.sqrt(self.sigma_2), 
            scale=np.exp(self.mu_2), 
            size=self.p2_n, 
            random_state=self.seed)
        self.lgn_mix = np.append(lgn_1, lgn_2)


def main() -> None:

    mu_1 = 0
    mu_2 = 1.2
    sigma_1 = 0.25
    sigma_2 = 0.02
    p = 0.8
    n = 2000
    p1_n = round(p * n)
    p2_n = n - p1_n
    seed = 721383

    lgn_mix = TwoLognormalMix(
        mu_1=mu_1, mu_2=mu_2, sigma_1=sigma_1, sigma_2=sigma_2, p=p, n=n, seed=seed)


    alpha = 2
    beta = 0.9
    n = 2000
    seed = 786112
    gamma.rvs(a=alpha, loc=0, scale=1/beta, size=n, random_state=seed)
    x = np.linspace(start=0, stop=12, num=1000)
    gy = gamma.pdf(x=x, a=alpha, loc=0, scale=1/beta)


    '''
    plt.clf()
    plt.close()
    plt.hist(lgn_mix.lgn_mix, bins=50)

    plt.show()


    plt.clf()
    plt.close()
    plt.plot(x, gy)

    plt.show()
    '''




if __name__ == '__main__()':
    main()

