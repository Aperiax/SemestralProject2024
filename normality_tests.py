import scipy.stats as stats
import numpy as np
import math
import scipy.optimize as scp


class tests:
    """
    I'll just pass the data in the statistics object into here, I will be calling this 
    module from the CurvesAndStats.py anyway and statistics class there already has raw_data
    in constructor.
    """
    @staticmethod
    def testlognormal(raw_data, id) -> bool:

        data: np.ndarray = raw_data.fillna(1).to_numpy()[:, id]

        cont = []
        for i in data.tolist():
            if i > 0:
                a = math.log(float(i))
                cont.append(a)
            else:
                continue

# TODO: Implement the actual decision
# IDEA: Couldn't it be lognormal distro?

        data = np.array(cont)
        skew = stats.skew(np.array(data), bias=False, nan_policy="omit")
        kurt = stats.kurtosis(np.array(data), bias=False, nan_policy="omit")
        LEN = len(data)

        d_skew = (6 * (LEN - 2)) / ((LEN + 1) * (LEN + 3))
        d_kurt = (24 * LEN * (LEN - 2) * (LEN - 3)) / ((LEN + 1) ** 2 * (LEN + 3) * (LEN + 5))
        e_skew = 0
        e_kurt = 6 / (LEN + 1)

        TK_SKEW = (skew - e_skew) / math.sqrt(d_skew)
        TK_KURT = (kurt - e_kurt) / math.sqrt(d_kurt)
        print(f"tk_A = {TK_SKEW}, tk_E = {TK_KURT}")
        alpha = 1 - 0.025
        critical = stats.norm.ppf(alpha)

        if critical > TK_SKEW and critical > TK_SKEW:
            return True   # this means that the distribution is lognormal
        else:
            return False  # the distribution isn't lognormal

    @staticmethod
    def testnormal(raw_data, id) -> bool:

        data: np.ndarray = raw_data.fillna(0).to_numpy()[:, id]
        skew = stats.skew(np.array(data), bias=False, nan_policy="omit")
        kurt = stats.kurtosis(np.array(data), bias=False, nan_policy="omit")
        LEN = len(data)

        d_skew = (6 * (LEN - 2)) / ((LEN + 1) * (LEN + 3))
        d_kurt = (24 * LEN * (LEN - 2) * (LEN - 3)) / ((LEN + 1) ** 2 * (LEN + 3) * (LEN + 5))
        e_skew = 0
        e_kurt = 6 / (LEN + 1)

        TK_SKEW = (skew - e_skew) / math.sqrt(d_skew)
        TK_KURT = (kurt - e_kurt) / math.sqrt(d_kurt)
        print(f"tk_A = {TK_SKEW}, tk_E = {TK_KURT}")
        alpha = 1 - 0.025
        critical = stats.norm.ppf(alpha)

        if critical > TK_SKEW and critical > TK_SKEW:
            return True   # this means that the distribution is normal
        else:
            return False

