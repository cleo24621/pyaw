import pandas as pd

from zh1 import configs
from zh1.read.scm import SCMUlf


class EFDUlf(SCMUlf):
    def __init__(self, fp):
        super().__init__(fp)
        self.fs = configs.efdulf_fs
        self.row_len = 256
        # self.target_fs = 25
        self.df1c_efd = self._concat_data()

    def _concat_data(self):
        """
        refer to SCMUlf.concat_data()
        Returns:

        """
        dict1c = {}
        for key in configs.efd_ulf_1c_vars:
            dict1c[key] = self.dfs[key].squeeze().values
        df1c = pd.DataFrame(index=self.datetime.values, data=dict1c)
        return df1c