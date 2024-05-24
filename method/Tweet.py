import pandas as pd
import numpy as np

class Tweet:

    @staticmethod
    def get_all_Datas(csv_directory):
        data=pd.read_csv(csv_directory,header=None)
        datas=np.asarray(data)
        print(datas)



