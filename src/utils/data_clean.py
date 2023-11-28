import re
import pandas as pd

## Special chars to clean from the text
def rm_special_char(series): 
    return series.apply(lambda x : re.sub('[-[\] ]+', ' ', x).strip())


class clean_pipe: 
    def __init__(self, steps=[]):
        self.steps = steps
    def fit(self, series:pd.Series)->pd.Series:
        for step in self.steps: 
            series = step(series)
        return series
        
    