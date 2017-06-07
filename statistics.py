import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])
print(df.describe())
# df.groupby('emotion').count()
print(df['emotion'].value_counts())
