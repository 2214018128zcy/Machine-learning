import pandas as pd

df = pd.DataFrame({'feature1': [3, 4, 5, 6],
                   'feature2': [6, 7, 8, 9]})
df['feature_merge'] = df['feature1'] + df['feature2']
print(df)