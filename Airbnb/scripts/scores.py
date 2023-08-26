import pandas as pd
df=pd.read_csv("../data/review_sentimentanalysis.csv")
df2=pd.read_csv("../data/reviews.csv")
df2=df2.dropna() # cleanup data by removing reviews with null values
df2.reset_index(drop=True, inplace=True)
df2['score']=df['compound']
df2=df2.sort_values(by=['id'])
df2.to_csv('../data/reviews_with_score.csv')