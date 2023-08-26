import pandas as pd
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer

df=pd.read_csv("../data/reviews.csv")
df=df.dropna() # cleanup data by removing reviews with null values
df.reset_index(drop=True, inplace=True)

res={}
sia=SentimentIntensityAnalyzer()
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    comment=row['comments']
    id=row['id'] # id of review
    res[id]=sia.polarity_scores(comment)
vaders=pd.DataFrame(res).T
vaders.to_csv('../data/review_sentimentanalysis.csv')