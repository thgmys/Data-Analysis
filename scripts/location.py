import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv("../data/listings.csv")
df=df.dropna(subset=['last_review']) ## drop all df that do not have any reviews as not relevant
df.reset_index(drop=True, inplace=True)

nhbrd=[]
df2=df['neighbourhood_cleansed'].dropna()
df2.reset_index(drop=True, inplace=True)
print(df2)

for i in df2:
    area = i.split(',')[0]
    # print(i)
    if i not in nhbrd:
        nhbrd.append(i)
area=[i.split(',')[0] for i in nhbrd]

df=pd.read_csv('../data/cleaned_listings.csv')
print(df)
df2=pd.DataFrame()
df2 = df.loc[:,'host_is_superhost':'number_of_reviews']  # Select columns by range
print(df2)
X=df2.to_numpy()
y=df['review_scores_rating']
print(y)

def raw():
    d={}
    #averages of each respective scoring for each of the areas in dublin
    compare='location'
    for i in nhbrd:
        score=df.loc[df['neighbourhood_cleansed']==i]['review_scores_' + compare].mean()
        d.update({i.split(',')[0]:score})

    # copy=d.copy()
    # for k,v in copy.items():
    #     if v<4.5:
    #         d.pop(k)
    plt.title('Location rating based on location of listing')
    plt.ylim(4,5)
    plt.bar(*zip(*d.items()))
    plt.xticks(rotation=20,fontsize=10)
    plt.show()
    print(d)

def onehot():
    df2=df[['neighbourhood_cleansed', 'accommodates', 'number_of_reviews', 'review_scores_location']].dropna()
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder.fit(df2[['neighbourhood_cleansed']])
    one_hot_features = onehot_encoder.transform(df2[['neighbourhood_cleansed']])
    df2['one_hot']=one_hot_features

    print(df2)
onehot()