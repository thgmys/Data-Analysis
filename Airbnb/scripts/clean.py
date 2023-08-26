import pandas as pd

df=pd.read_csv("../data/listings.csv")
df=df.dropna(subset=['last_review']) ## drop all df that do not have any reviews as not relevant
df=df.drop(columns=['listing_url','scrape_id','last_scraped','source','name','description','neighborhood_overview','picture_url','host_url','host_name',
'host_since','host_location','host_about','host_response_time','host_response_rate','host_acceptance_rate','host_thumbnail_url','host_picture_url','host_neighbourhood'
,'calculated_host_listings_count_shared_rooms','calculated_host_listings_count_private_rooms','host_verifications','license','instant_bookable','calculated_host_listings_count_entire_homes',
'host_has_profile_pic','host_id','calculated_host_listings_count','first_review','number_of_reviews_ltm','number_of_reviews_l30d','calendar_last_scraped','has_availability',
'availability_30','availability_60','availability_90','availability_365','maximum_nights_avg_ntm','minimum_nights_avg_ntm','maximum_maximum_nights','minimum_maximum_nights',
'maximum_minimum_nights','minimum_minimum_nights','maximum_nights','minimum_nights','bathrooms_text','room_type','property_type','longitude','latitude','neighbourhood_group_cleansed','neighbourhood',
'calendar_updated','host_total_listings_count','neighbourhood_cleansed','bathrooms','amenities','last_review','review_scores_accuracy','review_scores_value',
'review_scores_checkin','review_scores_communication','review_scores_cleanliness','beds','bedrooms'])
df.dropna()
df.reset_index(drop=True, inplace=True) # remove unimportant columns
df.host_is_superhost = df.host_is_superhost.replace({'t': 1, 'f': 0})
df.host_identity_verified = df.host_identity_verified.replace({'t': 1, 'f': 0})

price = df.loc[:,'price']
p=[]
for i in price:
    i = i.replace('$','')
    i = i.replace(',','')
    p.append(float(i))
df['price']=p

df.to_csv('../data/cleaned_listings.csv')
print(df)
