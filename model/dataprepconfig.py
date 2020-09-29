from sklearn import preprocessing

global minval
global maxval
global min_max_scaler
global catagory_features
global number_features

min_max_scaler = preprocessing.MinMaxScaler()
text_features = ['keywords','original_title']
catagory_features = ['production_companies','Director_1','Director_2','Director_3','Actor_1','Actor_2','Actor_3','Actor_4','Actor_5','Genres_Action','Genres_Adventure','Genres_Animation','Genres_Comedy','Genres_Crime','Genres_Documentary','Genres_Drama','Genres_Family','Genres_Fantasy','Genres_Foreign','Genres_History','Genres_Horror','Genres_Music','Genres_Mystery','Genres_Romance','Genres_Thriller','Genres_War','Genres_Western']
number_features = ['budget', 'revenue','runtime','vote_average','vote_count','number_of_cast','number_of_director']
all_selected_features = ['budget', 'id', 'keywords', 'original_title', 'popularity','production_companies', 'revenue', 'runtime', 'vote_average','vote_count', 'Genres_Action', 'Genres_Adventure', 'Genres_Animation','Genres_Comedy', 'Genres_Crime', 'Genres_Documentary', 'Genres_Drama','Genres_Family', 'Genres_Fantasy', 'Genres_Foreign', 'Genres_History','Genres_Horror', 'Genres_Music', 'Genres_Mystery', 'Genres_Romance', 'Genres_Thriller','Genres_War', 'Genres_Western', 'number_of_cast', 'number_of_director','Director_1', 'Director_2', 'Director_3', 'Actor_1', 'Actor_2','Actor_3', 'Actor_4', 'Actor_5']
eliminate_if_empty_list = ['production_companies','Director_1','Director_2','Director_3','Actor_1','Actor_2','Actor_3','Actor_4','Actor_5','budget', 'revenue','runtime','vote_average','vote_count']
