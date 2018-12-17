import os
import pandas as pd
import numpy as np
from scipy import stats
import requests
import json
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing,tree
from sklearn.metrics.pairwise import cosine_similarity
os.chdir('C:\\Users\\kunal\\Desktop\\final_Recon')

originalData = pd.read_csv("processed_data.csv",encoding='latin-1')
grouped_data = pd.read_table("processed_categories_frequency.csv",sep=",",encoding='latin-1')

sns.pairplot(grouped_data.iloc[:,1:])
plt.savefig('PairPlots.png')


def venueCheckinCount():
    venueData = pd.read_csv("processed_data.csv",encoding='latin-1')
    venueData.columns = ["UserID","VenueID","VenueCategoryID","Subcategory","Category"]
	 # Sorting the data into table based on the venueIDs. 
    count = venueData["VenueID"].value_counts().reset_index()
	 # Creating data frame of the venue based sorted data.
    count.columns = ['Var1','Freq']
    count = count.sort_values(['Freq'],ascending = [0])
    return venueData,count

#venueData,count = venueCheckinCount()

def clusterCreation():
	  # Reading the processed data containing frequencies based on venues.
    grouped_data = pd.read_csv("processed_categories_frequency.csv",encoding = 'latin-1')
    row_count,col_count = grouped_data.shape
        
    for i in range(1,col_count):
        for j in range(0,row_count):
            mean_val = stats.trim_mean(grouped_data.iloc[:,i],0.1)
            if (grouped_data.iloc[j,i] > mean_val):
                grouped_data.iloc[j,i] =1
            else:
                grouped_data.iloc[j,i] =0
                
	  # Identifying Test & Training Dataset  
	  # Forming Test data which consist of 10% of the total dataset    
    test_data = grouped_data.iloc[972:1083,:]
    training_data = grouped_data.iloc[:972,:]
    
	# Performing Kmeans Clustering, ignoring first column as it has the userID
    
    kmeans_result = KMeans(n_clusters = 7,n_init=1).fit(preprocessing.scale(training_data.iloc[:,1:].values))
    training_data['class'] = kmeans_result.labels_
    
    ind_labels =training_data.columns[1:10] 
    dep_labels = training_data.columns[10]
    
    # Using Recursive Partitioning and Classification Trees

    clf = tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=2, min_samples_leaf=1)
    clf.fit(training_data[ind_labels],training_data[dep_labels])
    
    # Model prediction using classification algorithm and test dataset

    classes_identified = clf.predict(test_data[ind_labels] )
    
    return grouped_data,training_data,classes_identified

	# Finding Similar user from the user set present in the training dataset using classes identified.
def	findingSimilarUsers(grouped_data, training_data, classes_identified):
    grouped_data_subcategory = pd.read_csv("processed_data_subcategories.txt",encoding='latin-1')
	  
	 # Forming Test data out of sub categories which consist of 10% of the total dataset
    test_data_subcategory = grouped_data_subcategory.iloc[972:1083,:]
    grouped_data_subcategory = grouped_data_subcategory.iloc[:972,:]
    
    test_count = test_data_subcategory.shape[0]
    
    cosinesim = pd.DataFrame()
    # test_count = 110
    for i in range(0,test_count):
        groupedData_subset = grouped_data.iloc[:,0][training_data.loc[training_data['class']==classes_identified[i]].index]
        groupedData_subset= grouped_data_subcategory.loc[groupedData_subset]
        
        usersim = [None]*groupedData_subset.shape[0]
        group_count = groupedData_subset.shape[0]
        
        # j =244
        for j in range(0,group_count):
            x= np.array(list(groupedData_subset.iloc[j,1:].values)).reshape(1,-1)
            y = np.array(list(test_data_subcategory.iloc[i,1:].values)).reshape(1,-1)
            try :
                usersim[j] = cosine_similarity(x,y )[0][0]
            except:
                usersim[j] = 0
            
            
        result = pd.DataFrame([list(groupedData_subset.iloc[:,0]),usersim]).T
        result.columns = ['SimUser','Sim']
        result = result.sort_values(['Sim'],ascending=[0])
        result = result.iloc[0:5,:]
        result['User'] = i+1
        
        cosinesim = pd.concat([cosinesim,result])
        
    return cosinesim 


 
def recommendedResults(userNum, originalData, cosinesim, frequencies, category):
    j = userNum
    test_data = originalData.loc[(originalData['UserID']==1070+j) & (originalData['Category']==category)]
    test_data = test_data['VenueID']
    
    # Finding unique test data
    test_data = list(set(test_data))
    
    # Finding places visited by similar user
    similarUserVenues = pd.DataFrame()
    
    # i =0
    for i in range(0,4):
        similarUserVenuesNew = originalData.loc[(originalData['UserID']==cosinesim.iloc[5*(j-1)+i,0]) & (originalData['Category']==category)]
        similarUserVenuesNew = similarUserVenuesNew['VenueID']
        similarUserVenues = pd.concat([similarUserVenues,pd.DataFrame(similarUserVenuesNew)])
        
    # Finding unique similar user venues
    similarUserVenues.columns = ['similarUserVenuesNew']
    similarUserVenues = similarUserVenues.drop_duplicates()
    
    unvisited_venues = similarUserVenues.loc[~(similarUserVenues['similarUserVenuesNew'].isin(test_data))]
    unvisited_venues = frequencies.loc[frequencies['Var1'].isin(unvisited_venues['similarUserVenuesNew'])]
    cat_subcat = originalData.iloc[:,3:4]
    
    if unvisited_venues.shape[0]<5:
        # Finding unique venues by categories
        VenueByCat = list(set(originalData.loc[originalData['Category']==category]['VenueID']))
        VenueByCat = frequencies.loc[frequencies['Var1'].isin(VenueByCat)]
        
        # Finding venues not visited by the user in the test data
        VenueByCat = VenueByCat[~(VenueByCat['Var1'].isin(test_data))]
        VenueByCat = VenueByCat[~(VenueByCat['Var1'].isin(frequencies['Var1']))]
        
        #  Number of venues to be added
        x = 5-(unvisited_venues.shape[0])
        
        unvisited_venues = pd.concat([unvisited_venues,VenueByCat.iloc[0:x,]])
    
    return unvisited_venues

def getNames(unvisited_venues):
    venueNames = []
    if unvisited_venues.shape[0]>0:
        for i in range(0,len(unvisited_venues.index)):
            vId = unvisited_venues.iloc[i,]['Var1']
            #vId = '49bbd6c0f964a520f4531fe3'
            url = "https://api.foursquare.com/v2/venues/"+vId+"?oauth_token=0KAKWZJ0R11DRVSY2JR0WVYRD55PARANMLEOYM45MLHJSCK3&v=20170506"
            
            obj = json.loads(requests.get(url).text)
            venueNames.append(obj['response']['venue']['name'])
        
    return venueNames
    

def main():
    originalData,frequencies = venueCheckinCount()
    grouped_data,training_data,classes_identified = clusterCreation()
    cosinesimtopusers = findingSimilarUsers(grouped_data,training_data, classes_identified)
    
    categories = pd.read_csv("Category.csv",header=None)
    categories.iloc[:,0] = categories.iloc[:,0].astype(str)
    
    again = 'y'
    
    while again!='n':
        userNum = int(input('Login ID: - \n'))
        msg = "Enter Category Number from the list \n \
        \n1:  Shop & Service                 \n2:  Outdoors & Recreation    \n3:  Residence \
		  \n4:  Professional & Other Places     \n5:  Food                     \n6:  Travel & Transport \
		  \n7:  Arts & Entertainment            \n8:  College & University       \n9:  Nightlife Spot \
		  \n10: Athletic & Sport \nCategory Number:"
          
        category = int(input(msg))
        categories.columns
        category = categories.iloc[category-1,][0]
	
        unvisit = recommendedResults(userNum,originalData,cosinesimtopusers,frequencies,category)
        withNames = getNames(unvisit)
        if len(withNames)>0:
            print('Top recommendations based on User History:\n')
            for i in range(0,len(withNames)):
                print('['+str(i+1)+'] '+withNames[i])
                
        else:
            print('No Recommendations Available')
        
        again = input(("\nDo you wish to continue (y/n): "))

main()

#cosinesim = cosinesimtopusers