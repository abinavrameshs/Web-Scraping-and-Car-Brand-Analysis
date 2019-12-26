import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from nltk.corpus import stopwords
import nltk
from nltk.corpus import reuters
from nltk.corpus import wordnet
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from operator import itemgetter
from sklearn.metrics import classification_report
import csv
import os
import itertools
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import Select
import time
from bs4 import BeautifulSoup
from sklearn import manifold


car_comments=pd.read_csv("car_comments.csv")
models=pd.read_csv("models_v1.csv",encoding = "latin")

car_comments.columns

#creating a new dataframe contains only comment
text_data_A = car_comments.loc[:,["comments"]]

#drop na. purpose: avoid erro
text_data_A['comments'].dropna(inplace=True)

#tokenize words
text_data_A["tokenized_comments"] = text_data_A["comments"].apply(nltk.word_tokenize)

# create a combined list of words

text_data_A["tokenized_comments"].dropna(inplace=True)

# Remove stop words and convert into lowercase

def content_without_stopwords(lst):
    content = list()
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w.lower() for w in lst if (w.lower() not in stopwords)& (w.lower().isalpha())]
    return content

text_data_A["tokenized_without_stop"] = text_data_A["tokenized_comments"].apply(lambda x : content_without_stopwords(x))


######################################### change models to brands
model_list = list(models.models)
brands_list = list(models.brands)

def remove_xao(list1) : 
    lst=list()
    for i in list1 : 
        i = i.replace(u'\xa0', u' ')
        i=i.strip()
        lst.append(i)
    return lst

model_list = remove_xao(model_list)
brands_list = remove_xao(brands_list)

text_data_A["tokenized_without_stop"].dropna(inplace=True)

#  replace models with brands
def replace_models(list2):
    list1 = list()
    for i in list2 : 
        try : 
            index = model_list.index(i)
            list1.append(brands_list[index])
        except : 
            list1.append(i)
    return list1

text_data_A["tokenized_without_stop_model_replaced"]=text_data_A["tokenized_without_stop"].apply(lambda x : replace_models(x))


# keep only unique per post

text_data_A["tokenized_without_stop_model_replaced"].dropna(inplace=True)

def keep_unique(lst):
    list1 = list()
    list1 = list(set(lst))
    return list1

text_data_A["tokenized_without_stop_model_replaced_unique"]=text_data_A["tokenized_without_stop_model_replaced"].apply(lambda x : keep_unique(x))

##### SQuish everything into one list

text_data_A["tokenized_without_stop_model_replaced_unique"].dropna(inplace=True)

combined_tokens = list()

for i in text_data_A["tokenized_without_stop_model_replaced_unique"]:
    for j in i: 
            combined_tokens.append(j.lower())

## Keep only those words that have brands
            
brands_present=list()

for i in combined_tokens :
    if i in brands_list : 
        brands_present.append(i)


frequencies=nltk.FreqDist(tag for tag in brands_present)
frequencies.most_common(100)
all_freqs=dict(frequencies)

top_10_brands=list()
for a,b in list(frequencies.most_common(12)) : 
    top_10_brands.append(a)
    
    


'''''''''''''''''''''''''''''''''''''''
TASK A
'''''''''''''''''''''''''''''''''''''''

#top_10_brands = #['honda','ford','toyota','hyundai','nissan','sedan','chevrolet','mazda','saturn','chrysler']
top_10_brands =['honda',
 'toyota',
 'ford',
 'nissan',
 'mazda',
 'hyundai',
 'chevrolet',
 'saturn',
 'chrysler',
 'kia']

all_combinations = list(itertools.combinations(top_10_brands, 2))

# Having all indivudual counts
probs_individual_counts=dict(frequencies)

# getting combined counts



pair_counts = dict()
    
for a,b in all_combinations:
    count=0
    for doc in text_data_A["tokenized_without_stop_model_replaced_unique"]: 
        if ((a in doc)&(b in doc)):
            count+=1
        else:
            count+=0
    pair_counts[(a,b)]=count


lift = dict()
N=len(text_data_A["tokenized_without_stop_model_replaced_unique"])

for a,b in all_combinations:
    lift[(a,b)]=N*((pair_counts[(a,b)])/((probs_individual_counts[a]*probs_individual_counts[b])))


###### distance = 1/lift          
########## Generate MDS Plot ########
# Create the array of "dissimilarities" (distances) between points
#top_10_brands = ['honda','ford','toyota','hyundai','nissan','sedan','chevrolet','mazda','saturn','chrysler']
distance=[]
for i in range(len(top_10_brands)):
    temp = [0] * len(top_10_brands)
    for j in range(len(temp)):
        if j==i : 
            temp[j] = 0
        else : 
            a=top_10_brands[i]
            b=top_10_brands[j]
            try : 
                temp[j]=1/lift[(a,b)]
            except : 
                temp[j]=1/lift[(b,a)]
                
    distance.append(temp)
    
    
similarity=np.array(distance)


mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
results = mds.fit(similarity)

coords = results.embedding_

plt.subplots_adjust(bottom = 0.1)
plt.scatter(
    coords[:, 0], coords[:, 1], marker = 'o'
    )
for label, x, y in zip(top_10_brands, coords[:, 0], coords[:, 1]):
    plt.annotate(
        label,
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.show()


'''''''''''''''''''''''''''''''''''''''
TASK B
'''''''''''''''''''''''''''''''''''''''

'''''''''''''''''''''''''''''''''''''''
TASK C
'''''''''''''''''''''''''''''''''''''''


### To select the common attributes of a car

##################################################  Scraping to get attributes of a car

############################# 1st link ###########
#instantiating chrome driver
driver_path = '/Users/abinavrameshsundararaman/Documents/McGill/Courses/Winter 2019/Advanced Info Systems/Group Assignment-2/chromedriver'
driver = webdriver.Chrome(executable_path= driver_path)
driver.get("https://www.words-to-use.com/words/cars-vehicles/")

#explicit waiting
driver.wait = WebDriverWait(driver, 2)
driver.find_element_by_xpath('//*[@id="content"]/div[2]/ul/li[1]/a').click()
time.sleep(2)


#parsing using BeautifulSoup
soup_stackoverflow = BeautifulSoup(driver.page_source,"html.parser")


attr_list = list()

for i in range(1,4):
   #explicit waiting
   driver.wait = WebDriverWait(driver, 2)
   xpath_text = '//*[@id="content"]/div[2]/ul/li['+str(i)+']/a'
   driver.find_element_by_xpath(xpath_text).click()
   time.sleep(2)
   
   soup_stackoverflow = BeautifulSoup(driver.page_source,"html.parser")
   
   for div in soup_stackoverflow.findAll('div', attrs={'class':'col-xs-6 col-md-3'}):
       text=div.find('ol',attrs={'class':'section-col list-unstyled'}).text
       attr_list.append(text)
   for div in soup_stackoverflow.findAll('div', attrs={'class':'col-xs-6 col-md-3 col-md-push-3'}):
       text=div.find('ol',attrs={'class':'section-col list-unstyled'}).text
       attr_list.append(text)
   for div in soup_stackoverflow.findAll('div', attrs={'class':"col-xs-6 col-md-3 col-md-pull-3"}):
       text=div.find('ol',attrs={'class':'section-col list-unstyled'}).text
       attr_list.append(text)

driver.close()
len(attr_list)

## to get a clean list of attributes

attr_list_clean = list()
for i in attr_list : 
    lst = list()
    lst=i.split(sep="\n")
    for j in lst : 
        if j.isalpha():
            attr_list_clean.append(j)

############################# 2nd link ###########
#instantiating chrome driver
driver_path = '/Users/abinavrameshsundararaman/Documents/McGill/Courses/Winter 2019/Advanced Info Systems/Group Assignment-2/chromedriver'
driver = webdriver.Chrome(executable_path= driver_path)
driver.get("https://describingwords.io/for/cars")

#explicit waiting
driver.wait = WebDriverWait(driver, 2)
driver.find_element_by_xpath('//*[@id="word-click-hint"]').click()
time.sleep(2)

#parsing using BeautifulSoup
soup_stackoverflow = BeautifulSoup(driver.page_source,"html.parser")
driver.close()

attr_list3=list()
for div in soup_stackoverflow.findAll('div', attrs={'class':'words'}):
    for div1 in soup_stackoverflow.findAll('span', attrs={'class':'item'}):
        try : 
            text=div1.find('span',attrs={'class':'word-sub-item'}).text
            #fetching the tags attached to a question
            attr_list3.append(text)
        except : 
            continue

len(attr_list3)

attr_list_clean2=list()
for i in attr_list3 : 
    i=i.strip()
    attr_list_clean2.append(i)

combined_attrs = list(set(attr_list_clean + attr_list_clean2))
len(combined_attrs)


# how many of the extracted attributes are common in text
common_attributes=list(set(combined_tokens).intersection(combined_attrs))
len(combined_tokens)
len(common_attributes)

pd.DataFrame(common_attributes).to_excel('attributes.xlsx', header=False, index=False)

### Frequency of common attributes in texts

text_data_A["tokenized_without_stop_model_replaced_unique"]


unique_text_combined=list()
for i in text_data_A["tokenized_without_stop_model_replaced_unique"] : 
    unique_text_combined+=i

frequencies_attributes=nltk.FreqDist(tag for tag in unique_text_combined if tag in common_attributes )


frequencies_attributes.most_common(100)

# We lemmatize words
wnl = nltk.WordNetLemmatizer()
lemmatized_attributes=[wnl.lemmatize(t) for t in common_attributes]

frequency_lemmatized = nltk.FreqDist(word for word in unique_text_combined if word in lemmatized_attributes)

frequency_lemmatized.most_common(100)

# keep only Nouns and adjectives for description

tagged_attributes=nltk.pos_tag(lemmatized_attributes)

lst = list()
for a,b in tagged_attributes:
    if b in ['NN','JJ']:
        lst.append(a)
len(lst)

#### DUMP IT IN excel sheet..WE CATEGORIZE MANUALLY AND THEN IMPORT BACK

pd.DataFrame(lst).to_excel('attributes_v1.xlsx', header=False, index=False)


#### We categorize them into general category.. then we import the excel sheet


categorize_attributes=pd.read_excel("attributes_v1.xlsx")
categorize_attributes.to_dict()
categorize_attributes=categorize_attributes[categorize_attributes.general_attributes !="blank"]

text_data_A["tokenized_without_stop_model_replaced_unique"].dropna(inplace=True)

attribute_list = list(categorize_attributes.attributes)
general_attribute_list = list(categorize_attributes.general_attributes)

def replace_attributes(list2):
    list1 = list()
    for i in list2 : 
        try : 
            index = attribute_list.index(i)
            list1.append(general_attribute_list[index])
        except : 
            list1.append(i)
    return list(set(list1))

text_data_A["tokenized_without_stop_model_replaced_unique_attr_replaced"]=text_data_A["tokenized_without_stop_model_replaced_unique"].apply(lambda x : replace_attributes(x))


combined_general_attributes= list()

text_data_A["tokenized_without_stop_model_replaced_unique_attr_replaced"].dropna(inplace=True)

#SQUISH ALL ELEMENTS INTO 1 LIST
for i in text_data_A["tokenized_without_stop_model_replaced_unique_attr_replaced"]:
    combined_general_attributes+=i

# ONLY INCLUDE GENERAL ATTRIBUTES
combined_general_attributes_clean = list()

for i in combined_general_attributes:
    if(i in general_attribute_list):
        combined_general_attributes_clean.append(i)

# CHECK THE FREQUENCY OF EACH TYPE
        
################################################## This gives the top attributes in the text
frequencies_attributes=nltk.FreqDist(tag for tag in combined_general_attributes_clean  )

frequencies_attributes.most_common(100)

##################################################  See which brand is associated with which attribute
top_10_brands=list()
for a,b in list(frequencies.most_common(10)) : 
    top_10_brands.append(a)
top_10_brands.append('bmw')

top_10_attributes=list()
for a,b in list(frequencies_attributes.most_common(5)) : 
    top_10_attributes.append(a)
    
all_combinations_brand_attr=list(itertools.product(top_10_brands, top_10_attributes))

# Having all indivudual counts
probs_individual_brand_counts=dict(frequencies)
probs_individual_attribute_counts=dict(frequencies_attributes)

# getting combined counts

pair_counts_brand_attr = dict()
    
for a,b in all_combinations_brand_attr:
    count=0
    for doc in text_data_A["tokenized_without_stop_model_replaced_unique_attr_replaced"]: 
        if ((a in doc)&(b in doc)):
            count+=1
        else:
            count+=0
    pair_counts_brand_attr[(a,b)]=count


lift_brand_attr = dict()
N=len(text_data_A["tokenized_without_stop_model_replaced_unique_attr_replaced"])

for a,b in all_combinations_brand_attr:
    lift_brand_attr[(a,b)]=N*((pair_counts_brand_attr[(a,b)])/((probs_individual_brand_counts[a]*probs_individual_attribute_counts[b])))


# Convert lift values to a dataframe

list_brand=list()
list_attr=list()
for a, b in list(lift_brand_attr.keys()):
    list_brand.append(a)
    list_attr.append(b)
list(lift_brand_attr.values())  

lift_brand_attr_df=pd.DataFrame(list(zip(list_brand, list_attr, list(lift_brand_attr.values()))),columns=['brand','attribute','lift'])


##################################################  Calculate conditional lift for top models--with respect to performance and equipment
top_10_brands=list()
for a,b in list(frequencies.most_common(10)) : 
    top_10_brands.append(a)
top_10_brands.append('bmw')


ultimate_driving_attributes = ['performance','equipment']

triplets=list()
for i in top_10_brands:
    lst=list()
    lst.append(i)
    triplets.append(tuple(lst+['performance','equipment']))
    


# Having all indivudual counts
probs_individual_brand_counts=dict(frequencies)

pair_counts_brand_attr



# getting combined counts

triplets_count=dict()

    
for a,b,c in triplets:
    count=0
    for doc in text_data_A["tokenized_without_stop_model_replaced_unique_attr_replaced"]: 
        if ((a in doc)&(b in doc)&(c in doc)):
            count+=1
        else:
            count+=0
    triplets_count[(a,b,c)]=count



conditional_lift = dict()
N=len(text_data_A["tokenized_without_stop_model_replaced_unique_attr_replaced"])

for a,b,c in triplets:
    conditional_lift[(a,b,c)]=triplets_count[(a,b,c)]*((probs_individual_brand_counts[a])/((pair_counts_brand_attr[(a,b)]*pair_counts_brand_attr[(a,c)])))


######################################################### For RELIABLE CATEGORIES #######
    
    #### We categorize them into general category.. then we import the excel sheet

categorize_attributes.columns
categorize_attributes=pd.read_excel("attributes_v1.xlsx")
categorize_attributes.to_dict()
categorize_attributes=categorize_attributes[categorize_attributes.reliable !="F"]

text_data_A["tokenized_without_stop_model_replaced_unique"].dropna(inplace=True)

attribute_list = list(categorize_attributes.attributes)
general_attribute_list = list(categorize_attributes.general_attributes)

def replace_attributes(list2):
    list1 = list()
    for i in list2 : 
        try : 
            index = attribute_list.index(i)
            list1.append('reliable123')
        except : 
            list1.append(i)
    return list(set(list1))



text_data_A["tokenized_without_stop_model_replaced_unique_attr_replaced_reliable"]=text_data_A["tokenized_without_stop_model_replaced_unique"].apply(lambda x : replace_attributes(x))


combined_general_attributes= list()

text_data_A["tokenized_without_stop_model_replaced_unique_attr_replaced_reliable"].dropna(inplace=True)

#SQUISH ALL ELEMENTS INTO 1 LIST
count_reliable=0
for i in text_data_A["tokenized_without_stop_model_replaced_unique_attr_replaced_reliable"]:
    try : 
        index = i.index('reliable123')
        count_reliable+=1
    except:
        continue



##################################################  See which brand is associated with which attribute ###
top_10_brands=list()
for a,b in list(frequencies.most_common(100)) : 
    top_10_brands.append(a)

    
all_combinations_brand_attr=list(itertools.product(top_10_brands, ['reliable123']))

# Having all indivudual counts
probs_individual_brand_counts=dict(frequencies)


count_reliable

# getting combined counts

pair_counts_brand_attr = dict()
    
for a,b in all_combinations_brand_attr:
    count=0
    for doc in text_data_A["tokenized_without_stop_model_replaced_unique_attr_replaced_reliable"]: 
        if ((a in doc)&(b in doc)):
            count+=1
        else:
            count+=0
    pair_counts_brand_attr[(a,b)]=count


lift_brand_attr = dict()
N=len(text_data_A["tokenized_without_stop_model_replaced_unique_attr_replaced_reliable"])

for a,b in all_combinations_brand_attr:
    lift_brand_attr[(a,b)]=N*((pair_counts_brand_attr[(a,b)])/((probs_individual_brand_counts[a]*count_reliable)))


# Convert lift values to a dataframe

list_brand=list()
list_attr=list()
for a, b in list(lift_brand_attr.keys()):
    list_brand.append(a)
    list_attr.append(b)
list(lift_brand_attr.values())  

lift_brand_attr_df=pd.DataFrame(list(zip(list_brand, list_attr, list(lift_brand_attr.values()))),columns=['brand','attribute','lift'])
