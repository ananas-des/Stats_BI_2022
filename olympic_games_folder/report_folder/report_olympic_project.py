#!/usr/bin/env python
# coding: utf-8

# # Project1. Olympic Games Athlets' Data for past 120 years

# ## Anastasia Zolotar

# ### 2022/11/24

# ## Structure of .csv data
# 
# For **the first Project** we are going to work with some data for *athlets participated in Olimpic Games during the past 120 years*.
# 
# Here we deal with *12 .csv files*. In each row there is an descriptive information for the *Olimpic Games athlet* (observation). The datasets have *the header* with the following features:
# 
# - **ID** – Unique number for each athlete
# - **Name** – Athlete's name
# - **Sex** – M or F
# - **Age** – Integer
# - **Height** – In centimeters
# - **Weight** – In kilograms
# - **Team** – Team name
# - **NOC** – National Olympic Committee 3-letter code
# - **Games** – Year and season
# - **Year** – Integer
# - **Season** – Summer or Winter
# - **City** – Host city
# - **Sport** – Sport
# - **Event** – Event
# - **Medal** – Gold, Silver, Bronze, or NA

# ## Importing necessary packages

# In[117]:


from os import walk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.stats import ttest_ind
from statannot import add_stat_annotation
from scipy.stats import chi2_contingency


# In[107]:


pip install git+https://github.com/webermarcolivier/statannot.git


# ## Task1. Loading the data

# For the **Task1** we need to combine the observations into a single dataframe. For this perpose the function **gen_df()** were created. It combines all files with a certain extension from a given folder and accepts three following arguments: directory path, files extension, and column separator for input files.

# In[118]:


def gen_df(path, extension, sep):
    '''Function gen_df() searches files with certain extension in given path
    and concatenates them in pandas.DataFame
    
    Parameters:
    path (str): path to directory with files
    extension (str): extension for files to concatenate
    sep (str): column separator for input files
    
    Returns: 
    df_res (pandas.DataFrame): the result of concatenation'''
    
    def for_skipfooter(file):
        with open(file, 'r') as file1:  
            first_line = file1.readline().strip().split(',')  
            last_line = file1.readlines()[-1].strip().split(',')
        if len(last_line) != len(first_line):
            return 1
        else:
            return 0
        
        
    filenames = next(walk(path), (None, None, []))[2]
    filenames = [i for i in filenames if i[-len(extension)-1:] == f'.{extension}']
    dtype_dict = {"ID": object}
    df_list = []
    for i in filenames:                
        df = pd.read_csv(
            path + i, sep=sep, header=0, dtype=dtype_dict, 
            skipfooter = for_skipfooter(path+i), engine='python'
        )
        df_list.append(df)
    df_res = pd.concat(df_list, axis=0, ignore_index=True)
    return df_res


# specify the directory path with files to concatenate 
path = "athlete_events/"
# specify files extention
extension = "csv"
# specify column separator
sep = ","
data = gen_df(path, extension, sep)
data


# *Comment*: I used additional file opening because the last lines in some of original .csv files are partial and in that way useless. I have compared the length of the first and last lines for each file and used *skipfooter=True* for skipping the wrong ones.

# In[175]:


data.to_csv("raw_olympic_data.csv")


# ## Task2. Basic information about data - EDA

# For the **Task2** we are going to perform EDA for our data and "fix the bugs".

# In[119]:


#Basic information
data.info()


# So, here we have 14 columns. The most of them has 'object' type. Noticeably, the 'Age', 'Height', 'Weight', and 'Year' belong to 'float' type. I tried to convert them into 'int' type but the error occured due to the presence of NA values. Thus, I decided to keep them float even for 'Year' (the 'Year' convertation into 'object' will make it impossible to deal with years as the numbers).   

# ### Checking for duplicates in data

# In[120]:


# Finding the number of duplicated observations
data.duplicated().sum()


# The above function have found 1385 duplicated observations. Let's look closer at them.

# In[121]:


duplicates = data.duplicated()
duplicates[duplicates == True]


# In[122]:


data.iloc[2004:2007]


# For example, rows from 2004-2006 are fully duplicated. So, let's remain the first of them (and do this for all duplicated data).

# Let's remove duplications

# In[123]:


data = data.drop_duplicates()
data


# Now we have 269720 unique observations (athlets) in total.

# ### Checking number of groups for categorial values 

# In this dataframe there are three main categorial groups: 'Sex', 'Season', and 'Medal'. Let's check the presence of something unexpected for them.  

# In[124]:


data['Season'].unique()


# In[125]:


data['Medal'].unique()


# In[126]:


data['Sex'].unique()


# As we expected there are two categories for 'Season': 'Summer', or 'Winter', four categories for 'Medal': nan, 'Gold', 'Silver', or 'Bronze'. But something happened for 'Sex':) We have unexpected 'G' sex for some athlets. Let's manualy check them.

# In[127]:


data[data.Sex == 'G']


# Visual inspection shows that both athlets with sex 'G' are men. Let's change 'Sex' value to 'M' for them.

# In[128]:


# fixing 'Sex' value for two athlets ('G' -> 'M')  
for i in (135128, 135160):
    data.at[i, 'Sex'] = 'M'


# ### Descriptive statistics

# Let's generate some descriptive statistics for numeric features in our data, such as athlets 'Age', 'Height', and 'Weight', using pandas `describe()` and box plot. 

# In[129]:


#describe the data
num_data = data[['Age', 'Height', 'Weight']] # subset with numeric values
num_data.describe().round(2)


# In[130]:


# boxplots for numeric data
plt.figure(figsize=(8, 6), dpi=90)
plt.title("Box plots for numeric data (with outliers)", weight='bold', fontsize=20)
sns.boxplot(num_data, palette='pastel')


# Obviously, there are outliers in our data. There are athlets with age near 250 years, or with height near 3.5 meters. For weight I decided to manualy check the kind of sports for outliers.

# In[131]:


data[data.Age > 200]


# In[132]:


data[data.Age == 10]


# Unfortunately, it is not yet possible to be at the age of 240. It's strong outlier. On the other hand, there is no age limits for taking part in Olympic Games. So, 10-years-old boy in Gymnastics might be possible.

# In[133]:


data[data.Age > 90]


# Wow! Previously, I haven't heard about [Art Competitions](https://en.wikipedia.org/wiki/Art_competitions_at_the_Summer_Olympics) at the Summer Olympic Games. Let's keep the athletes of advanced age in our data.

# In[134]:


data[data.Height > 300]


# Hm, 3.4 m woman. Unfortunately, outlier.

# In[135]:


data[data.Weight > 180]


# It seems OK for Judo and Wrestling Sports that athlets have the weight near 200 kg. Let's keep them in our data.

# So, outliers can be detected using **visualization**, **implementing mathematical formulas** on the dataset, or using the **statistical approach**. The most popular graphs for outliers visualization are *box plot with whiskers*, *scatterplot*, and *histogram*. *Z- Score* helps to understand that how far is the data point from the mean. And after setting up a threshold value we can utilize z-score values of data points to define the outliers. ALso to finding the outliers the *Inter Quartile Range (IQR)* is the most commonly used. To define the outlier, base value is defined above and below datasets normal range namely Upper and Lower bounds, define the upper and the lower bound (1.5*IQR value is considered).
# 
# In this case, I removed fully duplicated observations, fixed some "bugs" such as the presence of non-existent sex value, visualized outliers using box plot and checked them manualy to determine the possibility of their occurrence. As I noticed, each outlier does not affect other athlet feachures, so, I decided to replace them with NaN, calculate the median, and set the *median value* instead of outliers.

# In[136]:


# setting median values for outliers
data.at[112756, 'Height'] = np.nan
data.at[112567, 'Age'] = np.nan
data.at[112756, 'Height'] = data.Height.median()
data.at[112567, 'Age'] = data.Age.median()


# Now, let's visualize numeric data again.

# In[137]:


#describe the data
num_data2 = data[['Age', 'Height', 'Weight']] # subset with numeric values
num_data2.describe().round(2)


# In[138]:


# boxplots for numeric data
plt.figure(figsize=(8, 6), dpi=90)
plt.title("Box plots for numeric data (without outliers)", weight='bold', fontsize=20)
sns.boxplot(num_data2, palette='pastel')


# In[174]:


data.to_csv("processed_olympic_data.csv")


# ## Task3. What the age for the youngest Males and Females in 1992 Olympics?

# In[139]:


# the age of the youngest Olympic athlets in 1992
youngest_1992 = data[data.Year == 1992].groupby(['Sex']).min(['Age'])
youngest_1992


# In 1992 Summer Olympic Games the youngest female and male was 12 and 11 years-old athletes, respectively. 

# ## Task4. What are the mean and standart deviation for the both genders Height? 
# As long as many athletes have participated in multiple Olympics and Events, I think it is resonable to keep only one value for each athlet ID. Also we know that there is outliers for Height 

# In[140]:


sex_height = data.drop_duplicates(['ID']).groupby(['Sex']).agg({'Height': ['mean', 'std']})
sex_height.round(2)


# The mean *Height* for **Females** is $(168.93 \pm 8.51)$ and for **Males** is $(179.44 \pm 9.46)$.

# ## Task5. What are the mean and standart deviation for Female tennisists Height in 2000 Olympic Games?
# The same way with previous task, I decided to drop duplicates by ID.

# In[141]:


height_fem_2000 = data.drop_duplicates(['ID']).loc[
    (data.Sex == 'F') & (data.Year == 2000) & (data.Sport == 'Tennis')
].agg(
    {'Height': ['mean', 'std']}
)
height_fem_2000.round(1)


# So, *the mean Height for Female tennisists in 2000 Olympics Games* is $(172.4 \pm 6.4)$.

# ## Task6. What kind of sport in the 2006 Olympics Games did the most heavyweight athlete participate in?

# In[142]:


data.at[data.Weight[data.Year == 2006].idxmax(), 'Sport']


# The most heavyweight athlete participated in the *Skeleton* in the 2006 Olympics Games. 

# ## Task7. How many Gold medals did women win between 1980 and 2010?

# For this taks I decided to include the period bounderies in calculations.

# In[143]:


years = np.arange(1980, 2011)
women_gold = data.query('Medal == "Gold" & Sex == "F"')
women_gold[women_gold['Year'].isin(years)].shape[0]


# In[144]:


years = np.arange(1980, 2011)
men_gold = data.query('Medal == "Gold" & Sex == "M"')
men_gold[men_gold['Year'].isin(years)].shape[0]


# In 1980-2010 women won **2249 Gold medals**. I was also interested in Gold medals number for men. Hm, the difference in number is near 1200 medals.

# ## Task8. How many Olympic games has John Aalberg participated in?

# In[145]:


jaalberg_games = data[data.Name == 'John Aalberg'].Games.unique()
jaalberg_games


# John Aalberg has participated in **two Winter Olympic Games** in 1992 and 1994 but in 8 Events.

# ## Task9. Which age groups are the most and the least abundant in the 2008 Olympics? 

# Let's group our data by Age groups, such as \[10-15), [15-25), [25-35), [35-45), [45-55), [55-65), [65-70), and create column 'AgeGroup' for this age ranks. Also we need to take in account that each athlet could participate more then one Event (take only unique IDs).

# In[146]:


# subsetting for the 2008 Olympic Games
data_2008 = data[data.Year == 2008]
# setting age ranges
age_ranges = pd.IntervalIndex.from_arrays(
    [10, 15, 25, 35, 45, 55, 65], 
    [15, 25, 35, 45, 55, 65, 70], closed='left'
)
# categorizing data by age groups 
age_cat = pd.cut(data_2008['Age'], bins=age_ranges)
# inserting column 'AgeGroup'
data_2008.insert(4, 'AgeGroup', age_cat, True)
# counting number of athlets in each age group
age_groups_count = data_2008.groupby('AgeGroup').ID.nunique()
age_groups_count


# In the 2008 Olympic Games **the most of athlets** (5382 people) was in **the age from 25 to 35**, and **the 65-70 age group** was **the least abundant** with only one athlet. 

# ## Task10. What is the difference between the number of Sports in the Olympic Games 1994 and 2002?

# In[147]:


data[data.Year == 2002].Sport.unique()


# In[148]:


data[data.Year == 1994].Sport.unique() 


# In[149]:


data[data.Year == 2002].Sport.nunique() - data[data.Year == 1994].Sport.nunique()


# In 2002 there were **three** new Winter Sports (Curling, Snowboarding, and Skeleton) as compared to the 1994 Olympic Games.

# ## Task11. Top 3 countries by different types of medals in the Summer and Winter Olympic Games

# In[150]:


# subsetting data by 'Season', 'Medal', and 'Team' 
# and counting 'Medals' for each 'Team'
country_medals_subset = data.groupby(
    ['Season', 'Medal', 'Team']
).agg(
    n_Medals = ('Medal', 'count')
).reset_index(
    ['Season', 'Medal', 'Team']
)

# transforming 'Medal' to Categorical
medal_categories = ["Gold", "Silver", "Bronze"]
country_medals_subset['Medal'] = pd.Categorical(
    country_medals_subset['Medal'], categories = medal_categories
)

# getting top 3 countries indicies
country_medals_subset2 = country_medals_subset.groupby(
    ['Season', 'Medal'])['n_Medals'].nlargest(3).reset_index(
    ['Season', 'Medal'])

# resulted dataframe
top_countries = country_medals_subset.iloc[country_medals_subset2.index].sort_values(
    by=['Season', 'Medal']
).groupby(
    ['Season', 'Medal']
).head(3)
top_countries


# The resulted `top_countries` dataframe contains **18 top countries** sorted by number of Medals and Medal type for both Seasons.

# ## Task12. Athlets' Heights standardization

# For this Task I created new variable **Height_z_scores** for athlets' Heights standartization and added it into subset `athlets_features`.

# In[151]:


# subsetting data with some athlets features
athlets_features = data[data.columns[0:8]]
athlets_features


# In[152]:


# calculating Height zscores and putting them into Height_z_scores column
Height_z_scores = zscore(athlets_features.Height, nan_policy = 'omit')
athlets_features.insert(5, column='Height_z_scores', value = Height_z_scores)


# In[153]:


athlets_features


# ## Task13. Athlets' Heights min-max normalization

# In[154]:


Height_min_max_scaled = (athlets_features.Height - athlets_features.Height.min())/                        (athlets_features.Height.max() - athlets_features.Height.min())
athlets_features.insert(6, column='Height_min_max_scaled', value = Height_min_max_scaled)


# In[155]:


athlets_features


# ## Task14. Athlets' Height, Weight and Age comparison between men and women in the Winter Olympics Games 

# For this Task I created subset with appropriate features, performed t-test, and visualized results.

# In[156]:


# subsetting data for the Winter Olympic Games by 'Sex','Age', 'Height', and 'Weight'
df_comparison = data[data['Season'] == 'Winter'][
    ['ID','Sex','Age','Height','Weight']
].drop_duplicates().drop(columns=['ID'])
df_comparison


# In[157]:


# female group
fem_group = df_comparison[df_comparison.Sex == 'F']
# male group
male_group = df_comparison[df_comparison.Sex == 'M']


# In[158]:


# mean and std for Sex groups
df_comparison.groupby(['Sex']).agg(['mean', 'std']).round(2)


# In[159]:


# t-test for Age
ttest_ind(fem_group.Age, male_group.Age, equal_var=False, nan_policy = 'omit')


# In[160]:


# t-test for Height
ttest_ind(fem_group.Height, male_group.Height, equal_var=False, nan_policy = 'omit')


# In[161]:


# t-test for Weight
ttest_ind(fem_group.Weight, male_group.Weight, equal_var=False, nan_policy = 'omit')


# In[164]:


# violin plots
cols = ['Age','Height','Weight']
fig, axs = plt.subplots(1, 3, figsize=(14,7), dpi=300)
fig.tight_layout(pad=2.0)

fig.suptitle("Difference between Athlets' Height, Weight and Age for the Winter Olympic Games", weight='bold', fontsize=20, y=1)
for i, ax in enumerate(axs):
    sns.violinplot(x= 'Sex', y=cols[i] ,ax=ax, data=df_comparison, orient='v', palette="pastel")
    # add_stat_annotation() function from package 'statannot'
    add_stat_annotation(ax, data=df_comparison, x='Sex', y=cols[i],
                        box_pairs=[('M', 'F')],
                        test='t-test_ind', text_format='simple', loc='inside', verbose=0)


# ## Task15. Are Team and Medal features independent? 
# 
# In order to check I performed Chi-square test. For this test I decided to filter Teams that have at least 5 of each type of medals.

# In[165]:


# creating appropriate dataframe
df = pd.crosstab(index=data['Team'],columns=data['Medal'])
df = df[(df>=5).sum(axis=1) == 3]
CrosstabResult = df
CrosstabResult


# 79 teams passed filtering and won all kinds of medals. What does the Chi-square test say?

# In[166]:


# Chi-square test using chi2_contingency()
ChiSqResult = chi2_contingency(CrosstabResult) 
pval = ChiSqResult[1]
print(f'ChiSq test p-value: {pval}')
print('Variables are' +' not'*int(pval > 0.05) + ' correlated!') # Just for fun.


# With grate certainty we can say that not all Teams are equal. Some of them have won medals of certain kind more frequently then others.

# ## Additional Task16. Testing hypothesis

# - Is it possible that Summer Olympics Athlets are slimmer than the Winter Olympics Athlets?

# In[171]:


# BMI - Body mass index - mass(kg)/height(m)^2. 
data['BMI'] = data['Weight']/(data['Height']*.01)**2

first = data[data['Season'] == 'Winter'].drop_duplicates('ID')['BMI']
second = data[data['Season'] == 'Summer'].drop_duplicates('ID')['BMI']

fig, ax = plt.subplots(figsize=(8, 6), dpi=90)
plt.title("Athlets Body mass index for the Summer and Winter Olympics", weight='bold', fontsize=20)
sns.violinplot(x= 'Season', y='BMI', data=data.drop_duplicates('ID'), orient='v', palette='pastel')
add_stat_annotation(ax=ax, data=data.drop_duplicates('ID'), x='Season', y='BMI',
                    box_pairs=[('Summer', 'Winter')],
                    test='t-test_ind', text_format='simple', loc='inside', verbose=0)
print(f'Winter Olympics Athlets average BMI: {round(first.mean(), 2)}, Summer Olympics Athlets BMI: {second.mean()}')
print(ttest_ind(first, second, nan_policy='omit'))


# Yes! The Summer Olympics Athlets have lower BMI with hight significance (t-test p-value < $1e^{-5}$) but averagely all of them are lean and beautifull:)

# - Now let's do this for Male and Female Athlets. BMI for Male Athlets is significantly higher then for Female Athlets (t-test p-value < $1e^{-5}$).

# In[172]:


# subsetting data
first = data[data['Sex'] == 'M'].drop_duplicates('ID')['BMI']
second = data[data['Sex'] == 'F'].drop_duplicates('ID')['BMI']

fig, ax = plt.subplots(figsize=(8, 6), dpi=90)
plt.title("Body mass index for Male(M) and Female(F) Olympics Athlets", weight='bold', fontsize=20)
sns.violinplot(x= 'Sex', y='BMI', data=data.drop_duplicates('ID'), orient='v', palette='pastel')
add_stat_annotation(ax=ax, data=data.drop_duplicates('ID'), x='Sex', y='BMI',
                    box_pairs=[('M', 'F')],
                    test='t-test_ind', text_format='simple', loc='inside', verbose=0)

print(f'Male average BMI: {round(first.mean(), 2)}, Female average BMI: {round(second.mean(), 2)}')
print(ttest_ind(first, second, nan_policy='omit'))


# - Is it true that female figure skaters are getting younger every year? Surprisingly, it's more complicated. Even though average age is decreasing now, global minimum was in the 1960th.

# In[173]:


# subsetting data
df = data[(data['Sport'] == 'Figure Skating') & (
    data['Sex'] == 'F')].drop_duplicates(
    ['ID','Year']).groupby('Year')['Age'].mean()

fig, ax = plt.subplots(dpi=150)
plt.title('Age of female figure skaters by season', fontweight='bold', fontsize=16)
df.plot(figsize=(10,5), ylabel='Age')

