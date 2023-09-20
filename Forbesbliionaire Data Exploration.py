#!/usr/bin/env python
# coding: utf-8

# # WELCOME HERE 

# In[1]:


from PIL import Image

im = Image.open("C:\\Users\TOJMARK LTD\\DATA SCIENCE PROJECT\\Apple stock prediction project\\MYIMAGE.png")
display(im)


# Hi, I'm a data enthusiast with a knack for making sense of numbers. I thrive on turning data into practical insights that drive business decisions. My background in marketing gives me an edge in understanding customer behavior. I love experimenting with data, using statistical tools and machine learning to find hidden patterns. My goal is to become a data scientist, supercharging my data skills. My journey is guided by a passion for ethical data practices and a strong belief in data's power to transform businesses.

# # INTRODUCTION
# The Forbes data was collected in 2022 and is a comprehensive collection of information about billionaires around the world. It provides valuable insights into the wealth distribution, industry dominance, and demographic characteristics of the world's wealthiest individuals. The data set includes key attributes such as the billionaire's name, net worth, age, country, source of wealth, and industry. Analyzing this data can help uncover trends, patterns, and relationships among billionaires, providing a deeper understanding of the global economic landscape. By exploring the Forbes data set, we can gain insights into the sources of wealth, geographic distribution, and other significant factors that contribute to the billionaire phenomenon. This data set serves as a valuable resource for studying wealth inequality, economic impact, and individual success in the modern world.
# 
# My objective is to perform exploratory data analysis (EDA) and glean valuable insights from this wealth of information. Through data visualization, statistical analysis, and data cleaning, I aim to uncover intriguing patterns and trends within the billionaire community. I'll be looking at aspects like the distribution of billionaires by age, wealth sources, and industries, among others.
# 
# This analysis can offer a unique glimpse into the world's wealthiest individuals, shedding light on factors influencing their success and the broader economic landscape. Additionally, it can provide valuable insights for investors, policymakers, and anyone interested in understanding global wealth distribution and the dynamics of billionaire fortunes.

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
sklearn.__version__


# In[ ]:





# In[2]:


data = pd.read_csv("C:\\Users\\TOJMARK LTD\\DATA SCIENCE COURCES\\DATA SCIENCE PROJECTS\\forbes_richman.csv", 
                   encoding='latin-1')
data.head()


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


data.notnull().count()


# # First removing the $ and B sign under the net worth columns and covert it to integer

# In[6]:


# Remove dollar sign ($) and "B" from the Net Worth column
data['Net Worth'] = data['Net Worth'].str.replace('$', '').str.replace('B', '')

# Convert the Net Worth column to numeric
data['Net Worth'] = pd.to_numeric(data['Net Worth'])

# Print the updated column
data.head()


# In[7]:


#Let get the richest 
richest_individual = data[data['Net Worth'] == data['Net Worth'].max()]
richest_individual
#It shows elon musk is the richest person as at 2022


# In[8]:


poorest_individual = data[data['Net Worth'] == data['Net Worth'].min()]
poorest_individual.tail(10)
#There are many indivudiua with 1 billionaire net worth


# In[9]:


#The top 10 billionaire 
top_10_individuals = data.nlargest(10, 'Net Worth')
top_10_individuals


# In[10]:


#The last 10 individual
last_10_individuals = data.nsmallest(10, 'Net Worth')
last_10_individuals.head(-1)


# # The Richest In The United State

# In[11]:


richest_USA = data[data['Country'] == 'United States'].nlargest(10, 'Net Worth')
richest_USA


# In[12]:


#Let check the richest by different industries
richest_by_industry = data.groupby('Industry')['Net Worth'].max().reset_index()
richest_by_industry


# # The automotive industry has the highest net worth, follow by  construction & engineering

# In[13]:


#let check industry with the most higest number of billionaire
# Count the number of billionaires by industry
industry_counts = data['Industry'].value_counts()

# Get the industry with the highest number of billionaires
most_common_industry = industry_counts.idxmax()
most_common_industry
#Finance & investment has the most hihest number of billionaire


# In[14]:


#Let check the number of billionaire by each country
# Group the data by country and count the number of billionaires in each country
country_counts = data.groupby('Country').size().reset_index(name="Number of Billionaires")

#sort the country counts by the index
country_counts.sort_values(by="Number of Billionaires", ascending=False)
country_counts.reset_index(drop=True, inplace=True)

billionaire_by_countyr = country_counts.nlargest(10, 'Number of Billionaires')
billionaire_by_countyr


# # Youngest billionaire

# In[15]:


#let sort it by age
sorted_data = data.sort_values('Age')

# Get the youngest 10 billionaires
youngest_10 = sorted_data.head(10)
youngest_10 


# # Relationship between Age and Net Worth of Billionaires

# In[16]:


#Let check the relationship between  age a
# Extract the 'Age' and 'Net Worth' columns
age = data['Age']
net_worth = data['Net Worth']

# Create a scatter plot
plt.scatter(age, net_worth, color="blue")
plt.xlabel('Age')
plt.ylabel('Net Worth')
plt.title('Relationship between Age and Net Worth of Billionaires')

# Calculate the correlation coefficient
correlation = age.corr(net_worth)
print('Correlation coefficient:', correlation)

# Display the scatter plot
plt.show()


# In[17]:


#Let try to visualise in another method

# Sort the DataFrame by age in ascending order
data_sorted = data.sort_values('Age')

# Extract the 'Age' and 'Net Worth' columns
age = data_sorted['Age']
net_worth = data_sorted['Net Worth']

# Plot the line graph
plt.plot(age, net_worth, marker='o')
plt.xlabel('Age')
plt.ylabel('Net Worth')
plt.title('Relationship between Age and Net Worth of Billionaires')

# Display the plot
plt.show()


# # The distribution of number of billionnaire by industry

# In[18]:


# Group the data by industry and count the number of billionaires in each industry
industry_counts = data['Industry'].value_counts()

# Plot the distribution of the number of billionaires by industry
plt.figure(figsize=(12, 6))
industry_counts.plot(kind='bar')
plt.title('Distribution of Number of Billionaires by Industry')
plt.xlabel('Industry')
plt.ylabel('Number of Billionaires')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # The distribution of number of billionnaire by country

# In[19]:


# Group the data by industry and count the number of billionaires in each country
country_counts = data['Country'].value_counts()
country_counts =country_counts.head(10)

# Plot the distribution of the number of billionaires by industry
plt.figure(figsize=(12, 6))
country_counts.plot(kind='bar')
plt.title('Distribution of Number of Billionaires by Industry')
plt.xlabel('Industry')
plt.ylabel('Number of Billionaires')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # The most of source of wealth

# In[20]:


# Group the data by source and calculate the total net worth
source_net_worth = data.groupby('Source')['Net Worth'].sum().sort_values(ascending=False)

# Plot the top sources of wealth
plt.figure(figsize=(12, 6))
source_net_worth.head(10).plot(kind='bar')
plt.title('Top Sources of Wealth')
plt.xlabel('Source')
plt.ylabel('Total Net Worth (Billions)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # The distribution of number of billionnaire by Age

# In[21]:


# Age distribution
plt.figure(figsize=(15, 6))
plt.hist(data['Age'], bins=30)
plt.title('Age Distribution of Billionaires')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# # Distribution of Billionaires by Industry Usng Pie Chart

# In[22]:


# Count the number of billionaires in each industry
industry_counts = data['Industry'].value_counts()

# Plot a pie chart of the distribution of billionaires by industry
plt.figure(figsize=(10, 6))
plt.pie(industry_counts, labels=industry_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Billionaires by Industry')
plt.axis('equal')
plt.show()


# In[23]:


# Create a pivot table of the net worth of billionaires by country and industry
pivot_table = data.pivot_table(values='Net Worth', index='Country', columns='Industry', aggfunc='sum')

# Plot a heatmap of the net worth of billionaires by country and industry
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt=".1f")
plt.title('Net Worth of Billionaires by Country and Industry')
plt.xlabel('Industry')
plt.ylabel('Country')
plt.show()


# # Distribution of Net Worth of Billionaires

# In[24]:


# Create a boxplot of the net worth of billionaires by industry
plt.figure(figsize=(10, 6))
plt.boxplot(data['Net Worth'], vert=False)
plt.title('Distribution of Net Worth of Billionaires')
plt.xlabel('Net Worth')
plt.ylabel('Billionaires')
plt.show()


# # Create a line chart of the net worth of billionaires by age

# In[25]:


# Create a line chart of the net worth of billionaires by age
df_sorted = data.sort_values('Age')
plt.figure(figsize=(10, 6))
plt.plot(df_sorted['Age'], df_sorted['Net Worth'])
plt.title('Net Worth of Billionaires by Age')
plt.xlabel('Age')
plt.ylabel('Net Worth')
plt.grid(True)
plt.show()


# In[26]:


data['Age'] = data['Age'].replace('.0', '')

# Convert the Net Worth column to numeric
data['Age'] = pd.to_numeric(data['Age'])

# Print the updated column
data.head()


# # Time-series data using motion charts.

# # Net Worth of Billionaires

# In[27]:


import plotly.express as px
fig = px.scatter(data, x='Age', y='Net Worth', color='Industry',
                 animation_group='Name',
                 hover_name='Name', range_x=[min(data['Age']), max(data['Age'])],
                 range_y=[min(data['Net Worth']), max(data['Net Worth'])])
fig.update_layout(title='Net Worth of Billionaires',
                  xaxis_title='Age', yaxis_title='Net Worth')
fig.show()


# # Net Worth of Billionaires by Age

# In[30]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(data['Age'], data['Net Worth'],  c=data['Age'], s=data['Age'])
plt.title('Net Worth of Billionaires by Age')
plt.xlabel('Age')
plt.ylabel('Net Worth')
plt.colorbar(label='Age')
plt.show()


# In[29]:


data.to_excel('CleanedFrobesbilloniare.xlsx', index=False)


# This report provides a comprehensive overview of the characteristics, wealth distribution, and economic impact of billionaires. The analysis sheds light on the industries driving billionaire wealth, the demographic profile of billionaires, and their contributions to economies. Visualizations enhance the presentation of insights, making it easier to grasp key trends and patterns. The report concludes with recommendations for promoting equitable growth and harnessing the potential of billionaire wealth for broader societal benefits.

# # THE END 

# In[ ]:




