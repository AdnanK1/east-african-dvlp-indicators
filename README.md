# EASTERN AFRICA DEVELOPMENT INDICATORS

<img src="https://i.pinimg.com/564x/47/8b/dc/478bdc1de418ac0242d54eb00ece77fc.jpg" alt="FORECAST" width="950" height="400">

## GROUP MEMBERS
- Adnan Gitonga

- Anita Makori

- Christine Kibandi

- Dennis Wainaina

- Peris Wamoni

- Valary Thairu

The README consists of the following content:
1. [BUSINESS UNDERSTANDING](#business-understanding)
2. [DATA UNDERSTANDING](#data-understanding)
3. [DATA PREPARATION](#data-preparation)
4. [EXPLORATORY DATA ANALYSIS](#exploratory-data-analysis)
5. [MODELING](#modeling)
6. [FORECASTING](#forecasting)
7. [DEPLOYMENT](#deployment)
8. [CONCLUSIONS](#conclusions)
9. [RECOMMENDATIONS](#recommendations)

# Business Understanding
## 1.1. Problem Statement
We aim to create a predictive model to forecast percentage changes in future healthcare expenditure trends in Eastern Africa, using historical data on healthcare financing and expenditure.


## 1.2. Objectives
### 1.2.1 Main Objective
To analyze healthcare financing and expenditure trends in Eastern Africa to construct a model predicting future healthcare expenditure percentage changes

### 1.2.2 Specific Objectives
- To explore World Bank data, identifying relevant health system indicators for Eastern Africa.
  
- To analyze healthcare expenditure, distinguishing between public and private spending.
  
- To discover correlations between healthcare financing indicators to reveal patterns or relationships.
  
- To offer recommendations for governments, suggesting areas of improvement, potential collaborations, and strategies to enhance healthcare financing.



## 1.3. Success Metric
- To achieve a low root mean square error (RMSE) to ensure accurate forecasting.


# Data Understanding
### 2.1. Data Source
- The dataset is sourced from the World Bank's World Development Indicators (WDI) platform, which provides comprehensive development data covering various topics.

- The platform covers a wide range of topics, including poverty, prosperity, consumption, income distribution, population dynamics, education, labor, health, gender, agriculture, climate change, energy, biodiversity, water, sanitation, growth, economic structure, income and savings, trade, labor productivity, business, stock markets, military, communications, transport, technology, debt, trade, aid dependency, refugee, tourism, and migration.


### 2.2 Properties of the Data
The data is structured and tabulated, providing a clear representation of various development indicators across different countries and years. Each entry in the dataset corresponds to a specific country, year, and development indicator, accompanied by its respective value and other relevant metadata.


### 2.3.Data Limitations
- While the WDI dataset is extensive, it is essential to note that there might be gaps in data for certain countries or years. Additionally, the data might be subject to revisions as new information becomes available or as methodologies are refined. It's also worth noting that the dataset might not capture all nuances or local variations within countries.
- The dataset contains several NaN (Not a Number) entries, indicating missing data for certain years or indicators. This could be due to the unavailability of data for those specific years or the indicator not being relevant for that particular region or country at that time. It's essential to handle these missing values appropriately during analysis.
- Eritrea has constant values for ‘Domestic general government expenditure (% of general government expenditure)’. Meaning  Dickey fuller test is ineffective. We ended up differencing it without knowing if it's stationary or not because the values are constant.



# Data Preparation
### 3.1  Data Loading and Initial Examination
The main dataset, WDIData.csv, was loaded into a DataFrame named wdi_df. The WDISeries.csv file, potentially providing metadata about the main dataset, was also loaded into a DataFrame called series_df. Upon an initial examination, it was observed that there was a column named 'Unnamed: 67', which seemed extraneous. This column was dropped to tidy up the data.

### 3.2 Data Transformation
Given the structure of the data, it was transformed from a wide format to a long format to facilitate more straightforward analysis. The pd.melt() function was employed for this task.


### 3.3 Region Specific Data Filtering
Our focus is on Eastern African countries, so the data was further filtered to include only the relevant countries.



# Exploratory Data Analysis
### 4.1 Targeting Healthcare Expenditure Data
Given the core concern around healthcare financing in Eastern Africa, the data was further refined to zero in on indicators related to healthcare expenditure. A list comprehension was employed to select those indicators containing the term 'expenditure'.

### 4.2 Refining the Time Span
In order to focus the analysis on recent data, the dataset was filtered to include only the years from 2010 to 2020.

### 4.3 Merging with Metadata
To augment the healthcare expenditure data with additional contextual information, expenditure_df was merged with the series_df data frame which possibly contains metadata.
Post merging, extraneous columns were pruned to retain only the columns crucial to the analysis.

### 4.4 Concentrating on Key Indicators
Seven critical healthcare financing indicators were identified for deeper analysis. The dataset was filtered to include only these indicators

### 4.5 Time Series Adjustments and Renaming Columns
To facilitate time series analysis, the 'Years' column was converted into a datetime type and set as the dataframe's index.

### 4.6 Data Segmentation
-To simplify and optimize our analysis, we created two functions:
create_dfs: This function segregates the main dataframe based on unique indicators. The outcome is a dictionary where each key is an indicator, and the corresponding value is a dataframe containing data solely for that indicator.
create_country_dfs: This function further divides the data frames obtained from the create_dfs function by country. This nested segmentation aids in conducting a country-specific analysis of each indicator. The Percentage Diff column was added to these country-specific data frames to capture the year-on-year percentage change in the given indicator.

### 4.7 Visualization
-Given the segmented data, we sought to visualize the trends for each indicator across different Eastern African countries. The plot function was designed to represent a country's data trend over time for a specific indicator. For each indicator of interest, it creates a figure with multiple subplots—each subplot corresponding to a different country.

The resulting plots furnish insights into how each indicator has evolved over the years for all Eastern African countries under consideration.


# Modeling
## 5.1. PREREQUISITES FOR MODELING
## Stationarity Testing 
-The augmented Dickey-Fuller (ADF) test is a well-regarded technique to determine the stationarity of a time series dataset. Stationarity is a fundamental property for many time series models, and understanding whether data is stationary can guide model selection and further data preparation steps. We created an adf_test_indicator function that does the following:
- Country Filtering: The function begins by isolating data for a specified country.
- Iterative Testing: For every unique indicator related to the country, it does the following:
Before applying the ADF test, the function checks if the variance of the time series data for the indicator is zero. This is a prudent step since a time series with constant values will obviously be stationary, and there's no need to apply the ADF test to such series.

- The ADF test is executed on the 'Percentage' data for that indicator.
The results from the ADF test, including the ADF statistic, the p-value, and the critical values, are stored in a dictionary. A dictionary containing ADF results for all indicators associated with the specified country is returned. In essence, this function provides a concise summary of the stationarity status of all indicators for a specified country.

- A code that effectively aggregates the stationarity testing across multiple countries was written . It outlines the list of countries in Eastern Africa for which you want to determine the stationarity of health indicators.For each country in the list, the adf_test_indicator function is called, which performs the Augmented Dickey-Fuller test for each health indicator within the country. The results are then stored in the results dictionary, where each key is a country and the associated value is a list of tuple pairs containing indicators and their ADF test results.

- For each indicator within a country, it checks the ADF test statistics against the critical value and the p-value to infer stationarity.It then prints whether the indicator is "Stationary" or "Not Stationary".
## Splitting data 
- The data was differenced then split into a train and testing sets based on a given date (2018). This means data up to 2018 is used for training, and the data from 2018 onward is used for testing. 
- The split_data function is defined to split the health indicators data by country for both the testing and training sets using the helper function create_country_dfs.

### 5.2.ARIMA Model
For each health indicator in the training data, an ARIMA model was fitted on the 'Percentage Difference' of that indicator. Forecasts are then made for both the training and testing data.
The Root Mean Squared Error (RMSE) for the predictions on both the training and testing data is computed. These RMSE values give a measure of how well the model predictions match the actual observed values.
The RMSE values are stored in train_dict and test_dict dictionaries, respectively, for each indicator.


## 5.2.AR Model
-  In this section, we implemented Autoregressive models which are a class of statistical models. Used to analyze and predict time series data, where observations are dependent on past observations.The choice of the order "p" and the estimation of coefficients are critical steps in using AR models for forecasting and understanding time-dependent data.
  


## 5.3. Simple Exponential Smoothing (SES) Model
- Simple Exponential Smoothing (SES) is a time series forecasting method used to make short-term predictions for data with no significant trend or seasonality. It's a basic but effective technique, particularly when dealing with data that exhibits random fluctuations over time. 



## 5.4.LSTM
-It is a type of recurrent neural network (RNN) that is well-suited for processing sequential data such as time series. LSTM networks have an internal memory state that allows them to remember information over long periods of time and selectively forget irrelevant information, making them particularly useful for modeling complex temporal relationships in data.



#  Forecasting
- At this point, we aimed to forecast future values for each time series in the dataset using the best model selected for each series to fit the model. We fitted the models and forecasted the future values for the next 12 months.


# Deployment


# Conclusions

# Recommendations










