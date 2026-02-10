# Stat-628

A collection of Stat 628 assignments

# Presentation

6 and a half minute presentation.

# Notes for coding

1. Tasks:
   - predict PRSM with multiple regression model with CIs
   - _[Executive summary]_ include factors drive substantial variation in PRSM (esp. a statistically discernible effect but not a practically relevant (significant) one)
   - _[Executive summary]_ introduce a baseline potential borrower by selecting values of each predictor in your final model
   - _[Executive summary]_ include the main drivers of PRSM and indicate which are associated with greater or lesser credit risk relative to the baseline
   - _[Technical report]_ include the removal of any outliers or suspect observations
   - _[Technical report]_ include all transformations and the construction of any new predictors
   - _[Technical report]_ include the procedure used to select the final model
   - _[Technical report]_ include diagnostics to assess the extent to which the usual multiple regression model assumptions hold
   - _[Presentation]_ include drivers for credit risk
   - _[Presentation]_ future development or improvement
2. Data
   - Use training dataset to fit the model (estimate model parameters, perform inference, and check relevant model diagnostics)
     - Split the training dataset into `train` and `dev`. `train` dataset for EDA, and fit the model; `dev` dataset for perform inference
   - Use the model to predict PRSM in evaluation set
3. Unit of each variable
   - "1 unit changes in some predictors may not be relevant; consider using more realistic or practically relevant changes"
   - normalization, standardization, etc.?
4. Keep track of any references used, and list them in the executive summary.

# Dataset

## Overall

1. There may be errors in some of its historical data. Pay particular attention to values outside the allowable range of certain variables.
2. Certain predictors affect PRSM in a non-linear fashion?
   - transforming some predictors or creating new predictors by squaring or cubing individual numerical predictors or taking ratios of existing ones?

## Response Variable

1. PRSM (y)
   - 2\*{amount repaid at 6 months}/{total amount owed}
   - Should >= 0.
   - Expected to be 1. > 1 indicates ahead of schedule, and <1 indicates behind.
   - “discretizing” numerical predictors?

## Possible Predictor Variable

2. FICO
   - Ranges 300\~850.
   - Poor (300\~579), fair(580\~669), good(670\~739), very good(740\~799), excellent(800~850).
   - Information contained in the FICO score may be relevant when dealing with certain borrowers but not others?
3. TotalAmtOwed
   - load + interest
4. Volume
   - Expected volume of credit card transactions per month
5. Stress
   - Ratio of the monthly garnishment to the expected volume of credit card transactions.
6. Num_Delinquent
   - Number of delinquent credit lines.
   - Delinquency occurs when a business is more than 30 days behind payment of a debt.
7. Num_CreditLines
   - Total number of credit lines, including both delinquent and nondelinquent lines.
8. WomanOwned
   - An indicator of whether the business is owned by a woman.
   - 1 if woman-owned and 0 otherwise.
   - woman-owned businesses more likely to pay off their loans on time?
9. CorpStructure
   - A categorical predictor, records whether the business is structured as a sole proprietorship, corporation, limited liability corporation (LLC), or a partnership
   - Corporations may be slower than other businesses at paying back their loans?
10. NAICS
    - 6-digit NAICS code. The North American Industry Classification System provides a 5- or 6-digit code that classifies different industries. For instance, the code for universities and colleges is 611310. You can look-up individual codes at this link.
    - https://www.census.gov/naics/ tells us that the first two digits indicates the industry of the biz.
11. Months
    - The number of months for which the business has been open.
    - Business open longer are more credit-worthy? After a certain point, an additional month of operation has a diminished predictive effect?
