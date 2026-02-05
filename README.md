# Stat-628

A collection of Stat 628 assignments

# Notes for coding

1. Objective: to predict PRSM with multiple regression model.
2. Use training dataset to fit the model (estimate model parameters, perform inference, and check relevant model diagnostics)
   - Split the dataset into `train` and `dev`
   - `train` dataset for EDA, and fit the model
   - `dev` dataset for perform inference
3. Use the model to predict PRSM in evaluation set

# Dataset

## Response Variable

1. PRSM (y)
   - 2\*{amount repaid at 6 months}/{total amount owed}
   - Expected to be 1. > 1 indicates ahead of schedule, and <1 indicates behind.

## Possible Predictor Variable

2. FICO
   - Ranges 300~850.
   - Poor (300~579), fair(580~669), good(670~739), very good(740~799), excellent(800~850).
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
9. CorpStructure

- A categorical predictor, which records whether the business is structured as a sole proprietorship, corporation, limited liability corporation (LLC), or a partnership.

10. NAICS
    - 6-digit NAICS code. The North American Industry Classification System provides a 5- or 6-digit code that classifies different industries. For instance, the code for universities and colleges is 611310. You can look-up individual codes at this link.
11. Months
    - The number of months for which the business has been open.
