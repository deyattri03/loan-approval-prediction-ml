# loan-approval-prediction-with-machine-learning-finding the-risk-a-simple-rule-misses











\### What does machine learning add over a simple credit rule?



Every day, lenders decide who gets a loan and who doesn't. A wrong "yes" can turn into a loss, and a wrong "no" sends a good customer somewhere else. This project predicts loan approval from an applicant's financial profile. Along the way, the most interesting question turned out to be not \*which model scores best\*, but \*what a model can do that a simple rule can't\*.



\---



\## The Problem



Given an applicant's income, credit score, debt-to-income (DTI) ratio, savings, collateral, loan details and background, can loan approval be predicted?



A useful answer has to do more than score well:



\- \*\*It has to be precise.\*\* Approving a risky applicant usually costs a lender more than turning away a safe one, so false approvals matter most.

\- \*\*It has to be fair.\*\* Personal attributes like gender should not decide who gets credit.

\- \*\*It has to be explainable.\*\* If someone is rejected, the lender needs to be able to say why.



These three requirements were kept in mind throughout the project, alongside accuracy.



\---



\## The Data



The dataset contains 1,000 applications with 20 columns, and every column has 5% missing values. The 50 applications with a missing outcome were removed, since filling in a missing outcome would mean making up the answer. This left 950 applications, about two-thirds of which were rejected.



<!-- Dataset source: add link here -->



\---



\## Approach



\*\*The data was explored before any modelling.\*\* The EDA turned out to be the most important part of the project. A plot of credit score against DTI ratio revealed a striking pattern: every approved application sits in one corner of the plot, where \*\*credit score is above 650 and DTI is 0.4 or less\*\*. Outside that corner, no application was approved.



\*\*A baseline was set first.\*\* Before any feature engineering, 8 models were trained on the raw data. Without a baseline, there is no way to tell whether feature engineering actually helps or just adds complexity.



\*\*The EDA finding was turned into features.\*\* Two flags, `Good\_Credit` and `Low\_DTI`, were created to capture the cut-offs directly. Financial ratios a credit officer would look at, such as loan-to-income, EMI-to-income and collateral-to-loan, were added, and the skewed money columns were log-transformed.



\*\*Models were compared using cross-validation, not a single test score.\*\* With only 190 applications in the test set, one lucky split can make a model look better than it is. Five-fold cross-validated F1 gives a more honest comparison.



\*\*The analysis went beyond the leaderboard.\*\* The best model was compared against a simple rule, checked for its reliance on gender, explained with SHAP, and given a decision threshold suited to a lender's priorities.



\---



\## Key Findings



\### 1. Two numbers explain most decisions



Credit score and DTI ratio are by far the strongest signals. Most other features, including co-applicant income, savings and age, add very little. In practice, the approval process in this data behaves like a checklist: \*\*both cut-offs have to be passed first, and only then do the other details matter.\*\*



\### 2. Feature engineering helps the models that can't see cut-offs



The new features made a big difference to Logistic Regression, KNN, SVM and Naive Bayes; KNN's CV F1 rose from 0.58 to 0.82. These models look for smooth, gradual relationships, so a sharp line at 650 is hard for them to learn. The two flags give it to them directly.



Tree-based models barely changed. A decision tree already works by asking yes/no questions like \*"Is credit score above 650?"\*, so it finds the cut-offs on its own.



\### 3. Why tree-based models win



The structure of this problem suits trees naturally:



\- \*\*Approval depends on thresholds.\*\* Trees split data at threshold values, which is exactly the pattern here.

\- \*\*Approval depends on a combination.\*\* A good credit score is not enough if DTI is too high. Trees capture this "A and B" logic by splitting on one feature and then the other.

\- \*\*Ensembles are more stable.\*\* A single decision tree can change a lot with small changes in the data. Random Forest and XGBoost combine many trees, which smooths out these swings.



\### 4. Why XGBoost was chosen as the final model



Random Forest (CV F1 0.947) and XGBoost (0.941) are so close that the difference falls within normal cross-validation noise. The choice between them was therefore based on what matters to a lender:



\- \*\*XGBoost ranks applicants best.\*\* It has the highest ROC AUC (0.989), meaning it is the best at ordering applicants from safest to riskiest. This matters because a lender is not tied to a 0.5 cut-off and can move the threshold to match its risk appetite.

\- \*\*XGBoost is the best fair model.\*\* Once gender is removed, its CV F1 rises to 0.951, the highest score in the whole project.

\- \*\*Boosting focuses on the hard cases.\*\* XGBoost builds trees one after another, and each new tree concentrates on the applications the previous ones got wrong. In this data, the hard cases are applicants who pass both cut-offs but are still rejected, which is exactly where a model needs to do its work.



\### 5. What ML adds over a simple rule



Approving everyone with a credit score above 650 and a DTI of 0.4 or less already gives an F1 of 0.876. This rule never misses an approved applicant, but about 1 in 5 of the people it approves were actually rejected.



XGBoost raises F1 to 0.928, and almost all of the gain comes from \*\*precision\*\* (0.78 → 0.89). In other words, the real contribution of ML is \*\*sorting out the risky applicants among those who look fine on paper\*\*, which says far more than a high accuracy score.



\### 6. Why high recall alone is misleading



SVM and Naive Bayes reached a perfect recall of 1.0, which looks impressive at first. But this comes from approving almost everyone near the line, which gives them the lowest precision of all the models. For a lender, a model that approves too easily is a liability, not a strength.



\### 7. The model doesn't need gender



Removing gender did not hurt performance; the score of XGBoost actually improved slightly. This is the best possible outcome of a fairness check: the model becomes fairer with no trade-off at all.



\### 8. A higher threshold suits a lender better



By default, an application is approved when its predicted probability is above 0.5. Raising this threshold to 0.7 increased precision from 0.89 to 0.95, while recall dropped only from 0.967 to 0.950. \*\*More risky applicants are rejected, and almost no good ones are lost.\*\*



\### 9. The model's decisions make sense



SHAP shows that the model relies on the same factors a credit officer would. A high credit score pushes towards approval, a high DTI pushes strongly towards rejection, and higher income and a lower loan-to-income ratio help. Gender has almost no influence. A model whose reasoning matches common sense is much easier to trust, and much easier to explain to a rejected applicant.



\---



\## Conclusion



The recommended approach for this problem is \*\*XGBoost without the gender feature, with a decision threshold of 0.7\*\*.



The bigger lesson is that the most useful result did not come from the leaderboard. A simple two-condition rule already handles most decisions. Machine learning earns its place by catching the risky applicants who pass that rule, and by doing so fairly and in a way that can be explained.



\---



\## Limitations



\- \*\*The data appears synthetic.\*\* Real lending data rarely follows such sharp cut-offs, so real-world performance would likely be lower.

\- \*\*The test set is small\*\* (190 applications), so small differences between models should not be over-interpreted.

\- \*\*There is mild leakage.\*\* The 650 and 0.4 cut-offs were chosen by looking at the full dataset before it was split into training and test sets.

\- \*\*Mean imputation is basic.\*\* A more careful method could handle the missing values better.





