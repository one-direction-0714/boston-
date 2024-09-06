import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

data_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
data_boston = pd.read_csv(data_url)

#data_boston.describe()
print(data_boston.head(10))

# Mission 1: MEDV-建立相关的Boxplot
plt.boxplot(x=data_boston['MEDV'])
plt.title('Owners-occupied homes')
plt.show()


# Mission 2: Charles River Variable-建立一个柱状图
chas_counts = data_boston['CHAS'].value_counts()  # 计算每个唯一值的数量
plt.bar(x=chas_counts.index, height=chas_counts.values)
plt.title('number of homes near the Charles River')
plt.show()

#Mission 3: boxplot for the MEDV variable vs the AGE variable
data_boston.loc[(data_boston['AGE'] <= 35), 'Age_Group'] = '35 years and younger'
data_boston.loc[(data_boston['AGE'] > 35) & (data_boston['AGE'] < 70), 'Age_Group'] = 'between 35 and 70 years'
data_boston.loc[(data_boston['AGE'] >= 70), 'Age_Group'] = '70 years and older'
plt.figure(figsize=(15, 8))  # 设置画布大小为宽15英寸，高8英寸
ax3 = sns.boxplot(x = 'MEDV', y = 'Age_Group', data = data_boston)
plt.title('Median value of owner-occupied homes per Age group')
plt.show()

#Mission 4: Scatter plot shows the relationship between Nitric oxide
# concentrations and the proportion  of non-retail business acres per town
plt.figure(figsize=(12, 8))  # 设置画布大小为宽10英寸，高6英寸
ax4 = sns.scatterplot(y = 'NOX', x = 'INDUS', data = data_boston)
ax4.set_title('Nitric oxide concentration per proportion of non-retail business acres per town')
plt.show()

#Mission 5: histogram for the pupil to teacher ratio variable
ax5 = sns.countplot(x = 'PTRATIO', data = data_boston)
ax5.set_title('Pupil to teacher ratio per town')
plt.show()

#Mission 6: is there a significant difference in median value of
# houses bounded by the Charles River or not?(T-test)
#null hypothesis: there is no significant difference in median value...
#altertive hypothesis：there is significant difference in median value...
data_boston.loc[(data_boston['CHAS'] == 0), 'CHAS_T'] = 'FAR'
data_boston.loc[(data_boston['CHAS'] == 1), 'CHAS_T'] = 'NEAR'
data_boston.head(5)

tt = stats.ttest_ind(data_boston[data_boston['CHAS_T'] == 'FAR']['MEDV'],
                      data_boston[data_boston['CHAS_T'] == 'NEAR']['MEDV'], equal_var = True)
print(tt.pvalue)

#conclusion: we found that the p-value is less than 0.05, so we reject the null hypothesis,
# meaning that there is no difference in median value between houses near the Charles River and houses far away.

# Mission 7: is there a difference in Median values of houses for each proportion
# of owner occupied units built prior to 1940(AGE) (using ANOVA)
#null hypothesis:there is no difference in MEDV for each proportion of owner occupied units built prior to 1940
#alternative hypothesis: there is difference...
from statsmodels.formula.api import ols
lm = ols('MEDV ~ AGE', data = data_boston).fit()
table = sm.stats.anova_lm(lm)
print(table)
#conclusion: since the p-value is less than 0.05, we can reject the null hypothesis.

#Mission 8: there is any relationship between Nitric oxide concentrations and proportion
#of non-retail business acres per town? (using Pearson Correlation)
# null hypothesis: Nitric Oxide concentration is not correlated with the proportion...
# alternative hypothesis: Nitric Oxide concentration is correlated with the ...
a = data_boston['NOX']
b = data_boston['INDUS']
print(stats.pearsonr(a,b))
# conclusion: we can find the p-value is less than 0.05, so we reject the null hypothesis.

#Mission 9:What is the impact of an additional weighted distance to the five Boston employment
# centres on the median value of owner occupied homes? (Regression analysis)
x = data_boston['DIS']
y = data_boston['MEDV']
x2 = sm.add_constant(x)

#模型回归评估
est = sm.OLS(y, x2).fit()
print(est.summary())

#尝试一下另一种方法
x = data_boston['DIS']
y = data_boston['MEDV']
x2 = sm.add_constant(x)

model = sm.OLS(y, x2).fit()

print(model.summary())