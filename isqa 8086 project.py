import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

data1 = pd.read_csv("smmh.csv")
data2 = pd.read_csv("Students Social Media Addiction.csv")

#male is 1 and female is 2
data2["Gender"] = data2["Gender"].map({"Male": 1, "Female": 2})

#hs 1, ug 2, g 3
data2["Academic_Level"] = data2["Academic_Level"].map({"High School": 1, "Undergraduate": 2, "Graduate": 3})

#['Instagram'1, 'Twitter'2, 'TikTok'3, 'YouTube'4, 'Facebook'5, 'LinkedIn'6, 'Snapchat'7, 'LINE'8, 'KakaoTalk'9, 'VKontakte'10, 'WhatsApp'11, 'WeChat'12]
data2["Most_Used_Platform"] = data2["Most_Used_Platform"].map({'Instagram':1, 'Twitter':2, 'TikTok':3, 'YouTube':4, 'Facebook':5, 'LinkedIn':6, 'Snapchat':7,
                                                                    'LINE':8, 'KakaoTalk':9, 'VKontakte':10, 'WhatsApp':11, 'WeChat':12})

#no 0, yes 1
data2["Affects_Academic_Performance"] = data2["Affects_Academic_Performance"].map({"No": 0, "Yes":1})


#single 1, in relationship 2, complicated 3
data2["Relationship_Status"] = data2["Relationship_Status"].map({"Single":1, "In Relationship": 2, "Complicated": 3})

X = data2.drop(columns=["Mental_Health_Score", "Country"])
Y = data2["Mental_Health_Score"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))


X = sm.add_constant(X)
model1 = sm.OLS(Y, X).fit()
print(model1.summary())

import matplotlib.pyplot as plt

# get coefficients and sort
coef = model1.params.drop("const")
coef = coef.sort_values()

plt.figure(figsize=(8, 6))
plt.barh(coef.index, coef.values)
plt.title("Regression Coefficients (Impact on Mental Health Score)")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()