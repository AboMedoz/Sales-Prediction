import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sales_prediction import model

df = pd.read_csv('Dataset/Test.csv')
print(df.head(5))
df.info()

imputer = SimpleImputer(strategy='mean')
df['Item_Weight'] = imputer.fit_transform(df['Item_Weight'].values.reshape(-1, 1))

df['Outlet_Age'] = 2025 - df['Outlet_Establishment_Year']

x = df[['Item_MRP', 'Item_Identifier', 'Item_Visibility', 'Item_Type', 'Item_Weight', 'Outlet_Type', 'Outlet_Age']].copy()

# Factorization
x["Item_Identifier"], _ = pd.factorize(x["Item_Identifier"])

# OneHot encoding
x = pd.get_dummies(x, columns=['Item_Type', 'Outlet_Type'], drop_first=True)

numerical_cols = ['Item_MRP', 'Item_Visibility', 'Item_Weight', 'Outlet_Age']
scaler = StandardScaler()
x[numerical_cols] = scaler.fit_transform(x[numerical_cols])

final_predict = model.predict(x)

output = pd.DataFrame({
    'Item_Identifier': df['Item_Identifier'],
    'Outlet_Identifier': df['Outlet_Identifier'],
    'Item_Sales': final_predict
})
output.to_csv('Dataset/Predictions.csv')