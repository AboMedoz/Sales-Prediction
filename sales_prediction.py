import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data
df = pd.read_csv('Dataset/Train.csv')

print(df.head(20))
df.info()

imputer = SimpleImputer(strategy='mean')
df['Item_Weight'] = imputer.fit_transform(df['Item_Weight'].values.reshape(-1, 1))
print(df['Item_Weight'].head(20))

df['Outlet_Age'] = 2025 - df['Outlet_Establishment_Year']

x = df[['Item_MRP', 'Item_Identifier', 'Item_Visibility', 'Item_Type', 'Item_Weight', 'Outlet_Type', 'Outlet_Age']].copy()
y = df['Item_Outlet_Sales']

# Factorization
x["Item_Identifier"], _ = pd.factorize(df["Item_Identifier"])

# One-Hot encoding
x = pd.get_dummies(x, columns=['Item_Type', 'Outlet_Type'], drop_first=True)

numerical_cols = ['Item_MRP', 'Item_Visibility', 'Item_Weight', 'Outlet_Age']
scaler = StandardScaler()
x[numerical_cols] = scaler.fit_transform(x[numerical_cols])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(random_state=42)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_predict)}")
print(f"R2 Score: {r2_score(y_test, y_predict) * 100:.2f}")


