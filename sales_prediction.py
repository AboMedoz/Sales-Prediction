import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

# https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data
df = pd.read_csv('Dataset/Train.csv')

print(df.head(20))
df.info()

imputer = SimpleImputer(strategy='mean')
df['Item_Weight'] = imputer.fit_transform(df['Item_Weight'].values.reshape(-1, 1))
print(df['Item_Weight'].head(20))

df['Outlet_Age'] = 2025 - df['Outlet_Establishment_Year']

x = df[['Item_MRP', 'Item_Visibility', 'Item_Type', 'Item_Weight', 'Outlet_Type', 'Outlet_Age']]
y = df['Item_Outlet_Sales']

# OneHot encoding
x = pd.get_dummies(x, columns=['Item_Type', 'Outlet_Type'], drop_first=True)

numerical_cols = ['Item_MRP', 'Item_Visibility', 'Item_Weight', 'Outlet_Age']
scaler = StandardScaler()
x[numerical_cols] = scaler.fit_transform(x[numerical_cols])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.8, 1.0],
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=1)
grid_search.fit(x_train, y_train)
print(f"Optimal Grid Search Parameters: {grid_search.best_params_}")
print(f"Optimal R2 Score: {grid_search.best_score_:.2f}")

optimal_params = grid_search.best_params_
optimized_model = GradientBoostingRegressor(**optimal_params, random_state=42)
optimized_model.fit(x_train, y_train)

y_predict = optimized_model.predict(x_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_predict)}")
print(f"R2 Score: {r2_score(y_test, y_predict) * 100:.2f}")


