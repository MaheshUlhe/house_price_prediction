House Price Prediction
1.Importing Libraries

[ ]
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import seaborn as sns
import matplotlib.pyplot as plt
Load Dataset

[ ]
df = pd.read_csv('Housing.csv')
df


[ ]
df.head()


[ ]
df.shape
(545, 13)

[ ]
df.describe()


[ ]
df.columns
Index(['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
       'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
       'parking', 'prefarea', 'furnishingstatus'],
      dtype='object')

[ ]
df["price"].unique()
array([13300000, 12250000, 12215000, 11410000, 10850000, 10150000,
        9870000,  9800000,  9681000,  9310000,  9240000,  9100000,
        8960000,  8890000,  8855000,  8750000,  8680000,  8645000,
        8575000,  8540000,  8463000,  8400000,  8295000,  8190000,
        8120000,  8080940,  8043000,  7980000,  7962500,  7910000,
        7875000,  7840000,  7700000,  7560000,  7525000,  7490000,
        7455000,  7420000,  7350000,  7343000,  7245000,  7210000,
        7140000,  7070000,  7035000,  7000000,  6930000,  6895000,
        6860000,  6790000,  6755000,  6720000,  6685000,  6650000,
        6629000,  6615000,  6580000,  6510000,  6475000,  6440000,
        6419000,  6405000,  6300000,  6293000,  6265000,  6230000,
        6195000,  6160000,  6125000,  6107500,  6090000,  6083000,
        6020000,  5950000,  5943000,  5880000,  5873000,  5866000,
        5810000,  5803000,  5775000,  5740000,  5652500,  5600000,
        5565000,  5530000,  5523000,  5495000,  5460000,  5425000,
        5390000,  5383000,  5320000,  5285000,  5250000,  5243000,
        5229000,  5215000,  5145000,  5110000,  5075000,  5040000,
        5033000,  5005000,  4970000,  4956000,  4935000,  4907000,
        4900000,  4893000,  4865000,  4830000,  4795000,  4767000,
        4760000,  4753000,  4690000,  4655000,  4620000,  4613000,
        4585000,  4550000,  4543000,  4515000,  4480000,  4473000,
        4445000,  4410000,  4403000,  4382000,  4375000,  4340000,
        4319000,  4305000,  4277000,  4270000,  4235000,  4200000,
        4193000,  4165000,  4130000,  4123000,  4098500,  4095000,
        4060000,  4025000,  4007500,  3990000,  3920000,  3885000,
        3850000,  3836000,  3815000,  3780000,  3773000,  3745000,
        3710000,  3703000,  3675000,  3640000,  3633000,  3605000,
        3570000,  3535000,  3500000,  3493000,  3465000,  3430000,
        3423000,  3395000,  3360000,  3353000,  3332000,  3325000,
        3290000,  3255000,  3234000,  3220000,  3150000,  3143000,
        3129000,  3118850,  3115000,  3087000,  3080000,  3045000,
        3010000,  3003000,  2975000,  2961000,  2940000,  2870000,
        2852500,  2835000,  2800000,  2730000,  2695000,  2660000,
        2653000,  2604000,  2590000,  2520000,  2485000,  2450000,
        2408000,  2380000,  2345000,  2310000,  2275000,  2240000,
        2233000,  2135000,  2100000,  1960000,  1890000,  1855000,
        1820000,  1767150,  1750000], dtype=int64)

[ ]
df["price"].unique().sum()
1150235940
Check for infinite values and replace them with NaN

[ ]
df.replace([np.inf, -np.inf], np.nan, inplace=True)
Drop rows with NaN values

[ ]
df.dropna(subset=['price'], inplace=True)

[ ]
sns.set_style("whitegrid")
Scatter Plot of Price vs Area

[ ]
plt.figure(figsize=(10, 6))
sns.scatterplot(x='area', y='price', data=df, hue='bedrooms', palette='viridis')
plt.title('Price vs Area')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.legend(title='Bedrooms')
plt.show()

Distribution Plot of Price

[ ]
plt.figure(figsize=(12, 8))
sns.histplot(df['price'], bins=30, kde=True, color='blue', edgecolor='black')
plt.title('Distribution of House Prices', fontsize=20, weight='bold')
plt.xlabel('Price', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Adding a grid
plt.grid(True, linestyle='--', alpha=0.7)

# Display the plot
plt.show()

Count Plot of Bedrooms

[ ]
plt.figure(figsize=(10, 6))
sns.countplot(x='bedrooms', data=df, palette='viridis')
plt.title('Number of Bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
plt.show()

Count Plot of Bathrooms

[ ]
plt.figure(figsize=(10, 6))
sns.countplot(x='bathrooms', data=df, palette='viridis')
plt.title('Number of Bathrooms')
plt.xlabel('Bathrooms')
plt.ylabel('Count')
plt.show()

Selecting relevant columns

[ ]
df = df[['price', 'area', 'bedrooms', 'bathrooms']]
Creating Independent and Dependent Variables

[ ]
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

[ ]
X.shape
(545, 3)
Splitting the dataset into training and testing sets

[ ]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Initializing the linear regression model

[ ]
model = LinearRegression()
Training the model

[ ]
model.fit(X_train, y_train)


[ ]
# Making predictions

y_pred = model.predict(X_test)
y_pred
array([6383168.05981192, 6230250.35140428, 3597885.28090091,
       4289730.8386382 , 3930445.60575177, 4883933.33918115,
       5505773.16533075, 6068793.48103629, 3321512.02483442,
       3495157.72744705, 8731338.12527607, 3561265.8244721 ,
       3487335.97847431, 3633344.35548029, 3933900.2714526 ,
       6661080.95290716, 2893133.47793136, 4635197.40872131,
       4583377.42320885, 4274999.75826381, 4296640.17003986,
       4920207.32903988, 3462807.85199841, 3875170.95453847,
       5412497.19140831, 7425564.68389184, 3269692.03932195,
       5021117.35611339, 7122556.71331971, 3238600.04801447,
       5366583.9261965 , 3640253.68688195, 6057517.06636272,
       4847659.34932243, 4572198.51082701, 5573863.86824637,
       4167435.67282878, 4021179.33154444, 3764621.65211187,
       5307730.62714319, 5319221.21748587, 3456243.98716683,
       6202613.02579763, 4013357.58257171, 4534099.68582614,
       4235271.10270425, 6057517.06636272, 4525560.52386579,
       4983206.22693419, 3238600.04801447, 6567080.25723531,
       3238600.04801447, 4886475.58731091, 4116997.55359665,
       4180093.95378267, 3567705.7071645 , 5971150.42384195,
       3391517.75642211, 5150855.01571622, 3514973.30408096,
       4723103.68827085, 4722288.7729915 , 4427005.04910038,
       3915714.52537737, 4414098.80386812, 3861352.29173514,
       5904599.35795509, 3598797.69847198, 5824514.82737829,
       4531557.43769638, 4969297.37060092, 4759565.37395123,
       5104119.52646327, 7612116.63173672, 3129432.61186821,
       5816602.88487562, 3816441.63762434, 3865621.87271532,
       4635197.40872131, 4393370.80966313, 6542082.68205015,
       3971901.59416174, 5884783.78132117, 4866660.010677  ,
       3149691.15736393, 7466018.06120081, 3529704.38445536,
       3754257.65500938, 6908277.24633825, 7840034.37446164,
       3940907.10514598, 5343403.87739169, 4084993.1447181 ,
       3740438.99220605, 9091535.77573357, 4200821.94798766,
       4935974.80912452, 5988423.7523461 , 4462464.12367976,
       6624994.6588701 , 3681709.67529193, 5573863.86824637,
       3588433.70136949, 6547076.98477974, 4995395.05917882,
       5316678.96935611, 6417714.71682023, 6057517.06636272,
       6092063.72337104])
Evaluating the model

[ ]
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

[ ]
print(f"Mean Squared Error: {mse}")
print(f"R-Squared: {r2}")
Mean Squared Error: 2750040479309.0513
R-Squared: 0.45592991188724474
Visualizing the results

[ ]
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices (Linear Regression)')
plt.show()

