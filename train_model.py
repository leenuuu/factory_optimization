import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("data/Nassau Candy Distributor.csv")

print("✅ Original Rows:", len(df))

# =====================================================
# PRODUCT → FACTORY MAP
# =====================================================
factory_map = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory"
}

# Add Factory Column
df['Factory'] = df['Product Name'].map(factory_map)

# =====================================================
# DATE CLEANING
# =====================================================

# Convert dates safely
df['Order Date'] = pd.to_datetime(
    df['Order Date'],
    format='%d-%m-%Y',
    errors='coerce'
)

df['Ship Date'] = pd.to_datetime(
    df['Ship Date'],
    format='%d-%m-%Y',
    errors='coerce'
)

# =====================================================
# CREATE LEAD TIME
# =====================================================
df = df[
    (df['Lead_Time'] > 0) &
    (df['Lead_Time'] <= 30)
]

# =====================================================
# DATA CLEANING
# =====================================================

# Remove null values
df = df.dropna(subset=[
    'Lead_Time',
    'Product Name',
    'Factory',
    'Region',
    'Ship Mode'
])

# Keep realistic lead times
df = df[
    (df['Lead_Time'] > 0) &
    (df['Lead_Time'] <= 365)
]

print("✅ Remaining Rows:", len(df))

# =====================================================
# SAFETY CHECK
# =====================================================
if len(df) == 0:
    print("\n❌ ERROR: Dataset became empty.")
    print("Check your CSV date format.")
    exit()

# =====================================================
# PROFIT CALCULATION
# =====================================================
df['Profit'] = df['Sales'] - df['Cost']

# =====================================================
# ENCODING
# =====================================================
le_product = LabelEncoder()
le_factory = LabelEncoder()
le_region = LabelEncoder()
le_ship = LabelEncoder()

df['Product Name'] = le_product.fit_transform(df['Product Name'])
df['Factory'] = le_factory.fit_transform(df['Factory'])
df['Region'] = le_region.fit_transform(df['Region'])
df['Ship Mode'] = le_ship.fit_transform(df['Ship Mode'])

# =====================================================
# FEATURES
# =====================================================
X = df[['Product Name', 'Factory', 'Region', 'Ship Mode']]
y = df['Lead_Time']

# =====================================================
# TRAIN TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# MODEL
# =====================================================
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# =====================================================
# EVALUATION
# =====================================================
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds) ** 0.5
r2 = r2_score(y_test, preds)

print("\n📊 MODEL PERFORMANCE")
print("MAE  :", round(mae, 2))
print("RMSE :", round(rmse, 2))
print("R2   :", round(r2, 2))

# =====================================================
# SAVE MODEL + ENCODERS
# =====================================================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le_product, open("le_product.pkl", "wb"))
pickle.dump(le_factory, open("le_factory.pkl", "wb"))
pickle.dump(le_region, open("le_region.pkl", "wb"))
pickle.dump(le_ship, open("le_ship.pkl", "wb"))

print("\n✅ Model trained and saved successfully!")