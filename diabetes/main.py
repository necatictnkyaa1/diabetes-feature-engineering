#############################################
# DİYABET VERİ SETİ İLE FEATURE ENGINEERING
#############################################

# Gerekli kütüphaneler
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

#############################################
# Veriyi Yükleme
#############################################

def load_diabetes():
    data = pd.read_csv(r"C:\Users\necat\Downloads\diabetes-prediction\diabetes\diabetes.csv")
    return data

df = load_diabetes()
print("İlk 5 gözlem:")
print(df.head())
print("\nVeri seti boyutu:", df.shape)
print("\nEksik değer sayıları (başlangıç):")
print(df.isnull().sum())

#############################################
# 0. Değişken İsimlerini Büyük Harfe Çevirme (opsiyonel)
#############################################
df.columns = [col.upper() for col in df.columns]
print("\nDeğişkenler:", df.columns.tolist())

#############################################
# 1. GENEL RESİM İÇİN FONKSİYONLAR (grab_col_names)
#############################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
print("\nNumerik değişkenler:", num_cols)
print("Kategorik değişkenler:", cat_cols)

# OUTCOME hedef değişken, onu numerik işlemlerden hariç tutacağız
num_cols = [col for col in num_cols if col != "OUTCOME"]

#############################################
# 2. AYKIRI DEĞER ANALİZİ (Outliers)
#############################################

# Aykırı eşik fonksiyonu
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı kontrol fonksiyonu
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Aykırı değerleri baskılama (re-assignment)
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

print("\n--- Aykırı Değer Kontrolü (düzeltme öncesi) ---")
for col in num_cols:
    print(f"{col}: {check_outlier(df, col)}")

# Aykırı değerleri baskıla
for col in num_cols:
    replace_with_thresholds(df, col)

print("\n--- Aykırı Değer Kontrolü (düzeltme sonrası) ---")
for col in num_cols:
    print(f"{col}: {check_outlier(df, col)}")

#############################################
# 3. EKSİK DEĞER ANALİZİ (Missing Values)
#############################################

# 0 değerlerini biyolojik olarak imkansız olduğu için NaN yap
zero_to_nan_cols = ['GLUCOSE', 'BLOODPRESSURE', 'SKINTHICKNESS', 'INSULIN', 'BMI']
for col in zero_to_nan_cols:
    df[col] = df[col].replace(0, np.nan)

# Eksik değer tablosu
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

print("\n--- Eksik Değer Oranları ---")
na_cols = missing_values_table(df, True)

# Eksik değerleri doldurma (medyan ile)
for col in zero_to_nan_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

print("\nEksik değerler medyan ile dolduruldu.")
print(df.isnull().sum())

#############################################
# 4. ÖZELLİK TÜRETME (Feature Extraction)
#############################################

# Yaş grupları
df['NEW_AGE_CAT'] = pd.cut(df['AGE'], bins=[0, 30, 40, 50, 120], labels=['Young', 'Mature', 'Middle', 'Senior'])

# BMI kategorileri
df['NEW_BMI_CAT'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# İnteraksiyon değişkenleri
df['NEW_GLUCOSE_BMI'] = df['GLUCOSE'] * df['BMI']
df['NEW_AGE_PREG'] = df['AGE'] * df['PREGNANCIES']
df['NEW_INSULIN_GLUCOSE'] = df['INSULIN'] / (df['GLUCOSE'] + 1e-5)  # sıfır bölmeyi engelle

# Hamilelik durumu (binary)
df['NEW_PREGNANT'] = (df['PREGNANCIES'] > 0).astype(int)

# Metabolik risk skoru (örnek)
df['NEW_METABOLIC_RISK'] = (df['GLUCOSE'] * df['BMI'] * df['DIABETESPEDIGREEFUNCTION']) / (df['AGE'] + 1)

print("\nYeni türetilen değişkenler eklendi.")
print(df.columns.tolist())

#############################################
# 5. ENCODING İŞLEMLERİ
#############################################

# Label Encoding (Binary değişkenler için)
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

# Binary kategorik değişkenleri bul (2 unique değerli)
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

print(f"\nBinary kategorik değişkenler: {binary_cols}")

for col in binary_cols:
    df = label_encoder(df, col)

# One-Hot Encoding (çok sınıflı kategorikler)
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# Kategorik değişkenleri tekrar yakala (encoding sonrası güncelle)
cat_cols, num_cols, cat_but_car = grab_col_names(df)
ohe_cols = [col for col in cat_cols if df[col].nunique() > 2 and df[col].nunique() <= 10]
print(f"One-Hot encoding uygulanacak değişkenler: {ohe_cols}")

df = one_hot_encoder(df, ohe_cols, drop_first=True)

print("Encoding sonrası veri boyutu:", df.shape)

#############################################
# 6. ÖLÇEKLENDİRME (StandardScaler)
#############################################

# Numerik değişkenleri güncelle
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in ["OUTCOME"]]

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nÖlçeklendirme tamamlandı. Örnek:")
print(df[num_cols].head())

#############################################
# 7. MODEL KURMA ve DEĞERLENDİRME
#############################################

y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17, stratify=y)

# Random Forest modeli
rf_model = RandomForestClassifier(random_state=46, n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
print(f"\n--- MODEL PERFORMANSI ---")
print(f"Random Forest Doğruluk (Accuracy): {accuracy:.4f}")

# Feature importance görselleştirme
def plot_importance(model, features, num=10, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('En Önemli 10 Değişken')
    plt.tight_layout()
    plt.show()

plot_importance(rf_model, X_train, num=10)

#############################################
# 8. (OPSİYONEL) Hiç işlem yapılmamış ham veri ile karşılaştırma
#############################################
print("\n--- Ham Veri ile Karşılaştırma ---")
df_raw = load_diabetes()
df_raw.columns = [col.upper() for col in df_raw.columns]
y_raw = df_raw["OUTCOME"]
X_raw = df_raw.drop("OUTCOME", axis=1)
X_raw = X_raw.replace(0, np.nan)
for col in ['GLUCOSE', 'BLOODPRESSURE', 'SKINTHICKNESS', 'INSULIN', 'BMI']:
    X_raw[col].fillna(X_raw[col].median(), inplace=True)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_raw, y_raw, test_size=0.20, random_state=17, stratify=y_raw)
rf_raw = RandomForestClassifier(random_state=46)
rf_raw.fit(X_train_r, y_train_r)
y_pred_r = rf_raw.predict(X_test_r)
acc_raw = accuracy_score(y_pred_r, y_test_r)
print(f"Ham veri ile RF doğruluk: {acc_raw:.4f}")
print(f"Feature engineering sonrası doğruluk: {accuracy:.4f}")
print(f"İyileşme: {accuracy - acc_raw:.4f}")