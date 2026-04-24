
---

# 🧬 Diabetes Prediction: Uçtan Uca Feature Engineering

Bu proje, bir veri bilimcinin cephanesindeki en güçlü silah olan **Özellik Mühendisliği (Feature Engineering)** tekniklerini kullanarak, diyabet tahmin modelinin başarısını nasıl artırabileceğimizi kanıtlamak için geliştirilmiştir. 

Projenin en can alıcı noktası; ham veri ile işlenmiş veri arasındaki **%11'lik (veya senin çıktılarına göre değişen) başarı farkıdır.**

---

## 🛠️ Teknik Yaklaşım ve Fonksiyonlar

Proje sadece bir model kurma işlemi değil, yeniden kullanılabilir bir **analiz iskeleti** sunar:

### 🔍 Veri Analiz Fonksiyonları
* **`grab_col_names`**: Veri setindeki numerik, kategorik ve kardinal değişkenleri otomatik olarak ayrıştırır. Bu, büyük veri setlerinde manuel iş yükünü sıfıra indirir.
* **`missing_values_table`**: Eksik değerlerin sadece sayısını değil, oranını da çıkararak strateji belirlemeyi kolaylaştırır.

### 🧪 Veri Ön İşleme (Preprocessing)
* **Sıfır Değerleri Analizi:** Glikoz, İnsülin ve BMI gibi biyolojik olarak 0 olamayacak değişkenlerdeki hatalı veriler `NaN` ile değiştirilip medyan ile doldurulmuştur.
* **Aykırı Değer (Outlier) Yönetimi:** IQR yöntemiyle belirlenen eşik değerlere göre veriler baskılanmış (`replace_with_thresholds`), modelin gürültüden etkilenmesi önlenmiştir.

### 🏗️ Yeni Özellik Türetme (Feature Extraction)
Ham veriden daha yüksek bilgi çekmek için türetilen bazı kritik değişkenler:
* **NEW_METABOLIC_RISK**: Glikoz, BMI ve soya çekim fonksiyonunun yaşa oranlanmasıyla elde edilen risk skoru.
* **NEW_INSULIN_GLUCOSE**: İnsülinin glikoza oranı (metabolik verimlilik göstergesi).
* **Kategorik Dönüşümler**: Yaş ve BMI değerlerinin uzman görüşüne uygun segmentlere (`cut` metoduyla) ayrılması.




---

## 📊 Karşılaştırmalı Model Başarısı

Proje sonunda yapılan **Random Forest** testleri, veri mühendisliğinin gücünü şu tabloyla özetlemektedir:

| Metot | Accuracy Skoru |
| :--- | :---: |
| Ham Veri + Basit Doldurma | `~0.7247` |
| **Feature Engineering + Scaling** | **`~0.8312`** |
| **Net İyileşme** | **`+ 0.1065`** |

---

## 💻 Kullanım (Quick Start)

### Gereksinimler
```bash
pip install numpy pandas seaborn matplotlib scikit-learn missingno
```

### Çalıştırma
Kod içerisindeki `pd.read_csv` yolunun `diabetes.csv` dosyasının konumuyla eşleştiğinden emin olun ve dosyayı çalıştırın:
```bash
python main.py
```

---

## 📓 Çıktılar ve Görselleştirme
Kod otomatik olarak **Feature Importance** grafiğini oluşturur. Bu grafik, modelin hangi değişkeni (örneğin türettiğimiz interaksiyon değişkenlerini) ne kadar önemli bulduğunu gösterir.



---

## 👤 Yazar
**Muhammet Necati Çetinkaya**
* 🎓 Erciyes University - Software Engineering Student
* ⚡ AI Developer @ Miull

---
> **Not:** Bu çalışma Miuul Data Scientist Bootcamp projeleri kapsamında geliştirilmiştir.

---