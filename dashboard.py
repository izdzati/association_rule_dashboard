import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Fungsi untuk menampilkan pengertian dan info tentang Association Rule Mining
def display_info():
    st.title("Fuzzy Association Rule Mining")
    st.write("""
        ### Fuzzy Association Rule
        **Fuzzy Association Rule** merupakan pengembangan dari metode association rule yang hanya menemukan rule yang berasal dari data binary. 
        Fuzzy association rule memungkinkan menemukan aturan asosiasi / rules menggunakan konsep fuzzy sehingga atribut yang bernilai kuantitatif dapat ditangani. 
        Dalam fuzzy association rule data yang bernilai kuantitatif seperti harga bahan pokok diubah dalam bentuk fuzzy. 
        
        Proses fuzzy yang dapat dilakukan seperti mengubah harga bahan pokok menjadi persentase perubahan harga 
        yang lalu masuk dalam himpunan fuzzy dengan atribut linguistik seperti “naik”, “turun”, “stabil”. Kemudian hasil fuzzifikasi tersebut dilanjutkan dengan analisis association rule mining untuk memperoleh rules pergerakan harga yang terjadi secara bersamaan sebagai contoh diperoleh fuzzy association rule adalah {Cabai Merah Keriting=Naik, Cabai Merah Besar=Stabil}→{Bawang Merah=Naik}.
        Rule ini menunjukkan bahwa jika Cabai Merah Keriting harganya naik dan Cabai Merah Besar harganya stabil maka harga Bawang Merah akan naik

        ### Association Rule Mining
        **Association Rule Mining** atau aturan asosiasi merupakan teknik data mining yang digunakan untuk mencari hubungan asosiasi antar data. 
        Teknik association rule umumnya adalah pernyataan if/then atau jika/maka yang membantu dalam menemukan hubungan antara data yang tampaknya tidak terkait dalam basis data relasional atau tempat penyimpanan informasi lainnya. 
        Teknik association rule membantu dalam menampilkan kombinasi dari data yang sering muncul dan menemukan hubungan asosiasi yang ada di dalamnya. 
        Adapun parameter dalam Association rule yaitu sebagai berikut.
        
        1. **Support**  
           Support merupakan persentase kombinasi item (itemset) dalam database yang dinyatakan dengan rumus sebagai berikut:
           - $\\text{support} (A) = P(A) = \\frac{n(A)}{n(S)}$
           - $\\text{support} (A, B) = P(A \\cap B) = \\frac{n(A \\cap B)}{n(S)}$

           Keterangan:
           - $P(A)$ : Peluang itemset A
           - $P(A \\cap B)$ : Peluang itemset A dan B muncul bersamaan
           - $n(A)$ : Jumlah kejadian yang mengandung itemset A
           - $n(A \\cap B)$ : Jumlah kejadian yang mengandung itemset A dan B
           - $n(S)$ : Total kejadian
           - A : itemset A
           - B : itemset B

        2. **Confidence**  
           Confidence merupakan ukuran yang menunjukkan hubungan antar dua item secara conditional. Confidence dinyatakan dengan rumus sebagai berikut:
           - $\\text{confidence} (A \\rightarrow B) = P(B | A) = \\frac{P(A \\cap B)}{P(A)}$

           Keterangan:
           - $P(B | A)$ : Peluang itemset B muncul setelah itemset A muncul
           - $P(A)$ : Peluang itemset A
           - $P(A \\cap B)$ : Peluang itemset A dan B muncul bersamaan
           - A : itemset A
           - B : itemset B

        3. **Lift**  
           Lift merupakan suatu nilai yang mengukur seberapa penting rule yang telah terbentuk berdasarkan nilai support dan confidence. Lift dinyatakan dengan rumus sebagai berikut:
           - $\\text{lift}(A \\rightarrow B) = \\frac{\\text{confidence}(A \\rightarrow B)}{\\text{Support} (B)} = \\frac{\\frac{P(A \\cap B)}{P(A)}}{P(B)}$

           Keterangan:
           - $P(A \\cap B)$ : Peluang itemset A dan B muncul bersamaan
           - $P(A)$ : Peluang itemset A
           - $P(B)$ : Peluang itemset B
           - A : itemset A
           - B : itemset B

           Adapun ketentuan nilai lift adalah sebagai berikut:
           - Jika $\\text{Lift}(A, B) < 1$, kejadian A berkorelasi negatif dengan kejadian B atau hanya menggunakan item B
           - Jika $\\text{Lift}(A, B) > 1$, kejadian A berkorelasi positif dengan kejadian B atau menggunakan item A dan B secara bersama-sama
           - Jika $\\text{Lift}(A, B) = 1$, tidak ada korelasi antara kejadian A dan B

        ### Algoritma FP-Growth
        **Algoritma FP-Growth** merupakan pengembangan dari algoritma apriori sehingga kekurangan dari algoritma apriori diperbaiki oleh algoritma FP-Growth. 
        Pada algoritma apriori diperlukan generate candidate untuk mendapatkan frequent itemset, tetapi pada algoritma FP-Growth generate candidate tidak dilakukan karena telah menggunakan konsep pembangunan tree dalam pencarian frequent itemsets yang disebut dengan FP-Tree. 
        Dengan menggunakan FP-Tree, frequent itemset dapat langsung diekstrak.

        Adapun tahapan dalam algoritma FP-Growth adalah sebagai berikut:
        
        1. **Tahapan pembangkitan Conditional Pattern Base**  
           Conditional Pattern Base merupakan sub database yang berisi prefix path (lintasan prefix) dan suffix pattern (pola akhiran). 
           Pembangkitan conditional pattern base didapatkan melalui FP-Tree yang telah dibangun sebelumnya. 
           FP-Tree diperoleh dari menghitung frekuensi pergerakan harga bahan pokok yang terjadi, kemudian diurutkan dari terbesar hingga terkecil. 
           Setelah diurutkan maka FP-Tree dapat dibentuk.

        2. **Tahap pembangkitan Conditional FP-Tree**  
           Pada tahap ini, support count dari setiap pergerakan harga bahan pokok pada setiap conditional pattern base dijumlahkan. 
           Lalu, setiap pergerakan harga bahan pokok yang memiliki jumlah support count lebih besar atau sama dengan minimum support count akan dibangkitkan dengan conditional FP-Tree.

        3. **Tahap pencarian Frequent Pattern**  
           Apabila conditional FP-Tree merupakan lintasan tunggal (single path), maka didapatkan frequent pattern dengan melakukan kombinasi pergerakan harga bahan pokok untuk setiap conditional FP-Tree. 
           Apabila conditional FP-Tree bukan lintasan tunggal, maka dilakukan pembangkitan FP-Growth secara rekursif.
    """)

# Implementasi fungsi berdasarkan code pertama
def calculate_percentage_change(data):
    percentage_change = data.pct_change().fillna(0) * 100
    classification = percentage_change.applymap(lambda x: "increase" if x > 0 else ("decrease" if x < 0 else "stable"))
    return percentage_change.abs(), classification

def fuzzify(data_abs):
    fuzzy_data = pd.DataFrame(index=data_abs.index)
    for column in data_abs.columns:
        fuzzy_data[column + '_L'] = data_abs[column].apply(lambda x: 
            1 if x <= 25 else 
            (0.33 - (x - 25) / (33 - 25) if 25 < x <= 33 else 0)
        )
        fuzzy_data[column + '_M'] = data_abs[column].apply(lambda x: 
            (x - 33) / (50 - 33) if 33 < x <= 50 else 
            (0.66 - (x - 50) / (66 - 50) if 50 < x <= 66 else 0)
        )
        fuzzy_data[column + '_H'] = data_abs[column].apply(lambda x: 
            (x - 66) / (75 - 66) if 66 < x <= 75 else 
            (1 if x > 75 else 0)
        )
    return fuzzy_data

def categorize_fuzzy(classification, fuzzy_data):
    combined_data = pd.DataFrame(index=classification.index)
    for column in classification.columns:
        for level in ['L', 'M', 'H']:
            fuzzy_label = column + "_" + level
            if level == 'L':
                combined_data[fuzzy_label + "_I"] = (classification[column] == "increase") & (fuzzy_data[column + '_L'] > 0)
                combined_data[fuzzy_label + "_D"] = (classification[column] == "decrease") & (fuzzy_data[column + '_L'] > 0)
            elif level == 'M':
                combined_data[fuzzy_label + "_I"] = (classification[column] == "increase") & (fuzzy_data[column + '_M'] > 0)
                combined_data[fuzzy_label + "_D"] = (classification[column] == "decrease") & (fuzzy_data[column + '_M'] > 0)
            elif level == 'H':
                combined_data[fuzzy_label + "_I"] = (classification[column] == "increase") & (fuzzy_data[column + '_H'] > 0)
                combined_data[fuzzy_label + "_D"] = (classification[column] == "decrease") & (fuzzy_data[column + '_H'] > 0)
        combined_data[column + "_S"] = (classification[column] == "stable")
    combined_data = combined_data.astype(bool)
    return combined_data

# Fungsi untuk menjalankan FP-Growth dan Association Rules
def run_fpgrowth_and_association_rules(data, min_support, min_confidence, min_lift):
    frequent_itemsets = fpgrowth(data, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        st.warning("Tidak ada frequent itemsets yang ditemukan dengan min_support yang dipilih.")
        return pd.DataFrame()
    rules = association_rules(frequent_itemsets, num_itemsets=2, metric="confidence", min_threshold=min_confidence)
    filtered_rules = rules[rules['lift'] >= min_lift]
    return filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

# Halaman Input Data
def input_data():
    st.title("Input Data")
    uploaded_file = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        if "Periode" in data.columns:
            data.set_index("Periode", inplace=True)
        else:
            st.warning("Kolom 'Periode' tidak ditemukan. Data akan digunakan tanpa mengatur 'Periode' sebagai index.")
        st.write("Data Harga Komoditas", data)
        
        # Pra-pemrosesan data
        data_abs, classification = calculate_percentage_change(data)
        fuzzy_data = fuzzify(data_abs)
        combined_data = categorize_fuzzy(classification, fuzzy_data)
        st.session_state['data'] = combined_data  # Simpan hasil kombinasi di session_state
    else:
        st.info("Silakan unggah file CSV atau Excel untuk memulai analisis.")

# Halaman Hasil Association Rule
def display_results():
    st.title("Hasil Association Rule")
    data = st.session_state.get('data', None)
    if data is not None:
        min_support = st.number_input("Minimum Support", min_value=0.005, max_value=1.0, value=0.1, step=0.01)
        min_confidence = st.number_input("Minimum Confidence", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
        min_lift = st.slider("Minimum Lift", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        rules = run_fpgrowth_and_association_rules(data, min_support, min_confidence, min_lift)
        if not rules.empty:
            st.write("Hasil Association Rules:")
            st.write(rules)
            antecedent_options = data.columns.tolist()
            selected_antecedents = st.multiselect("Pilih Antecedents", antecedent_options)
            if selected_antecedents:
                filtered_rules = rules[rules['antecedents'].apply(lambda x: any(item in x for item in selected_antecedents))]
                st.write("Association Rules dengan Antecedents Terpilih", filtered_rules)
        else:
            st.write("Tidak ada aturan asosiasi yang ditemukan.")
    else:
        st.info("Silakan unggah data pada bagian Input Data terlebih dahulu.")

# Fungsi utama untuk navigasi
def main():
    st.sidebar.title("Navigasi")
    page = st.sidebar.selectbox("Pilih Halaman", ["Main", "Input Data", "Hasil Association Rule"])

    if page == "Main":
        display_info()
    elif page == "Input Data":
        input_data()
    elif page == "Hasil Association Rule":
        display_results()

if __name__ == "__main__":
    main()
