import streamlit as st
import pandas as pd
import pickle as pkl
import plotly.express as px
from streamlit_option_menu import option_menu
import numpy as np
import streamlit as st
import pandas as pd
import pickle

# Streamlit Page Configuration
st.set_page_config(
    page_title="Aplikasi Klasifikasi Pneumonia",
    page_icon="https://eu-images.contentstack.com/v3/assets/blt6b0f74e5591baa03/blt7c0bf7e21d4410b4/6319700b8cc2fa14e223aa27/8895.png",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get help": "https://github.com/AdieLaine/Streamly",
        "Report a bug": "https://github.com/AdieLaine/Streamly",
        "About": """
            ## Streamly Streamlit Assistant
            ### Powered using GPT-4o-mini

            **GitHub**: https://github.com/AdieLaine/

            The AI Assistant named, Streamly, aims to provide the latest updates from Streamlit,
            generate code snippets for Streamlit widgets,
            and answer questions about Streamlit's latest features, issues, and more.
            Streamly has been trained on the latest Streamlit updates and documentation.
        """
    })

# Sidebar untuk memilih halaman
with st.sidebar:
    selected = option_menu(
    menu_title=None,
    options=["Beranda", "Visualisasi", "Klasifikasi"],
    icons=["house", "book", "calculator"],
    menu_icon="cast",
    default_index=0,
    # orientation="horizontal",
)

# Load dataset
df = pd.read_csv('datapneumonia.csv')
df_1 = pd.read_csv('datapneumonia_clean.csv')


if selected == "Beranda":
    #Judul Aplikasi
    st.title("Aplikasi Klasifikasi Pneumonia")
    st.text("Aplikasi Klasifikasi Pneumonia adalah sebuah aplikasi yang dapat menghasilkan klasifikasi pneumonia pada pasien berdasarkan beberapa fitur yang dimasukkan. Fitur-fitur yang diinput yaitu Jenis Kelamin, Usia, Gejala seperti batuk, pilek, demam dan kesulitan bernapas serta hasil dari pemeriksaan CRP dan Darah putih pada pasien. Hasil inputan berupa klasifikasi Pneumonia Berat, Pneumonia Ringan dan Pneumonia Sedang.")
    st.image("pneumonia.jpg", caption='sumber: https://www.halodoc.com/', use_container_width=True)
    
    #Menunjukkan dataset dan deskripsinya
    st.markdown("Ini adalah sampel dataset")
    st.write(df.head())
    st.markdown("Hasil Pengolahan Data")
    st.write(df_1.head())
    st.markdown('Atribut Dataset :')
    st.markdown("1. Jenis Kelamin : Merupakan jenis kelamin pasien (kelas 1 untuk Perempuan, kelas 0 untuk Laki-Laki)")
    st.markdown("2. Batuk : Apakah pasien mengalami gejala batuk atau tidak?")
    st.markdown("3. Demam : Apakah pasien mengalami gejala demam atau tidak?")
    st.markdown("4. Kesulitan Bernapas : Apakah pasien mengalami gejala kesulitan bernapas atau tidak?")
    st.markdown("5. Pilek : Apakah pasien mengalami gejala pilek atau tidak?")
    st.markdown("6. Usia (tahun) : Merupakan usia pasien (dalam format tahun)")
    st.markdown("7. Riwayat Penyakit (encoded) : Apakah pasien memiliki riwayat penyakit atau tidak? seperti Asma, Hipertensi, diabetes dll.")
    st.markdown("8. CRP (C-Reactive Protein)  : CRP adalah protein yang diproduksi oleh hati sebagai respons terhadap peradangan atau infeksi di tubuh. Kadar CRP dalam darah meningkat signifikan selama proses inflamasi, termasuk infeksi seperti pneumonia. Nilai Normal: Biasanya <10 mg/L, tetapi pada pneumonia bisa meningkat menjadi >100 mg/L, tergantung pada tingkat keparahan.")
    st.markdown("9. Darah Putih (Leukosit): Sel darah putih adalah bagian dari sistem kekebalan tubuh yang berperan dalam melawan infeksi. Jumlah leukosit sering diukur sebagai indikator infeksi. Leukositosis (peningkatan leukosit): Biasanya terjadi pada pneumonia bakteri, dengan jumlah leukosit >11.000/µL. Peningkatan neutrofil (jenis leukosit) sering ditemukan pada pneumonia bakteri. Leukopenia (penurunan leukosit): Bisa terjadi pada infeksi berat atau pneumonia yang disebabkan oleh virus tertentu. Nilai Normal: 4.000–11.000 sel/µL. ")
    st.markdown("10. Label Kelas : Merupakan hasil diagnosis pasien Pneumonia, kelas 0 berarti Pneumonia Berat, kelas 1 berarti Pneumonia Ringan, dan kelas 2 berarti Pneumonia Sedang")

elif selected == "Visualisasi":
    st.title("Hasil dan Visualisasi Data")
    model_name = "Random Forest"
    accuracy = 98.0
    # Menampilkan hasil akurasi
    st.subheader(f"Model yang Digunakan: {model_name}")
    st.metric(label="Akurasi", value=f"{accuracy}%", delta="")
    
    #Visualisasi Label Kelas
    fig = px.histogram(df_1, x='Label Kelas', color='Label Kelas', hover_data=df_1.columns)
    fig.update_layout(title='Jumlah Pasien Pneumonia', xaxis_title='Label Kelas', yaxis_title='Jumlah', font=dict(size=15))
    st.plotly_chart(fig)
    st.markdown(" Keterangan : Kelas 0 = Pneumonia Berat, Kelas 1 = Pneumonia Ringan, Kelas 2 = Pneumonia Sedang")
    
    #Visualisasi fitur histogram
    st.subheader('Pilih fitur yang ingin ditampilkan histogramnya')
    fitur = st.selectbox('Fitur', df_1.columns.tolist())
    fig = px.histogram(df_1, x=fitur, color='Label Kelas', marginal='box', hover_data=df_1.columns)
    st.plotly_chart(fig)
    
    #Visualisasi fitur scatter plot
    st.subheader('Pilih fitur yang ingin ditampilkan scatter plotnya')
    fitur1 = st.selectbox('Fitur 1', df_1.columns.tolist())
    fitur2 = st.selectbox('Fitur 2', df_1.columns.tolist())
    fig = px.scatter(df_1, x=fitur1, y=fitur2, color='Label Kelas', hover_data=df_1.columns)
    st.plotly_chart(fig)
    
    #visualisasi antar label dan fitur
    fig = px.histogram(df_1, x='Label Kelas', color='Riwayat Penyakit (Encoded)', barmode='group', hover_data=df_1.columns)
    fig.update_layout(title='Jumlah pasien dengan Riwayat Penyakit yang dimiliki', xaxis_title='Label Kelas', yaxis_title='Jumlah', font=dict(size=15))
    st.plotly_chart(fig)
    st.markdown(" Keterangan : Riwayat Penyakit: Kelas 0 = Tidak Ada, Kelas 1 = Memiliki Riwayat Penyakit")
    
elif selected == "Klasifikasi":
    st.title("Klasifikasi Pneumonia")
    st.write("Silahkan Input Data Pasien")
    
    # Input dari pengguna
    nama = st.text_input('Masukkan nama', ' ')
    usia = st.number_input("Masukkan Usia:", min_value=0.0, step=0.1, format="%.1f")
    satuan = st.selectbox("Pilih Satuan Usia:", ["Tahun", "Bulan", "Hari"])
    # Konversi usia ke tahun jika input dalam bulan
    if satuan == "Bulan":
        usia_tahun = usia / 12  # Konversi bulan ke tahun
    elif satuan == "Hari":
        usia_tahun = usia / 365 # Konversi hari ke tahun
    else:
        usia_tahun = usia
    st.write("Usia dalam tahun:", usia_tahun)
    jk = st.radio('Masukkan Jenis Kelamin', ['Perempuan','Laki-Laki'], index=None)
    batuk = st.radio('Apakah pasien mengalami batuk?', ['Tidak','Ya'], index=None)
    demam = st.radio('Apakah pasien mengalami demam?', ['Ya','Tidak'], index=None)
    sulit_bernapas = st.radio('Apakah pasien mengalami kesulitan bernapas?', ['Ya','Tidak'], index=None)
    pilek = st.radio('Apakah pasien mengalami pilek?', ['Ya','Tidak'], index=None)
    crp = st.number_input("Masukkan CRP(mg/L)", min_value=0.0, max_value=300.0, step=0.1)
    darah_putih = st.number_input("Masukkan Darah Putih(x10⁹/L)", min_value=0.0, max_value=100.0, step=0.1)
    # Normalisasi CRP dan Darah Putih menggunakan MinMaxScaler
    input_data = np.array([[crp, darah_putih]])
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    # Menormalkan kedua fitur
    normalized_data = scaler.transform(input_data)
    # Input untuk Riwayat Penyakit
    riwayat_penyakit = st.multiselect("Pilih Riwayat Penyakit (bisa lebih dari satu):", options=["Tidak Ada", "Diabetes", "Hipertensi", "Jantung", "Asma"])
    # Proses input Riwayat Penyakit menjadi encoding
    riwayat_encoded = 0 if "Tidak Ada" in riwayat_penyakit else 1

    # Mengonversikan input menjadi data frame
    data_input = {
        'Jenis Kelamin': [1 if jk == 'Perempuan' else 0],
        'Batuk': [1 if batuk == 'Ya' else 0],
        'Demam': [1 if demam == 'Ya' else 0],
        'Kesulitan Bernapas': [1 if sulit_bernapas == 'Ya' else 0],
        'Pilek': [1 if pilek == 'Ya' else 0],
        'Usia (tahun)': [usia_tahun],
        'Riwayat Penyakit (Encoded)': [riwayat_encoded],
        'CRP(mg/L) (norm)': [normalized_data[0][0]],
        'Darah Putih(x10⁹/L) (norm)': [normalized_data[0][1]]
    }

    # Membuat data frame dari inputan
    df = pd.DataFrame(data_input)
    
     #model
    model = pickle.load(open("model_random_forest.pkl", "rb"))
    # Hasil Klasifikasi
    if st.button("Hasil Klasifikasi"):
        
        prediction = model.predict(df)
        if prediction[0] == 0:
            st.success("Hasil Klasifikasi: Pneumonia Berat")
        elif prediction[0] == 1:
            st.success("Hasil Klasifikasi: Pneumonia Ringan")
        elif prediction[0] == 2:
            st.success("Hasil Klasifikasi: Pneumonia Sedang")
        else:
            st.error("Kelas yang diprediksi tidak dikenal")
    
