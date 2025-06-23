import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

# Set the title of the Streamlit application
st.set_page_config(layout="wide", page_title="Aplikasi Data Mining Potabilitas Air 💧")

st.title("Aplikasi Data Mining Potabilitas Air 💧")
st.write("""
Aplikasi ini memungkinkan Anda untuk mengeksplorasi dataset potabilitas air dan memprediksi apakah air tersebut layak minum atau tidak,
menggunakan model Machine Learning.
""")

# --- Bagian Sidebar untuk Navigasi atau Opsi ---
st.sidebar.header("Pengaturan Aplikasi")
analysis_type = st.sidebar.radio(
    "Pilih Tipe Analisis:",
    ("Gambaran Umum Data", "Analisis Data Eksplorasi (EDA)", "Prediksi Potabilitas Air")
)

# --- Muat Dataset ---
@st.cache_data # Cache the data loading for better performance
def load_data():
    # Attempt to load the data from the accessible file name
    try:
        df = pd.read_csv("water_potability.csv")
    except FileNotFoundError:
        st.error("Pastikan file 'water_potability.csv' berada di direktori yang sama dengan aplikasi.")
        st.stop() # Stop the app if the file is not found
    return df

df = load_data()

# --- Preprocessing Data Awal (Imputasi Missing Values) ---
# Check for missing values before imputation
st.sidebar.subheader("Penanganan Missing Values")
missing_values_before = df.isnull().sum()
if missing_values_before.any():
    st.sidebar.write("Missing values terdeteksi. Menggunakan imputasi mean...")
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
else:
    st.sidebar.write("Tidak ada missing values. Data bersih.")
    df_imputed = df.copy()

# Ensure 'Potability' column is integer type after imputation if it becomes float
df_imputed['Potability'] = df_imputed['Potability'].astype(int)

# Split features (X) and target (y)
X = df_imputed.drop('Potability', axis=1)
y = df_imputed['Potability']

# --- Bagian 1: Gambaran Umum Data ---
if analysis_type == "Gambaran Umum Data":
    st.header("Gambaran Umum Dataset")
    st.write("Berikut adalah 5 baris pertama dari dataset Anda:")
    st.dataframe(df.head())

    st.write("Bentuk Dataset (Jumlah Baris, Jumlah Kolom):", df.shape)

    st.write("Informasi Statistik Deskriptif:")
    st.dataframe(df.describe())

    st.write("Jumlah Missing Values (Setelah Imputasi jika ada):")
    st.dataframe(df_imputed.isnull().sum().to_frame(name='Missing Values'))

    st.write("Distribusi Kelas 'Potability':")
    potability_counts = df_imputed['Potability'].value_counts().rename(index={0: 'Tidak Layak Minum', 1: 'Layak Minum'})
    st.dataframe(potability_counts)
    fig_pie = px.pie(
        values=potability_counts.values,
        names=potability_counts.index,
        title='Distribusi Potabilitas Air',
        color_discrete_sequence=px.colors.sequential.RdBu # Choose a nice color sequence
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

# --- Bagian 2: Analisis Data Eksplorasi (EDA) ---
elif analysis_type == "Analisis Data Eksplorasi (EDA)":
    st.header("Analisis Data Eksplorasi (EDA)")
    st.write("Visualisasikan distribusi fitur-fitur penting dalam dataset.")

    feature_columns = X.columns.tolist()
    selected_feature = st.selectbox(
        "Pilih Fitur untuk Visualisasi Distribusi:",
        feature_columns
    )

    if selected_feature:
        st.subheader(f"Histogram Distribusi {selected_feature}")
        fig_hist = px.histogram(df_imputed, x=selected_feature, marginal="box",
                                color='Potability', # Color by Potability
                                color_discrete_map={0: 'red', 1: 'green'}, # Specific colors
                                title=f'Distribusi {selected_feature} berdasarkan Potabilitas')
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader(f"Scatter Plot: {selected_feature} vs. Fitur Lain")
        selected_feature_y = st.selectbox(
            "Pilih Fitur Y untuk Scatter Plot:",
            [col for col in feature_columns if col != selected_feature]
        )
        if selected_feature_y:
            fig_scatter = px.scatter(df_imputed, x=selected_feature, y=selected_feature_y,
                                     color='Potability',
                                     color_discrete_map={0: 'red', 1: 'green'},
                                     hover_data=df_imputed.columns,
                                     title=f'Scatter Plot {selected_feature} vs. {selected_feature_y}')
            st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Heatmap Korelasi Antar Fitur")
    corr_matrix = df_imputed.corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=px.colors.sequential.Plasma, # A different color scale
        title='Heatmap Korelasi'
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# --- Bagian 3: Prediksi Potabilitas Air ---
elif analysis_type == "Prediksi Potabilitas Air":
    st.header("Prediksi Potabilitas Air")
    st.write("Model Machine Learning (Random Forest Classifier) akan dilatih untuk memprediksi potabilitas air.")

    # Model Training
    st.subheader("Pelatihan Model")
    test_size = st.slider("Ukuran Data Uji (proporsi):", 0.1, 0.5, 0.2, 0.05)
    random_state = st.slider("Random State (untuk reproduktifitas):", 0, 100, 42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y) # stratify for balanced classes

    n_estimators = st.slider("Jumlah Estimator (pohon) untuk Random Forest:", 50, 500, 100, 50)
    max_depth = st.slider("Kedalaman Maksimal Pohon:", 3, 20, 10, 1)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Evaluasi Model")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Akurasi", f"{accuracy:.2f}")
    with col2:
        st.metric("Presisi", f"{precision:.2f}")
    with col3:
        st.metric("Recall", f"{recall:.2f}")
    with col4:
        st.metric("F1-Score", f"{f1:.2f}")

    st.write("Model telah dilatih dan dievaluasi. Sekarang Anda bisa memasukkan nilai untuk prediksi.")

    # User Input for Prediction
    st.subheader("Masukkan Nilai Air untuk Prediksi")

    # Create input fields dynamically based on features
    input_data = {}
    for column in X.columns:
        # Get min and max for slider, if applicable
        min_val = float(df_imputed[column].min())
        max_val = float(df_imputed[column].max())
        mean_val = float(df_imputed[column].mean())
        
        # Adjust step based on the range of values for better UX
        if max_val - min_val > 100:
            step = 10.0
        elif max_val - min_val > 10:
            step = 1.0
        else:
            step = 0.01 # For smaller ranges, allow more precision

        # Using number_input for more control and precision
        input_data[column] = st.number_input(
            f"{column.replace('_', ' ').title()}:",
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=step,
            format="%.4f" # Maintain precision in display
        )

    predict_button = st.button("Prediksi Potabilitas")

    if predict_button:
        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data])
        
        # Ensure the order of columns matches X_train/X
        input_df = input_df[X.columns]

        # Predict using the trained model
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader("Hasil Prediksi:")
        if prediction[0] == 1:
            st.success("💧 Air Ini **LAYAK MINUM**!")
        else:
            st.error("🚫 Air Ini **TIDAK LAYAK MINUM**.")

        st.write(f"Probabilitas Layak Minum: **{prediction_proba[0][1]*100:.2f}%**")
        st.write(f"Probabilitas Tidak Layak Minum: **{prediction_proba[0][0]*100:.2f}%**")
        st.markdown("""
        <div style="background-color:#e0f7fa; padding:10px; border-radius:5px;">
            <p><strong>Catatan:</strong> Prediksi ini didasarkan pada model yang dilatih dari data yang tersedia. Selalu konsultasikan dengan ahli atau lakukan pengujian lebih lanjut untuk keputusan kritis terkait kualitas air.</p>
        </div>
        """, unsafe_allow_html=True)