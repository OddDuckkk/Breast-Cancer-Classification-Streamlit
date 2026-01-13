import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay
)

from model.GaussianNaiveBayes import GaussianNaiveBayes
from model.RandomForest import ManualRandomForest

from preprocessing.validation import validate_dataset, validate_data_klasifikasi, REQUIRED_COLUMNS, FITUR
from preprocessing.preprocessingGNB import preprocess_gnb
from preprocessing.preprocessingRF import preprocess_rf

# =========================
# SIDEBAR MENU
# =========================
st.sidebar.title("📌 Menu Navigasi")

menu_group = st.sidebar.radio(
    "Pilih Modul",
    ["Preparasi Model", "Klasifikasi Data Baru"]
)

if menu_group == "Preparasi Model":
    page = st.sidebar.radio(
        "Preparasi Model",
        ["Dataset", "Preprocessing", "Training"]
    )

elif menu_group == "Klasifikasi Data Baru":
    page = st.sidebar.radio(
        "Klasifikasi Data Baru",
        ["Input Data", "Klasifikasi"]
    )

# =========================
# HALAMAN DATASET
# =========================
if page == "Dataset":
    st.header("📂 Dataset")

    st.info("📂 Silakan unggah dataset CSV untuk pembuatan model.")

    with st.expander("📌 Fitur wajib dalam dataset"):
        st.table(pd.DataFrame({"Fitur": REQUIRED_COLUMNS}))

    # ===== INIT STATE =====
    if "dataset" not in st.session_state:
        st.session_state.dataset = None

    # ===== UPLOAD (HANYA JIKA BELUM ADA DATASET) =====
    if st.session_state.dataset is None:
        uploaded_file = st.file_uploader(
            "Upload dataset Breast Cancer (CSV)",
            type=["csv"],
            key="dataset_uploader"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                is_valid, missing, extra = validate_dataset(df)

                if not is_valid:
                    st.error("❌ Dataset tidak valid")
                    st.code(missing)
                    st.stop()

                st.session_state.dataset = df
                st.success("✅ Dataset berhasil disimpan dan siap diproses")

                st.rerun()  

            except Exception as e:
                st.error("❌ Gagal membaca CSV")
                st.exception(e)

    # ===== PREVIEW & VALIDASI =====
    if st.session_state.dataset is not None:
        df = st.session_state.dataset

        st.subheader("📊 Preview Dataset")
        st.dataframe(df.head())

        st.subheader("🔎 Validasi Kolom")
        is_valid, missing, extra = validate_dataset(df)

        st.success("✅ Semua kolom wajib tersedia")

        if extra:
            st.warning("⚠️ Kolom tambahan (diabaikan):")
            st.code(extra)

        if st.button("🗑️ Hapus Dataset"):
            st.session_state.dataset = None
            st.rerun()


# =========================
# HALAMAN PREPROCESSING
# =========================
elif page == "Preprocessing":
    st.header("🧹 Preprocessing")

    # ===== CEK DATASET =====
    if "dataset" not in st.session_state or st.session_state["dataset"] is None:
        st.warning(
            "⚠️ Dataset belum tersedia.\n\n"
            "Silakan unggah dataset terlebih dahulu pada halaman **Dataset**."
        )

    else:
        df = st.session_state["dataset"]

        st.info(
            "📌 **Tahapan preprocessing yang akan dilakukan:**\n\n"
            "🔹 **Gaussian Naive Bayes**\n"
            "- Data Cleaning\n"
            "- Pemisahan X & Y\n"
            "- Feature selection berbasis korelasi\n\n"
            "🔹 **Random Forest**\n"
            "- Data Cleaning\n"
            "- Pemisahan X & Y\n"
        )

        # ===== INIT FLAG =====
        if "preprocessing_done" not in st.session_state:
            st.session_state["preprocessing_done"] = False

        # ===== MULAI PREPROCESS =====
        if not st.session_state["preprocessing_done"]:
            if st.button("🚀 Mulai Preprocessing"):
                try:
                    # ---- GNB ----
                    X_gnb, y_gnb, dropped_features = preprocess_gnb(df)

                    # ---- RANDOM FOREST ----
                    X_rf, y_rf, le_rf, scaler_rf = preprocess_rf(df)

                    # ---- SIMPAN KE SESSION STATE ----
                    st.session_state["X_gnb"] = X_gnb
                    st.session_state["y_gnb"] = y_gnb
                    st.session_state["gnb_dropped_features"] = dropped_features

                    st.session_state["X_rf"] = X_rf
                    st.session_state["y_rf"] = y_rf
                    st.session_state["rf_label_encoder"] = le_rf
                    st.session_state["rf_scaler"] = scaler_rf

                    st.session_state["preprocessing_done"] = True
                    st.success("✅ Preprocessing berhasil dilakukan untuk semua model")
                    st.rerun()

                except Exception as e:
                    st.error("❌ Terjadi kesalahan saat preprocessing")
                    st.exception(e)

        # ===== RINGKASAN HASIL =====
        if st.session_state["preprocessing_done"]:
            st.success("✅ Preprocessing telah dilakukan")

            col1, col2 = st.columns(2)

            # ---- GNB ----
            with col1:
                st.subheader("📊 Gaussian Naive Bayes")
                st.write(f"Jumlah fitur: **{st.session_state['X_gnb'].shape[1]}**")
                st.write(f"Jumlah data: **{st.session_state['X_gnb'].shape[0]}**")

                dropped = st.session_state.get("gnb_dropped_features", [])
                if dropped:
                    st.warning("Fitur dihapus (korelasi tinggi):")
                    st.code(dropped)
                else:
                    st.success("Tidak ada fitur yang dihapus")

            # ---- RANDOM FOREST ----
            with col2:
                st.subheader("🌲 Random Forest")
                st.write(f"Jumlah fitur: **{st.session_state['X_rf'].shape[1]}**")
                st.write(f"Jumlah data: **{st.session_state['X_rf'].shape[0]}**")

            # with st.expander("🔍 Preview Data Setelah Preprocessing (GNB)"):
            #     st.dataframe(st.session_state["X_gnb"].head())
            # with st.expander("🔍 Preview Data Setelah Preprocessing (RF)"):
            #     st.dataframe(st.session_state["X_rf"].head())

elif page == "Training":
    st.header("🔁 Training Model")
    # ===== CEK PREPROCESSING =====
    if "preprocessing_done" not in st.session_state or st.session_state["preprocessing_done"] is None:
        st.warning(
            "⚠️ Preprocessing belum dilakukan.\n\n"
            "Silakan lakukan preprocessing terlebih dahulu pada halaman **Preprocessing**."
        )

    else:
        X_gnb = st.session_state["X_gnb"]
        y_gnb = st.session_state["y_gnb"]
        X_np = np.asarray(X_gnb, dtype=float)
        y_np = np.asarray(y_gnb)

        X_rf = st.session_state["X_rf"] 
        y_rf = st.session_state["y_rf"] 

        # ===== INIT FLAG =====
        if "training_done" not in st.session_state:
            st.session_state["training_done"] = False

        # ===== MULAI TRAINING =====
        if not st.session_state["training_done"]:
            if st.button("🚀 Mulai Training"):
                try:
                    #TRAINING GNB
                    model = GaussianNaiveBayes(var_smoothing=0.001)

                    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

                    accuracies, precisions, recalls, f1s = [], [], [], []
                    y_true_all, y_pred_all = [], []

                    for train_idx, val_idx in skf.split(X_np, y_np):
                        X_train, X_val = X_np[train_idx], X_np[val_idx]
                        y_train, y_val = y_np[train_idx], y_np[val_idx]

                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)

                        accuracies.append(accuracy_score(y_val, y_pred))
                        precisions.append(precision_score(y_val, y_pred, average="weighted", zero_division=0))
                        recalls.append(recall_score(y_val, y_pred, average="weighted", zero_division=0))
                        f1s.append(f1_score(y_val, y_pred, average="weighted", zero_division=0))

                        y_true_all.append(y_val)
                        y_pred_all.append(y_pred)

                    # ===== SIMPAN HASIL =====
                    st.session_state["gnb_model"] = model
                    st.session_state["gnb_metrics"] = {
                        "accuracy": np.mean(accuracies),
                        "precision": np.mean(precisions),
                        "recall": np.mean(recalls),
                        "f1": np.mean(f1s)
                    }

                    y_true_all = np.concatenate(y_true_all)
                    y_pred_all = np.concatenate(y_pred_all)
                    cm = confusion_matrix(y_true_all, y_pred_all)

                    st.session_state["gnb_cm"] = cm

                    #TRAINING RF
                    X_rf = st.session_state["X_rf"]
                    y_rf = st.session_state["y_rf"]

                    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

                    acc_scores, recall_scores = [], []
                    cms = []

                    for train_idx, test_idx in skf.split(X_rf, y_rf):
                        X_train, X_test = X_rf[train_idx], X_rf[test_idx]
                        y_train, y_test = y_rf[train_idx], y_rf[test_idx]

                        model_rf = ManualRandomForest(
                            n_estimators=10,
                            max_depth=5,
                            max_features='sqrt'
                        )

                        model_rf.fit(X_train, y_train)
                        y_pred = model_rf.predict(X_test)

                        acc_scores.append(accuracy_score(y_test, y_pred))
                        recall_scores.append(recall_score(y_test, y_pred))
                        cms.append(confusion_matrix(y_test, y_pred))

                    st.session_state["rf_model"] = model_rf
                    st.session_state["rf_metrics"] = {
                        "accuracy": np.mean(acc_scores),
                        "recall": np.mean(recall_scores)
                    }

                    st.session_state["rf_cm"] = np.sum(cms, axis=0)
                    st.session_state["training_done"] = True

                    st.success("✅ Training Gaussian Naive Bayes dan Random Forest selesai")
                    st.rerun()

                except Exception as e:
                    st.error("❌ Terjadi kesalahan saat training")
                    st.exception(e)
        # =========================
        # HASIL TRAINING
        # =========================
        if st.session_state["training_done"]:
            st.success("✅ Model telah ditraining")

            st.info("Metrics Gaussian Naive Bayes:")
            gnbmetrics = st.session_state["gnb_metrics"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{gnbmetrics['accuracy']*100:.2f}%")
            col2.metric("Precision", f"{gnbmetrics['precision']*100:.2f}%")
            col3.metric("Recall", f"{gnbmetrics['recall']*100:.2f}%")
            col4.metric("F1-Score", f"{gnbmetrics['f1']*100:.2f}%")

            st.subheader("🧩 Confusion Matrix")

            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(
                confusion_matrix=st.session_state["gnb_cm"]
            )
            disp.plot(ax=ax, values_format="d", cmap=plt.cm.Blues)
            st.pyplot(fig)

            st.info("Metrics Random Forest:")
            rfmetrics = st.session_state["rf_metrics"]

            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{rfmetrics['accuracy']*100:.2f}%")
            col2.metric("Recall", f"{rfmetrics['recall']*100:.2f}%")

            st.subheader("🧩 Confusion Matrix")

            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(
                confusion_matrix=st.session_state["rf_cm"]
            )
            disp.plot(ax=ax, values_format="d", cmap=plt.cm.Blues)
            st.pyplot(fig)

            st.divider()
            st.header("⚖️ Perbandingan Model")

            gnb_acc = gnbmetrics["accuracy"]
            gnb_rec = gnbmetrics["recall"]

            rf_acc = rfmetrics["accuracy"]
            rf_rec = rfmetrics["recall"]

            comparison_df = pd.DataFrame({
                "Model": ["Gaussian Naive Bayes", "Random Forest"],
                "Accuracy (%)": [gnb_acc * 100, rf_acc * 100],
                "Recall (%)": [gnb_rec * 100, rf_rec * 100]
            })
            # =========================
            # PILIH MODEL TERBAIK
            # =========================
            if gnb_acc > rf_acc:
                chosen_model = "GNB"
            elif rf_acc > gnb_acc:
                chosen_model = "RF"
            else:
                # Jika accuracy sama → bandingkan recall
                chosen_model = "GNB" if gnb_rec >= rf_rec else "RF"

            st.session_state["chosen_model"] = chosen_model

            if chosen_model == "GNB":
                st.session_state["best_model_object"] = st.session_state["gnb_model"]
            else:
                st.session_state["best_model_object"] = st.session_state["rf_model"]

            st.dataframe(comparison_df, use_container_width=True)

elif page == "Input Data":
    st.header("🧾 Input Data Pasien")
    st.info("Silakan unggah dataset CSV untuk diklasifikasi")

    # =========================
    # CEK MODEL TERPILIH
    # =========================
    if "chosen_model" not in st.session_state:
        st.warning("⚠️ Model belum dipilih. Silakan lakukan training terlebih dahulu.")
        st.stop()

    chosen_model = st.session_state["chosen_model"]

    with st.expander("📌 Fitur wajib dalam data pasien"):
        st.table(pd.DataFrame({"Fitur": REQUIRED_COLUMNS}))

    # =========================
    # INIT STATE
    # =========================
    if "data_klasifikasi" not in st.session_state:
        st.session_state["data_klasifikasi"] = None

    if "X_input" not in st.session_state:
        st.session_state["X_input"] = None

    # =========================
    # UPLOAD CSV
    # =========================
    if st.session_state["data_klasifikasi"] is None:
        uploaded_file = st.file_uploader(
            "Upload data untuk diklasifikasi",
            type=["csv"]
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                is_valid, missing, extra = validate_data_klasifikasi(df)

                if not is_valid:
                    st.error("❌ Data tidak valid")
                    st.code(missing)
                    st.stop()

                st.session_state["data_klasifikasi"] = df
                st.success("✅ Data berhasil diunggah")
                st.rerun()

            except Exception as e:
                st.error("❌ Gagal membaca file CSV")
                st.exception(e)

    # =========================
    # PREVIEW & PREPROCESSING
    # =========================
    if st.session_state["data_klasifikasi"] is not None:
        df = st.session_state["data_klasifikasi"]

        st.subheader("📊 Preview Data")
        st.dataframe(df.head())


        # =========================
        # RESET
        # =========================
        if st.button("🗑️ Hapus Dataset"):
            st.session_state["data_klasifikasi"] = None
            st.session_state["X_input"] = None
            st.rerun()

elif page == "Klasifikasi":
    st.header("🔍 Klasifikasi Data Pasien")

    # ===== CEK MODEL =====
    if "chosen_model" not in st.session_state:
        st.warning("⚠️ Model belum dipilih. Lakukan training terlebih dahulu.")

    model_type = st.session_state["chosen_model"]
    st.success(f"✅ Model yang digunakan: **{model_type}**")

    # ===== CEK DATA =====
    if "data_klasifikasi" not in st.session_state:
        st.warning("⚠️ Data belum diinput pada halaman Input Data.")
        st.stop()

    df = st.session_state["data_klasifikasi"]
    st.subheader("📊 Preview Data")
    st.dataframe(df.head())

    # ===== TOMBOL KLASIFIKASI =====
    if st.button("🚀 Jalankan Klasifikasi"):

        try:
            # ==================================================
            # GNB
            # ==================================================
            if model_type == "GNB":
                st.info("⚙️ Preprocessing untuk Gaussian Naive Bayes")
                TO_DROP_GNB = st.session_state["gnb_dropped_features"]
                model = st.session_state["gnb_model"]
                X_input = (
                    df
                    .loc[:, FITUR]
                    .drop(columns=TO_DROP_GNB, errors="ignore")
                    .astype(float)
                    .values
                )

                y_pred = model.predict(X_input)

            # ==================================================
            # RANDOM FOREST
            # ==================================================
            elif model_type == "RF":
                model = st.session_state["rf_model"]
                st.info("⚙️ Preprocessing Random Forest")

                scaler = st.session_state["rf_scaler"]
                label_encoder = st.session_state["rf_label_encoder"]

                X_input = (
                    df
                    .loc[:, FITUR]
                    .astype(float)
                )

                X_scaled = scaler.transform(X_input)

                y_pred_encoded = model.predict(X_scaled)
                y_pred = label_encoder.inverse_transform(y_pred_encoded)

            # ===== SIMPAN HASIL =====
            df_result = X_input.copy()
            df_result["Hasil Prediksi"] = y_pred

            st.session_state["hasil_klasifikasi"] = df_result

            # =========================
            # TAMPILKAN HASIL KLASIFIKASI
            # =========================
            if "hasil_klasifikasi" in st.session_state:
                st.subheader("📈 Hasil Klasifikasi")

                # Tampilkan dataframe
                df_result = st.session_state["hasil_klasifikasi"]
                st.dataframe(df_result)

                # Statistik singkat (misal jumlah per kelas)
                st.subheader("📊 Distribusi Hasil Prediksi")
                st.bar_chart(df_result["Hasil Prediksi"].value_counts())

                # Tombol download CSV
                st.download_button(
                    label="⬇️ Download Hasil CSV",
                    data=df_result.to_csv(index=False),
                    file_name="hasil_klasifikasi.csv",
                    mime="text/csv"
                )

            st.success("✅ Klasifikasi berhasil")

        except Exception as e:
            st.error("❌ Terjadi kesalahan saat klasifikasi")
            st.exception(e)


