import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Загрузка модели
@st.cache_resource
def load_model():
    model = joblib.load('model_data/KRR_model.pkl')
    feature_info = joblib.load('model_data/feature_info.pkl')
    return model, feature_info

def main():
    st.set_page_config(page_title="ML Concrete Compressive Strenght Predictor Interface", layout="wide")
    
    # Заголовок по центру
    st.markdown("""
    <h1 style='text-align: center;'>
         ML Concrete Compressive Strength Predictor Interface
    </h1>
    """, unsafe_allow_html=True)
    
    # Загрузка модели
    model, feature_info = load_model()
    feature_names = feature_info['feature_names']
    target_name = feature_info['target_name']
    
    st.success("✅ Модель успешно загружена!")
    
    # Сайдбар для ввода
    st.sidebar.header("📝 Ввод параметров")
    input_data = {}
    
    for i, feature in enumerate(feature_names):
        input_data[feature] = st.sidebar.number_input(
            f"{feature}", value=0.0, step=0.1, key=f"input_{i}"
        )
    
    # Основная область
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Предсказание модели")
        if st.button("Сделать предсказание", type="primary"):
            try:
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                st.success(f"**{target_name}: {prediction:.4f}**")
                
                with st.expander("Детали предсказания"):
                    st.dataframe(input_df)
                    
            except Exception as e:
                st.error(f"Ошибка: {e}")
    
    with col2:
        st.subheader("ℹ️ О модели")
        st.info(f"Признаков: {len(feature_names)}")
        st.info(f"Целевая: {target_name}")
    
    # Загрузка файлов
    st.markdown("---")
    st.subheader("📁 Пакетное предсказание")
    
    file_type = st.radio("Тип файла:", ["CSV", "Excel"], horizontal=True)
    uploaded_file = st.file_uploader(
        f"Загрузите {file_type} файл",
        type=['csv'] if file_type == "CSV" else ['xlsx', 'xls']
    )
    
    if uploaded_file:
        try:
            if file_type == "CSV":
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            # Проверяем колонки
            missing = set(feature_names) - set(data.columns)
            if missing:
                st.error(f"Отсутствуют: {missing}")
            else:
                predictions = model.predict(data[feature_names])
                result_df = data.copy()
                result_df[f'Predicted_{target_name}'] = predictions
                
                st.success(f"✅ Обработано {len(data)} строк")
                
                # КОМПАКТНЫЙ ГРАФИК с Plotly
                st.subheader("📊 Распределение предсказаний")
                fig = px.histogram(
                    x=predictions, 
                    nbins=20,
                    title="",
                    labels={'x': 'Предсказания', 'y': 'Частота'},
                    color_discrete_sequence=['#1f77b4']
                )
                fig.update_layout(
                    height=300,  # Компактная высота
                    width=400,   # Компактная ширина
                    showlegend=False,
                    margin=dict(l=40, r=40, t=30, b=40),  # Минимальные отступы
                    font=dict(size=10)
                )
                st.plotly_chart(fig, use_container_width=False)
                
                # Скачивание
                csv = result_df.to_csv(index=False)
                st.download_button("📥 Скачать CSV", data=csv, file_name="predictions.csv")
                
                st.subheader("📋 Предпросмотр результатов")
                st.dataframe(result_df.head(10))
                
        except Exception as e:
            st.error(f"Ошибка обработки: {e}")

if __name__ == "__main__":
    main()