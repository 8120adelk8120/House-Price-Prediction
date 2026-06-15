import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os

def load_assets():
    model_path = "deep_house_model.h5"
    scaler_path = "scaler.pkl"
    columns_path = "features_columns.pkl"
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(columns_path)):
        return None, None, None
        
    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    columns = joblib.load(columns_path)
    return model, scaler, columns

model, scaler, columns = load_assets()

def predict_house_price(area, rooms, parking, warehouse, elevator, address):
    if model is None:
        return "⚠️ خطای سیستم: فایل‌های مدل یافت نشدند!"
        
    input_df = pd.DataFrame(0, index=[0], columns=columns)
    input_df['Area'] = float(area)
    input_df['Room'] = int(rooms)
    input_df['Parking'] = 1 if parking else 0
    input_df['Warehouse'] = 1 if warehouse else 0
    input_df['Elevator'] = 1 if elevator else 0
    
    address_col = f"Address_{address}"
    if address_col in input_df.columns:
        input_df[address_col] = 1
        
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0][0]
    
    if prediction < 0:
        return "❌ ورودی‌های نامعتبر! لطفاً متراژ یا محله را تغییر دهید."
    
    return f"💰 قیمت تخمینی هوش مصنوعی: {prediction:,.0f} تومان"

if columns is not None:
    addresses = sorted([col.replace("Address_", "") for col in columns if "Address_" in col])
else:
    addresses = []

with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple", secondary_hue="indigo")) as demo:
    gr.Markdown("# 🚀 سامانه هوشمند تخمین قیمت مسکن تهران (Gradio)")
    gr.Markdown("مشخصات ملک را در کادرهای زیر وارد کنید تا شبکه عصبی عمیق ارزش آن را محاسبه کند.")
    
    with gr.Row():
        with gr.Column():
            area = gr.Number(label="📐 متراژ بنا (متر مربع)", value=85)
            rooms = gr.Slider(minimum=0, maximum=5, step=1, label="🛏️ تعداد اتاق خواب", value=2)
            address = gr.Dropdown(choices=addresses, label="📍 انتخاب محله", value=addresses[0] if addresses else None)
            
            gr.Markdown("**🛠️ امکانات ملک:**")
            parking = gr.Checkbox(label="🅿️ پارکینگ دارد", value=True)
            warehouse = gr.Checkbox(label="📦 انباری دارد", value=True)
            elevator = gr.Checkbox(label="🔼 آسانسور دارد", value=True)
            
            submit_btn = gr.Button("🔮 محاسبه قیمت", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(label="📡 خروجی هسته پردازش مدل", placeholder="منتظر ورود اطلاعات...", interactive=False, lines=4)

    submit_btn.click(
        fn=predict_house_price,
        inputs=[area, rooms, parking, warehouse, elevator, address],
        outputs=output_text
    )

demo.launch()