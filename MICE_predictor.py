import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import io

# Load the model
model = joblib.load('XGBoost_best_model.pkl')

# Define feature names used for the model
feature_names = [
    "Age", "Drinking", "Chronic", "Diabetes", "HDLC", "LDLC", "PTINR"
]

# Streamlit user interface
st.title("急性冠状动脉综合征患者1年内发生心血管不良事件风险预测")

# Age: numerical input
Age = st.number_input(
    "年龄 (岁):", 
    min_value=0.0,    # 必须是 float
    max_value=100.0,  # 必须是 float
    value=45.0,       # 必须是 float
    format="%d",      # 显示为整数（不显示.0）
    step=1.0          # 必须指定 step 为 float
)

# Drinking: categorical selection
Drinking = st.selectbox("是否饮酒 (0=否, 1=是):", options=[0, 1], format_func=lambda x: '否 (0)' if x == 0 else '是 (1)')

# Diabetes: categorical selection
Diabetes = st.selectbox("是否患有糖尿病 (0=否, 1=是):", options=[0, 1], format_func=lambda x: '否 (0)' if x == 0 else '是 (1)')

# Chronic: categorical selection
Chronic = st.selectbox("是否合并其他慢性疾病 (0=否, 1=是):", options=[0, 1], format_func=lambda x: '否 (0)' if x == 0 else '是 (1)')

# HDLC: numerical input
HDLC = st.number_input("高密度脂蛋白胆固醇 (HDL-C, mmol/L):", min_value=0.0, max_value=10.0, value=1.2, step=0.1)

# LDLC: numerical input
LDLC = st.number_input("低密度脂蛋白胆固醇 (LDL-C, mmol/L):", min_value=0.0, max_value=20.0, value=3.0, step=0.1)

# PTINR: numerical input (keeping high max as per your data)
PTINR = st.number_input(
    "凝血相关指标 (PTINR):",
    min_value=0.0,
    max_value=5000.0,
    value=1.0 
)

# Process inputs and make predictions
feature_values = [Age, Drinking, Chronic, Diabetes, HDLC, LDLC, PTINR]
features = np.array([feature_values], dtype=float)

if st.button("预测"):
    try:
        # Predict class and probabilities
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]

        # Display prediction results
        st.write(f"**预测类别:** {'高风险' if predicted_class == 1 else '低风险'} (MACE=1表示高风险)")
        st.write(f"**预测概率:** 无事件={predicted_proba[0]:.2%}, 事件={predicted_proba[1]:.2%}")

        # Generate advice with your requested wording
        if predicted_class == 1:
            probability_mace = predicted_proba[1] * 100  # 正确定义变量
            advice = (
                "根据模型预测，您在1年内发生心血管不良事件 (MACE) 的风险较高。\n"
                f"模型预测的发病概率为 {probability_mace:.1f}%。\n"
                "强烈建议您与主治医生详细讨论此结果，并采取积极的干预措施，如调整药物治疗、改变生活方式等。"
            )
        else:
            probability_no_mace = predicted_proba[0] * 100  # 正确定义变量
            advice = (
                "根据模型预测，您在1年内发生心血管不良事件 (MACE) 的风险较低。\n"
                f"模型预测的无事件概率为 {probability_no_mace:.1f}%。\n"
                "建议您继续保持健康的生活方式，并遵医嘱定期复查。"
            )
        
        st.info(advice)  # 使用info框使建议更醒目

        # Calculate SHAP values and display force plot
        explainer = shap.TreeExplainer(model)
        input_df = pd.DataFrame([feature_values], columns=feature_names)
        
        # Get SHAP values
        shap_values = explainer.shap_values(input_df)
        
        # For binary classification, shap_values is a list with two arrays
        # We want the SHAP values for the positive class (class 1)
        shap_values_for_positive_class = shap_values[1] if isinstance(shap_values, list) else shap_values
        
        # Create force plot in memory
        plt.figure(figsize=(20, 3))
        shap.force_plot(
            explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
            shap_values_for_positive_class[0],
            input_df.iloc[0],
            matplotlib=True,
            show=False,
            text_rotation=45
        )
        
        # Save to buffer instead of file
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        buf.seek(0)
        
        # Display the plot
        st.image(buf, caption='特征贡献度解释 (红色增加风险，蓝色降低风险)', use_column_width=True)
        
        # Add clinical interpretation note
        st.caption("""
        **结果解读**：上图展示了各因素对预测结果的贡献。箭头向右（红色）表示增加MACE风险，箭头向左（蓝色）表示降低风险。
        基线值（Base value）是所有患者的平均预测概率，最终预测值（f(x)）是考虑所有特征后的结果。
        """)

    except Exception as e:
        st.error(f"预测过程中出错: {str(e)}")

        st.exception(e)  # 显示完整错误堆栈

