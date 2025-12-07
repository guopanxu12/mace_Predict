import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('XGBoost_best_model.pkl')

# Define feature names used for the model
feature_names = [
    "Age", "Drinking", "chronic", "Diabetes", "HDLC", "LDLC", "PTINR"
]

# Streamlit user interface
st.title("急性冠状动脉综合征患者1年内发生心血管不良事件风险预测")

# Age: categorical selection
Age = st.number_input("年龄 (岁):", min_value=0, max_value=100, value=45)

# Drinking: categorical selection
Drinking = st.selectbox("是否饮酒:", options=[0, 1], format_func=lambda x: "否" if x == 0 else "是")

# chronic: categorical selection
chronic = st.selectbox("是否合并其他慢性疾病:", options=[0, 1], format_func=lambda x: "否" if x == 0 else "是")

# Diabetes: categorical selection
Diabetes = st.selectbox("是否患有糖尿病:", options=[0, 1], format_func=lambda x: "否" if x == 0 else "是")

# HDLC: numerical input
HDLC = st.number_input("高密度脂蛋白胆固醇 (HDL-C, mmol/L):", min_value=0.0, max_value=50.0, value=1.2, step=0.1)

# LDLC: numerical input
LDLC = st.number_input("低密度脂蛋白胆固醇 (LDL-C, mmol/L):", min_value=0.0, max_value=50.0, value=3.0, step=0.1)

# PTINR: numerical input
PTINR = st.number_input("凝血相关指标 (PTINR):", min_value=0, max_value=10000, value=500)

# Process inputs and make predictions
feature_values = [Age, Drinking, chronic, Diabetes, HDLC, LDLC, PTINR]
features = np.array([feature_values], dtype=float)

if st.button("预测"):
    try:
        # Predict class and probabilities
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]

        # Display prediction results
        st.write(f"**预测类别:** {predicted_class}")
        st.write(f"**预测概率:** {predicted_proba}")

        # Generate advice based on prediction results
        # 获取发生MACE的概率 (类别1)
        probability_mace = predicted_proba[1] * 100
        # 获取不发生MACE的概率 (类别0)
        probability_no_mace = predicted_proba[0] * 100

        if predicted_class == 1:
            advice = (
                "根据模型预测，您在1年内发生心血管不良事件 (MACE) 的风险较高。\n"
                f"模型预测的发病概率为 {probability_mace:.1f}%。\n"
                "强烈建议您与主治医生详细讨论此结果，并采取积极的干预措施，如调整药物治疗、改变生活方式等。"
            )
        else:
            advice = (
                "根据模型预测，您在1年内发生心血管不良事件 (MACE) 的风险较低。\n"
                f"模型预测的无事件概率为 {probability_no_mace:.1f}%。\n"
                "建议您继续保持健康的生活方式，并遵医嘱定期复查。"
            )

        st.info(advice) # 使用 st.info 给建议一个信息框样式

        # Calculate SHAP values and display waterfall plot (more suitable for single predictions in Streamlit)
        explainer = shap.TreeExplainer(model)
        shap_values_all = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
        expected_value_all = explainer.expected_value

        # --- 处理 SHAP 值和期望值 (适用于二分类) ---
        # 对于二分类，TreeExplainer 通常返回 MACE=1 的SHAP值
        # 但有时会返回 [shap_values_for_class_0, shap_values_for_class_1]
        if isinstance(shap_values_all, list):
            shap_values = shap_values_all[1] # 取 MACE=1 的SHAP值
            expected_val = expected_value_all[1] # 取 MACE=1 的期望值
        else:
            shap_values = shap_values_all
            # 检查 expected_value 是否也是列表
            if isinstance(expected_value_all, list):
                expected_val = expected_value_all[1]
            else:
                expected_val = expected_value_all

        # Create and display SHAP waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        # 使用 shap.Explanation 对象以提高兼容性
        shap.waterfall_plot(shap.Explanation(values=shap_values, base_values=expected_val, feature_names=feature_names), ax=ax, show=False)
        ax.set_title("SHAP Waterfall Plot for this Prediction")
        plt.tight_layout()

        # Save and display the plot
        plt.savefig("shap_waterfall_plot.png", bbox_inches='tight', dpi=300)  # 降低DPI以避免保存时间过长
        plt.close(fig) # 关闭图形以释放内存
        st.image("shap_waterfall_plot.png", caption="SHAP预测解释图", use_column_width=True)

    except Exception as e_pred:
        st.error(f"预测或生成解释图时发生错误: {e_pred}")