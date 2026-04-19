import streamlit as st
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from keras.models import load_model

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# ── Load model & encoders ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = load_model('model.h5')
    with open('label_encoder_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        ohe_geo = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, le_gender, ohe_geo, scaler

@st.cache_data
def load_data():
    """Load dataset for insights. Falls back to synthetic data if CSV not found."""
    try:
        df = pd.read_csv('Churn_Modelling.csv')
        return df
    except FileNotFoundError:
        # Synthetic fallback so the Insights tab always renders
        np.random.seed(42)
        n = 1000
        geo = np.random.choice(['France', 'Germany', 'Spain'], n, p=[0.5, 0.25, 0.25])
        gender = np.random.choice(['Male', 'Female'], n)
        age = np.random.randint(18, 92, n)
        exited = (
            (age > 45).astype(int) * 0.4
            + (geo == 'Germany').astype(int) * 0.3
            + np.random.rand(n) * 0.3
        ) > 0.5
        df = pd.DataFrame({
            'CreditScore':      np.random.randint(350, 850, n),
            'Geography':        geo,
            'Gender':           gender,
            'Age':              age,
            'Tenure':           np.random.randint(0, 11, n),
            'Balance':          np.round(np.random.uniform(0, 250000, n), 2),
            'NumOfProducts':    np.random.randint(1, 5, n),
            'HasCrCard':        np.random.randint(0, 2, n),
            'IsActiveMember':   np.random.randint(0, 2, n),
            'EstimatedSalary':  np.round(np.random.uniform(10000, 200000, n), 2),
            'Exited':           exited.astype(int),
        })
        return df

model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()
df = load_data()

# ── Colour palette ─────────────────────────────────────────────────────────────
CLR_STAY  = '#1D9E75'   # teal
CLR_CHURN = '#D85A30'   # coral
CLR_SEQ   = px.colors.sequential.Teal

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_predict, tab_insights = st.tabs(["🔮 Churn Predictor", "📊 Dataset Insights"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.title('Customer Churn Prediction')
    st.markdown("Fill in the customer details below and click **Predict** to see the churn probability.")

    col1, col2, col3 = st.columns(3)

    with col1:
        geography    = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
        gender       = st.selectbox('Gender', label_encoder_gender.classes_)
        age          = st.slider('Age', 18, 92, 35)
        balance      = st.number_input('Balance', min_value=0.0, value=0.0, step=1000.0)

    with col2:
        credit_score      = st.number_input('Credit Score', min_value=300, max_value=900, value=650)
        estimated_salary  = st.number_input('Estimated Salary', min_value=0.0, value=50000.0, step=1000.0)
        tenure            = st.slider('Tenure (years)', 0, 10, 3)

    with col3:
        num_of_products   = st.slider('Number of Products', 1, 4, 1)
        has_cr_card       = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
        is_active_member  = st.selectbox('Is Active Member',  [0, 1], format_func=lambda x: 'Yes' if x else 'No')

    if st.button('Predict Churn', type='primary', use_container_width=True):
        input_data = pd.DataFrame({
            'CreditScore':    [credit_score],
            'Gender':         [label_encoder_gender.transform([gender])[0]],
            'Age':            [age],
            'Tenure':         [tenure],
            'Balance':        [balance],
            'NumOfProducts':  [num_of_products],
            'HasCrCard':      [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary':[estimated_salary],
        })

        geo_encoded    = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded,
                                      columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
        input_data     = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        input_scaled   = scaler.transform(input_data)

        prob = float(model.predict(input_scaled)[0][0])

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn Probability", f"{prob:.1%}")
        m2.metric("Retention Probability", f"{1-prob:.1%}")
        m3.metric("Risk Level",
                  "🔴 High" if prob > 0.7 else ("🟡 Medium" if prob > 0.4 else "🟢 Low"))

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = prob * 100,
            delta = {'reference': 50, 'suffix': '%'},
            title = {'text': "Churn Probability (%)"},
            gauge = {
                'axis':  {'range': [0, 100]},
                'bar':   {'color': CLR_CHURN if prob > 0.5 else CLR_STAY},
                'steps': [
                    {'range': [0,  40], 'color': '#E1F5EE'},
                    {'range': [40, 70], 'color': '#FAEEDA'},
                    {'range': [70, 100],'color': '#FAECE7'},
                ],
                'threshold': {'line': {'color': 'black', 'width': 3}, 'value': 50},
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if prob > 0.5:
            st.error(f"⚠️ This customer is **likely to churn** ({prob:.1%} probability).")
        else:
            st.success(f"✅ This customer is **not likely to churn** ({prob:.1%} probability).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – DATASET INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.title('📊 Dataset Insights')

    total   = len(df)
    churned = df['Exited'].sum()
    rate    = churned / total

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers",  f"{total:,}")
    k2.metric("Churned",          f"{churned:,}")
    k3.metric("Retained",         f"{total - churned:,}")
    k4.metric("Churn Rate",       f"{rate:.1%}")

    st.divider()

    # ── Row 1: Churn distribution + Geography breakdown ───────────────────────
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        churn_counts = df['Exited'].value_counts().reset_index()
        churn_counts.columns = ['Exited', 'Count']
        churn_counts['Label'] = churn_counts['Exited'].map({0: 'Retained', 1: 'Churned'})

        fig_pie = px.pie(
            churn_counts, values='Count', names='Label',
            title='Overall Churn Distribution',
            color='Label',
            color_discrete_map={'Retained': CLR_STAY, 'Churned': CLR_CHURN},
            hole=0.45,
        )
        fig_pie.update_traces(textposition='outside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=False, margin=dict(t=50, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with r1c2:
        geo_churn = (
            df.groupby('Geography')['Exited']
            .agg(['sum', 'count'])
            .reset_index()
            .rename(columns={'sum': 'Churned', 'count': 'Total'})
        )
        geo_churn['Churn Rate (%)'] = (geo_churn['Churned'] / geo_churn['Total'] * 100).round(1)

        fig_geo = px.bar(
            geo_churn, x='Geography', y='Churn Rate (%)',
            title='Churn Rate by Geography',
            color='Churn Rate (%)',
            color_continuous_scale='Oranges',
            text='Churn Rate (%)',
        )
        fig_geo.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_geo.update_layout(coloraxis_showscale=False, margin=dict(t=50, b=10))
        st.plotly_chart(fig_geo, use_container_width=True)

    # ── Row 2: Age distribution + Gender churn ────────────────────────────────
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        fig_age = px.histogram(
            df, x='Age', color=df['Exited'].map({0: 'Retained', 1: 'Churned'}),
            barmode='overlay', nbins=30,
            title='Age Distribution by Churn Status',
            labels={'color': 'Status'},
            color_discrete_map={'Retained': CLR_STAY, 'Churned': CLR_CHURN},
            opacity=0.75,
        )
        fig_age.update_layout(margin=dict(t=50, b=10))
        st.plotly_chart(fig_age, use_container_width=True)

    with r2c2:
        gender_churn = (
            df.groupby('Gender')['Exited']
            .agg(['sum', 'count'])
            .reset_index()
            .rename(columns={'sum': 'Churned', 'count': 'Total'})
        )
        gender_churn['Retained'] = gender_churn['Total'] - gender_churn['Churned']

        fig_gender = go.Figure()
        fig_gender.add_bar(name='Retained', x=gender_churn['Gender'],
                           y=gender_churn['Retained'], marker_color=CLR_STAY)
        fig_gender.add_bar(name='Churned',  x=gender_churn['Gender'],
                           y=gender_churn['Churned'],  marker_color=CLR_CHURN)
        fig_gender.update_layout(
            barmode='group', title='Churn Count by Gender',
            margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    # ── Row 3: Balance distribution + Credit score vs churn ──────────────────
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        fig_bal = px.box(
            df, x=df['Exited'].map({0: 'Retained', 1: 'Churned'}),
            y='Balance', color=df['Exited'].map({0: 'Retained', 1: 'Churned'}),
            title='Account Balance by Churn Status',
            labels={'x': 'Status', 'color': 'Status'},
            color_discrete_map={'Retained': CLR_STAY, 'Churned': CLR_CHURN},
        )
        fig_bal.update_layout(showlegend=False, margin=dict(t=50, b=10))
        st.plotly_chart(fig_bal, use_container_width=True)

    with r3c2:
        fig_cs = px.histogram(
            df, x='CreditScore',
            color=df['Exited'].map({0: 'Retained', 1: 'Churned'}),
            barmode='overlay', nbins=25,
            title='Credit Score Distribution by Churn Status',
            labels={'color': 'Status'},
            color_discrete_map={'Retained': CLR_STAY, 'Churned': CLR_CHURN},
            opacity=0.75,
        )
        fig_cs.update_layout(margin=dict(t=50, b=10))
        st.plotly_chart(fig_cs, use_container_width=True)

    # ── Row 4: Products heatmap + Active member / credit card ─────────────────
    r4c1, r4c2 = st.columns(2)

    with r4c1:
        prod_churn = (
            df.groupby('NumOfProducts')['Exited']
            .agg(['sum', 'count'])
            .reset_index()
            .rename(columns={'sum': 'Churned', 'count': 'Total'})
        )
        prod_churn['Churn Rate (%)'] = (prod_churn['Churned'] / prod_churn['Total'] * 100).round(1)

        fig_prod = px.bar(
            prod_churn, x='NumOfProducts', y='Churn Rate (%)',
            title='Churn Rate by Number of Products',
            color='Churn Rate (%)',
            color_continuous_scale='RdYlGn_r',
            text='Churn Rate (%)',
        )
        fig_prod.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_prod.update_layout(
            xaxis_title='Number of Products',
            coloraxis_showscale=False,
            margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig_prod, use_container_width=True)

    with r4c2:
        flags = pd.DataFrame({
            'Segment': ['Active Member', 'Inactive Member', 'Has Credit Card', 'No Credit Card'],
            'Churn Rate (%)': [
                df[df['IsActiveMember'] == 1]['Exited'].mean() * 100,
                df[df['IsActiveMember'] == 0]['Exited'].mean() * 100,
                df[df['HasCrCard']      == 1]['Exited'].mean() * 100,
                df[df['HasCrCard']      == 0]['Exited'].mean() * 100,
            ]
        })
        flags['Churn Rate (%)'] = flags['Churn Rate (%)'].round(1)

        fig_flags = px.bar(
            flags, x='Churn Rate (%)', y='Segment',
            orientation='h',
            title='Churn Rate: Activity & Credit Card Status',
            color='Churn Rate (%)',
            color_continuous_scale='Oranges',
            text='Churn Rate (%)',
        )
        fig_flags.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_flags.update_layout(coloraxis_showscale=False, margin=dict(t=50, b=10))
        st.plotly_chart(fig_flags, use_container_width=True)

    # ── Row 5: Tenure + Salary ────────────────────────────────────────────────
    r5c1, r5c2 = st.columns(2)

    with r5c1:
        tenure_churn = (
            df.groupby('Tenure')['Exited']
            .mean()
            .reset_index()
            .rename(columns={'Exited': 'Churn Rate'})
        )
        tenure_churn['Churn Rate (%)'] = (tenure_churn['Churn Rate'] * 100).round(1)

        fig_ten = px.line(
            tenure_churn, x='Tenure', y='Churn Rate (%)',
            title='Churn Rate Across Tenure',
            markers=True,
        )
        fig_ten.update_traces(line_color=CLR_CHURN, marker_color=CLR_CHURN, marker_size=8)
        fig_ten.update_layout(margin=dict(t=50, b=10))
        st.plotly_chart(fig_ten, use_container_width=True)

    with r5c2:
        fig_sal = px.box(
            df, x=df['Exited'].map({0: 'Retained', 1: 'Churned'}),
            y='EstimatedSalary',
            color=df['Exited'].map({0: 'Retained', 1: 'Churned'}),
            title='Estimated Salary by Churn Status',
            labels={'x': 'Status', 'color': 'Status'},
            color_discrete_map={'Retained': CLR_STAY, 'Churned': CLR_CHURN},
        )
        fig_sal.update_layout(showlegend=False, margin=dict(t=50, b=10))
        st.plotly_chart(fig_sal, use_container_width=True)

    # ── Row 6: Correlation heatmap ─────────────────────────────────────────────
    st.subheader('Feature Correlation Heatmap')
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance',
                'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                'EstimatedSalary', 'Exited']
    corr = df[num_cols].corr().round(2)

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title='Pearson Correlation Matrix',
        aspect='auto',
    )
    fig_corr.update_layout(margin=dict(t=50, b=10), height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Raw data (expandable) ─────────────────────────────────────────────────
    with st.expander("🔍 View Raw Dataset"):
        st.dataframe(df, use_container_width=True)
        st.caption(f"{len(df):,} rows × {len(df.columns)} columns")
