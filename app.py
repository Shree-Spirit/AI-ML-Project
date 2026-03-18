import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import json

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("AIzaSyBlTf8NVOCAEcX90XEzJJSfHC295yJ-pVI"))
model = genai.GenerativeModel('gemini-1.5-flash')


st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


st.markdown("""
    <style>
        .main { background-color: #f5f7fa; }
        .title-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            color: white;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        .title-box h1 { font-size: 2.8rem; margin: 0; font-weight: 800; }
        .title-box p  { font-size: 1.2rem; margin-top: 10px; opacity: 0.95; }
        .ai-badge {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
            margin-top: 10px;
        }
        .prediction-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            color: white;
            margin: 20px 0;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        }
        .prediction-card h2 { font-size: 3rem; margin: 10px 0; font-weight: 800; }
        .prediction-card p  { font-size: 1.1rem; opacity: 0.9; }
        .insight-box {
            background: white;
            border-left: 5px solid #667eea;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .insight-box h3 { color: #667eea; margin-top: 0; }
        .chat-message {
            background: white;
            border-radius: 10px;
            padding: 15px 20px;
            margin: 10px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: 20%;
        }
        .ai-message {
            background: #f8f9fa;
            margin-right: 20%;
        }
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
        .metric-card h3 { color: #667eea; font-size: 2rem; margin: 0; }
        .metric-card p { color: #666; margin-top: 8px; }
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 40px;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            width: 100%;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        .section-header {
            font-size: 1.4rem;
            font-weight: 700;
            color: #667eea;
            margin: 25px 0 15px 0;
        }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown("""
    <div class="title-box">
        <h1>🏠 AI House Price Predictor</h1>
        <p>Powered by Google Gemini AI · Intelligent Real Estate Analysis</p>
        <div class="ai-badge">🧠 AI-Powered Brain</div>
    </div>
""", unsafe_allow_html=True)



@st.cache_data
def generate_market_data():
    if os.path.exists("housing_data.csv"):
        return pd.read_csv("housing_data.csv")

    np.random.seed(42)
    n = 500
    locations = ["Urban", "Suburban", "Rural"]
    conditions = ["Excellent", "Good", "Fair", "Poor"]

    data = []
    for _ in range(n):
        area = np.random.randint(800, 4500)
        bedrooms = np.random.randint(2, 6)
        bathrooms = np.random.randint(1, 4)
        floors = np.random.randint(1, 3)
        age = np.random.randint(0, 40)
        garage = np.random.randint(0, 3)
        location = np.random.choice(locations)
        condition = np.random.choice(conditions)

        # Price calculation
        base_price = area * 150
        if location == "Urban":
            base_price += 100000
        elif location == "Suburban":
            base_price += 50000

        if condition == "Excellent":
            base_price *= 1.2
        elif condition == "Good":
            base_price *= 1.05
        elif condition == "Fair":
            base_price *= 0.95
        else:
            base_price *= 0.85

        base_price += bedrooms * 20000 + bathrooms * 15000 - age * 3000
        price = max(100000, base_price + np.random.randint(-30000, 30000))

        data.append([area, bedrooms, bathrooms, floors, age,
                    garage, location, condition, int(price)])

    df = pd.DataFrame(data, columns=["Area_sqft", "Bedrooms", "Bathrooms", "Floors",
                                     "Age_years", "Garage", "Location", "Condition", "Price"])
    df.to_csv("housing_data.csv", index=False)
    return df


market_data = generate_market_data()



def predict_price_with_ai(area, bedrooms, bathrooms, floors, age, garage, location, condition):
    """Use Gemini AI to predict house price and provide insights"""

    # Get market stats for context
    similar_houses = market_data[
        (market_data['Location'] == location) &
        (market_data['Bedrooms'] == bedrooms)
    ]
    avg_price = similar_houses['Price'].mean() if len(
        similar_houses) > 0 else market_data['Price'].mean()

    prompt = f"""You are an expert real estate AI analyst. Based on the following house details, predict a realistic market price and provide detailed insights.

House Details:
- Area: {area} square feet
- Bedrooms: {bedrooms}
- Bathrooms: {bathrooms}
- Floors: {floors}
- Age: {age} years
- Garage Spaces: {garage}
- Location: {location}
- Condition: {condition}

Market Context:
- Average price for similar houses in {location}: ${avg_price:,.0f}
- Total market data points: {len(market_data)}

Please provide your response in the following JSON format:
{{
    "predicted_price": <numeric value>,
    "confidence_level": "<High/Medium/Low>",
    "price_breakdown": {{
        "base_value": <value>,
        "location_premium": <value>,
        "condition_adjustment": <value>,
        "age_depreciation": <value>
    }},
    "key_factors": [
        "factor 1 explanation",
        "factor 2 explanation",
        "factor 3 explanation"
    ],
    "market_position": "<Above/At/Below> market average",
    "recommendations": [
        "recommendation 1",
        "recommendation 2",
        "recommendation 3"
    ],
    "investment_advice": "brief investment advice"
}}

Be realistic with pricing based on the market context provided."""

    try:
        response = model.generate_content(prompt)
        # Extract JSON from response
        response_text = response.text.strip()
        if "```json" in response_text:
            response_text = response_text.split(
                "```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split(
                "```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        return result
    except Exception as e:
        st.error(f"AI Error: {str(e)}")
        # Fallback to simple calculation
        return {
            "predicted_price": int(avg_price),
            "confidence_level": "Medium",
            "price_breakdown": {},
            "key_factors": ["Using fallback calculation due to AI error"],
            "market_position": "At market average",
            "recommendations": ["Please try again"],
            "investment_advice": "Based on market averages"
        }


def chat_with_ai(question, house_data=None):
    """Chat with AI about real estate"""
    context = ""
    if house_data:
        context = f"\nCurrent house being discussed: {house_data}"

    prompt = f"""You are a helpful real estate AI assistant. Answer the following question about real estate and housing markets.
    {context}
    
    Question: {question}
    
    Provide a helpful, concise answer."""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"


# ─────────────────────────────────────────
#  INITIALIZE SESSION STATE
# ─────────────────────────────────────────
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# ─────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 AI Prediction", "💬 Chat with AI", "📊 Market Data", "ℹ️ About"])

# ════════════════════════════════════════
#  TAB 1 — AI PREDICTION
# ════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Enter House Details</p>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        area = st.number_input(
            "📐 Area (sq ft)", min_value=500, max_value=10000, value=2000, step=100)
        bedrooms = st.slider("🛏 Bedrooms", min_value=1, max_value=8, value=3)
        bathrooms = st.slider("🚿 Bathrooms", min_value=1, max_value=5, value=2)

    with col2:
        floors = st.slider("🏢 Floors", min_value=1, max_value=4, value=2)
        age = st.slider("📅 Age (years)", min_value=0, max_value=50, value=5)
        garage = st.slider("🚗 Garage Spaces", min_value=0,
                           max_value=4, value=2)

    with col3:
        location = st.selectbox("📍 Location", ["Urban", "Suburban", "Rural"])
        condition = st.selectbox(
            "🏗 Condition", ["Excellent", "Good", "Fair", "Poor"])

    st.markdown("---")

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🧠 Get AI Prediction", use_container_width=True):
            with st.spinner("🤖 AI is analyzing your property..."):
                result = predict_price_with_ai(
                    area, bedrooms, bathrooms, floors, age, garage, location, condition)
                st.session_state.last_prediction = {
                    'result': result,
                    'inputs': f"{area}sqft, {bedrooms}bed, {bathrooms}bath, {location}, {condition}"
                }

    # Display results
    if st.session_state.last_prediction:
        result = st.session_state.last_prediction['result']

        # Price prediction card
        st.markdown(f"""
            <div class="prediction-card">
                <p style="font-size:1rem; opacity:0.9; margin:0;">AI Predicted Price</p>
                <h2>${result['predicted_price']:,.0f}</h2>
                <p>Confidence: {result['confidence_level']} • {result['market_position']}</p>
            </div>
        """, unsafe_allow_html=True)

        # Key Factors
        st.markdown(
            '<p class="section-header">🎯 Key Price Factors</p>', unsafe_allow_html=True)
        for factor in result['key_factors']:
            st.markdown(f"""
                <div class="insight-box">
                    <p style="margin:0;">• {factor}</p>
                </div>
            """, unsafe_allow_html=True)

        # Recommendations
        col_rec1, col_rec2 = st.columns(2)

        with col_rec1:
            st.markdown(
                '<p class="section-header">💡 Recommendations</p>', unsafe_allow_html=True)
            for rec in result['recommendations']:
                st.markdown(f"✓ {rec}")

        with col_rec2:
            st.markdown(
                '<p class="section-header">📈 Investment Advice</p>', unsafe_allow_html=True)
            st.info(result['investment_advice'])

        # Price Breakdown (if available)
        if result.get('price_breakdown'):
            st.markdown(
                '<p class="section-header">💰 Price Breakdown</p>', unsafe_allow_html=True)
            breakdown_cols = st.columns(len(result['price_breakdown']))
            for idx, (key, value) in enumerate(result['price_breakdown'].items()):
                with breakdown_cols[idx]:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>${value:,.0f}</h3>
                            <p>{key.replace('_', ' ').title()}</p>
                        </div>
                    """, unsafe_allow_html=True)

# ════════════════════════════════════════
#  TAB 2 — CHAT WITH AI
# ════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">💬 Ask AI About Real Estate</p>',
                unsafe_allow_html=True)

    # Display chat history
    for chat in st.session_state.chat_history:
        if chat['role'] == 'user':
            st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {chat['message']}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message ai-message">
                    <strong>🤖 AI:</strong> {chat['message']}
                </div>
            """, unsafe_allow_html=True)

    # Chat input
    st.markdown("---")
    user_question = st.text_input("Ask me anything about real estate, home prices, or the housing market:",
                                  placeholder="e.g., How does location affect house prices?")

    col_chat1, col_chat2, col_chat3 = st.columns([1, 2, 1])
    with col_chat2:
        if st.button("💬 Ask AI", use_container_width=True):
            if user_question:
                st.session_state.chat_history.append(
                    {'role': 'user', 'message': user_question})

                with st.spinner("🤖 AI is thinking..."):
                    house_context = st.session_state.last_prediction[
                        'inputs'] if st.session_state.last_prediction else None
                    ai_response = chat_with_ai(user_question, house_context)
                    st.session_state.chat_history.append(
                        {'role': 'ai', 'message': ai_response})

                st.rerun()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# ════════════════════════════════════════
#  TAB 3 — MARKET DATA
# ════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">📊 Market Overview</p>',
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
            <div class="metric-card">
                <h3>{len(market_data)}</h3>
                <p>Properties</p>
            </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
            <div class="metric-card">
                <h3>${market_data['Price'].mean():,.0f}</h3>
                <p>Avg Price</p>
            </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
            <div class="metric-card">
                <h3>${market_data['Price'].median():,.0f}</h3>
                <p>Median Price</p>
            </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
            <div class="metric-card">
                <h3>{int(market_data['Area_sqft'].mean())}</h3>
                <p>Avg Size (sqft)</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">📈 Market Analysis</p>',
                unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        avg_by_location = market_data.groupby(
            'Location')['Price'].mean().sort_values(ascending=False)
        bars = ax.bar(avg_by_location.index, avg_by_location.values,
                      color=['#667eea', '#764ba2', '#f093fb'])
        ax.set_title("Average Price by Location",
                     fontweight='bold', fontsize=14)
        ax.set_ylabel("Price ($)")
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    with chart_col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        avg_by_condition = market_data.groupby(
            'Condition')['Price'].mean().sort_values(ascending=False)
        bars = ax.bar(avg_by_condition.index, avg_by_condition.values,
                      color=['#667eea', '#764ba2', '#f093fb', '#4facfe'])
        ax.set_title("Average Price by Condition",
                     fontweight='bold', fontsize=14)
        ax.set_ylabel("Price ($)")
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown('<p class="section-header">🗂️ Sample Market Data</p>',
                unsafe_allow_html=True)
    st.dataframe(market_data.head(
        10), use_container_width=True, hide_index=True)

# ════════════════════════════════════════
#  TAB 4 — ABOUT
# ════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">🧠 About This AI System</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <h3>🤖 Powered by Google Gemini AI</h3>
        <p>This application uses Google's Gemini 1.5 Flash model as the central intelligence for house price prediction and real estate analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <h3>🎯 How It Works</h3>
        <ul>
            <li><strong>AI-Driven Analysis:</strong> Gemini AI analyzes house features using advanced reasoning</li>
            <li><strong>Market Context:</strong> Considers local market data for realistic predictions</li>
            <li><strong>Intelligent Insights:</strong> Provides detailed breakdowns and recommendations</li>
            <li><strong>Interactive Chat:</strong> Ask any real estate questions to the AI</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <h3>🛠️ Technology Stack</h3>
        <ul>
            <li><strong>AI Brain:</strong> Google Gemini 1.5 Flash API</li>
            <li><strong>Frontend:</strong> Streamlit</li>
            <li><strong>Language:</strong> Python</li>
            <li><strong>Data:</strong> Synthetic market dataset (CSV)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <h3>💡 Key Features</h3>
        <ul>
            <li>✅ AI-powered price predictions with confidence levels</li>
            <li>✅ Detailed price breakdowns and factor analysis</li>
            <li>✅ Personalized recommendations to increase property value</li>
            <li>✅ Interactive chat for real estate questions</li>
            <li>✅ Market data visualization and trends</li>
            <li>✅ Investment advice based on AI analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("""
    <p style='text-align:center; color:#888; font-size:0.9rem;'>
        🏠 AI House Price Predictor · Powered by Google Gemini AI · Built with Python & Streamlit
    </p>
""", unsafe_allow_html=True)
