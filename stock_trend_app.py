import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ================= APP CONFIG =================
st.set_page_config(page_title="MACD + RSI Stock Trend Predictor", page_icon="📈", layout="wide")

# ===== DARK APP WITH SKY BLUE SIDEBAR AND GREEN HEADER =====
st.markdown(
    """
    <style>
    /* Main background and text */
    .stApp {
        background-color: #1E1E1E 
        color: #ffffff;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color:  #C0C0C0;
        color: #000000;
    }
    [data-testid="stSidebar"]::-webkit-scrollbar {
        width: 8px;
    }
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb {
        background-color: #555;
        border-radius: 4px;
    }
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
        background-color: #777;
    }

    /* DataFrame background (dark tables) */
    .stDataFrame {
        background-color: #1e1e1e;
        color: #ffffff;
        border-radius: 8px;
        padding: 8px;
    }

    /* Remove default dividers */
    .stDivider {
        border-top: none;
        margin: 0;
    }

    /* Header styling */
    header[data-testid="stHeader"] {
        background-color: #4169E1 !important;
    }
    header[data-testid="stHeader"] * {
        color: white !important;
    }
    header[data-testid="stHeader"] button {
        background-color: #4169E1 !important;
        color: white !important;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("My Buddy - Stock Trend Analyser")

# ================= SIDEBAR =================
st.sidebar.header("🔧 Controls")
symbol = st.sidebar.text_input("Stock Symbol (e.g., TCS.NS, INFY.NS)", key="stock_symbol_input")
period = st.sidebar.selectbox("Select Data Period", ["15d", "1mo", "3mo", "6mo", "1y"], index=1)
predict_btn = st.sidebar.button("Analyse Trend", use_container_width=True)

# ================= HELPER: RSI =================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=series.index).rolling(window=period).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# ================= MAIN =================
if predict_btn:
    if not symbol:
        st.warning("Please enter a stock symbol after stock name (.NS) in the sidebar.")
    else:
        try:
            data = yf.download(symbol, period=period, interval="1d")
            if data.empty:
                st.error("No data found as (.NS) after stock name.")
            else:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                # Calculate Indicators
                data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
                data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = data['EMA12'] - data['EMA26']
                data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                data['RSI'] = calculate_rsi(data['Close'], period=14)
                valid = data[['MACD', 'Signal', 'RSI']].dropna()

                if len(valid) == 0:
                    st.error("❌ Not enough data to compute indicators. Try a longer period.")
                else:
                    latest = valid.iloc[-1]
                    latest_macd = latest['MACD']
                    latest_signal = latest['Signal']
                    latest_rsi = latest['RSI']

                    # ===== METRICS =====
                    st.markdown("### 📊 Key Metrics")
                    col1, col2 = st.columns(2)
                    col1.metric("💰 Last Close", f"{data['Close'].iloc[-1]:.2f}")
                   # col2.metric("📈 MACD", f"{latest_macd:.4f}")
                    col2.metric("📊 RSI", f"{latest_rsi:.2f}")

                    # ===== TREND MESSAGE =====
                    if len(valid) >= 2:
                        prev = valid.iloc[-2]
                        if prev['MACD'] < prev['Signal'] and latest_macd > latest_signal:
                            st.success("📈 Bullish crossover detected → Possible UP trend.")
                        elif prev['MACD'] > prev['Signal'] and latest_macd < latest_signal:
                            st.error("📉 Bearish crossover detected → Possible DOWN trend.")
                        else:
                            st.info("➡️ No crossover. Trend continuing.")

                    if latest_rsi > 70:
                        st.warning(f" RSI {latest_rsi:.2f} → Overbuying")
                    elif latest_rsi < 30:
                        st.success(f" RSI {latest_rsi:.2f} → Overselling")
                    else:
                        st.info(f" RSI {latest_rsi:.2f} → Neutral")

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ===== LAST 7 DAYS TABLE =====
                    st.subheader("📅 Last 7 Days Stock Data")
                    data_reset = data.copy().reset_index()
                    data_reset['Date'] = data_reset['Date'].dt.strftime('%Y-%m-%d')

                    st.dataframe(data_reset[['Date', 'Open', 'High', 'Low', 'Close', 'RSI']].tail(7))

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ===== CSV DOWNLOAD BUTTON =====
                    csv_data = data_reset[['Date', 'Open', 'High', 'Low', 'Close', 'RSI']].to_csv(index=False).encode('utf-8')
                    st.download_button(
                    label="📥 Download Full Data",
                    data=csv_data,
                    file_name=f"{symbol}_stock_data.csv",
                    mime="text/csv",
                    use_container_width=True
                    )

                    # ===== COMMON CONFIG FOR CHARTS =====
                    chart_config = {
                        "responsive": True,
                        "displaylogo": False,
                        "displayModeBar": True,
                        "zoom": True,
                       # "modeBarButtonsToRemove": [
                            #"zoom" "pan", "select", "lasso2d",
                           # "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"
                       # ],
                        "toImageButtonOptions": {"format": "png", "filename": symbol, "scale": 2}
                    }

                    # ===== CANDLESTICK CHART =====
                    st.subheader("📊 Candlestick Chart")
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=data.index, open=data['Open'], high=data['High'],
                        low=data['Low'], close=data['Close'],
                        increasing_line_color='green', decreasing_line_color='red',
                        line=dict(width=1), whiskerwidth=0.5
                    ))
                    vol_colors = np.where(data['Close'] >= data['Open'], 'rgba(0,255,0,0.3)','rgba(255,0,0,0.3)')
                    fig.add_trace(go.Bar(
                        x=data.index, y=data['Volume'], name='Volume',
                        marker=dict(color=vol_colors), yaxis='y2', opacity=1
                    ))
                    fig.update_layout(
                        xaxis_rangeslider_visible=True,
                        template="plotly_dark",
                        height=500,
                        dragmode=False,
                        title=f"{symbol} Candlestick Chart",
                        yaxis=dict(title="Price Bar", automargin=True),
                        yaxis2=dict(overlaying='y', side='right', visible=False),
                        margin=dict(l=10,r=10,t=40,b=20),
                        autosize=True
                    )
                    st.plotly_chart(fig, use_container_width=True, config=chart_config)

                    # ===== MACD CHART =====
                    st.subheader("📉 MACD Indicator")
                    macd_fig = go.Figure()
                    macd_hist = data['MACD'] - data['Signal']
                    macd_fig.add_trace(go.Bar(x=data.index, y=macd_hist, name='Histogram',
                                              marker_color=np.where(macd_hist>=0,'green','red'), opacity=1))
                    macd_fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines',
                                                  name='MACD', line=dict(color='orange', width=2)))
                    macd_fig.add_trace(go.Scatter(x=data.index, y=data['Signal'], mode='lines',
                                                  name='Signal', line=dict(color='purple', width=2, dash='dot')))
                    macd_fig.update_layout(template="plotly_dark", height=400, dragmode=False,
                                           title="MACD (12,26) & Signal (9)",
                                           margin=dict(l=10,r=10,t=40,b=20), autosize=True)
                    st.plotly_chart(macd_fig, use_container_width=True, config=chart_config)

                    # ===== RSI CHART =====
                    st.subheader("📈 RSI Indicator (14)")
                    rsi_fig = go.Figure()
                    rsi_fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines+markers',
                                                 name='RSI', line=dict(color='cyan', width=2)))
                    rsi_fig.add_trace(go.Scatter(x=data.index, y=[70]*len(data), mode='lines',
                                                 name='Overbought (70)', line=dict(color='red', dash='dash')))
                    rsi_fig.add_trace(go.Scatter(x=data.index, y=[30]*len(data), mode='lines',
                                                 name='Oversold (30)', line=dict(color='green', dash='dash')))
                    rsi_fig.update_layout(template="plotly_dark", height=350, dragmode=False,
                                          title="RSI (14) with Overbuying/Overselling Levels",
                                          yaxis=dict(range=[0,100], automargin=True),
                                          margin=dict(l=10,r=10,t=40,b=20), autosize=True)
                    st.plotly_chart(rsi_fig, use_container_width=True, config=chart_config)

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
else:
    st.info("👈 Enter a stock symbol and click **Predict Trend**.")
























