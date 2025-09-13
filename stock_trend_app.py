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
    /* ===== APP BACKGROUND ===== */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background-color: #87CEFA;  
        color: #000000;
    }
    [data-testid="stSidebar"]::-webkit-scrollbar {
        width: 8px;
    }
    [data-testid="stSidebar"]::-webkit-scrollbar-track {
        background: #87CEFA;
    }
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb {
        background-color: #555555;
        border-radius: 4px;
        border: 1px solid #87CEFA;
    }

    /* Dataframe table background */
    .stDataFrame div[data-testid="stHorizontalBlock"] {
        background-color: #1e1e1e;
        color: #ffffff;
    }

    /* Remove default divider */
    .stDivider {
        border-top: none;
        margin: 0;
    }

    /* ===== TOP HEADER ===== */
    header[data-testid="stHeader"] {
        background-color: #28a745 !important;  
    }
    header[data-testid="stHeader"] * {
        color: white !important;  
    }
    header[data-testid="stHeader"] button {
        background-color: #28a745 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 MACD + RSI Based Stock Trend Predictor")

# ================= SIDEBAR =================
st.sidebar.header("🔧 Controls")
symbol = st.sidebar.text_input("📌 Stock Symbol (e.g., TCS.NS, INFY.NS)", key="stock_symbol_input")
period = st.sidebar.selectbox("📅 Select Data Period", ["15d", "1mo", "3mo", "6mo", "1y"], index=1)
predict_btn = st.sidebar.button("🔮 Predict Trend", use_container_width=True)

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
        st.warning("⚠️ Please enter a stock symbol in the sidebar.")
    else:
        try:
            data = yf.download(symbol, period=period, interval="1d")
            if data.empty:
                st.error("❌ No data found. Try another symbol or longer period (e.g., 3mo).")
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
                    col1, col2, col3 = st.columns(3)
                    col1.metric("💰 Last Close", f"{data['Close'].iloc[-1]:.2f}")
                    col2.metric("📈 MACD", f"{latest_macd:.4f}")
                    col3.metric("📊 RSI", f"{latest_rsi:.2f}")

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
                        st.warning(f"⚠️ RSI {latest_rsi:.2f} → Overbought")
                    elif latest_rsi < 30:
                        st.success(f"💚 RSI {latest_rsi:.2f} → Oversold")
                    else:
                        st.info(f"📊 RSI {latest_rsi:.2f} → Neutral")

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ===== LAST 7 DAYS TABLE =====
                    st.subheader("📅 Last 7 Days Stock Data")
                    data_reset = data.copy().reset_index()
                    data_reset['Date'] = data_reset['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    st.dataframe(data_reset[['Date', 'Open', 'High', 'Low', 'Close', 'MACD', 'Signal', 'RSI']].tail(7))

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ===== CANDLESTICK CHART PROFESSIONAL & RESPONSIVE =====
                    st.subheader("📊 Candlestick Chart")
                    candle_width = max(0.1, min(0.5, 20 / max(1, len(data))))
                    chart_height = min(600, 400 + len(data)//2) # scale height based on data
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=data.index, open=data['Open'], high=data['High'],
                        low=data['Low'], close=data['Close'],
                        increasing_line_color='green', decreasing_line_color='red',
                        line=dict(width=1), whiskerwidth=0.5,
                        hovertext=[
                            f"<b>Date:</b> {d.strftime('%Y-%m-%d')}<br>"
                            f"<b>Open:</b> {o:.2f}<br>"
                            f"<b>High:</b> {h:.2f}<br>"
                            f"<b>Low:</b> {l:.2f}<br>"
                            f"<b>Close:</b> {c:.2f}<br>"
                            f"<b>Volume:</b> {v:,}"
                            for d,o,h,l,c,v in zip(data.index, data['Open'], data['High'], data['Low'], data['Close'], data['Volume'])
                        ],
                        hoverinfo="text"
                    ))

                    vol_colors = np.where(data['Close'] >= data['Open'], 'rgba(0,255,0,0.3)','rgba(255,0,0,0.3)')
                    fig.add_trace(go.Bar(
                        x=data.index, y=data['Volume'], name='Volume',
                        marker=dict(color=vol_colors), yaxis='y2', opacity=0.5,
                        hovertext=[f"<b>Date:</b> {d.strftime('%Y-%m-%d')}<br><b>Volume:</b> {v:,}" for d,v in zip(data.index, data['Volume'])],
                        hoverinfo="text"
                    ))

                    fig.update_layout(
                        xaxis_rangeslider_visible=False,
                        template="plotly_dark",
                        height=650,
                        title=f"{symbol} Candlestick Chart",
                        yaxis=dict(title="Price", automargin=True),
                        yaxis2=dict(overlaying='y', side='right', visible=False),
                        margin=dict(l=10,r=10,t=40,b=20),
                        autosize=True
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})

                    # ===== MACD CHART RESPONSIVE =====
                    st.subheader("📉 MACD Indicator")
                    macd_fig = go.Figure()
                    macd_hist = data['MACD'] - data['Signal']

                    macd_fig.add_trace(go.Bar(
                        x=data.index, y=macd_hist, name='Histogram',
                        marker_color=np.where(macd_hist>=0,'green','red'), opacity=0.5,
                        hovertext=[f"<b>Date:</b> {d.strftime('%Y-%m-%d')}<br><b>MACD:</b> {m:.4f}<br><b>Signal:</b> {s:.4f}<br><b>Hist:</b> {h:.4f}" for d,m,s,h in zip(data.index,data['MACD'],data['Signal'],macd_hist)],
                        hoverinfo="text"
                    ))
                    macd_fig.add_trace(go.Scatter(
                        x=data.index, y=data['MACD'], mode='lines',
                        name='MACD', line=dict(color='orange', width=2),
                        hovertemplate='MACD: %{y:.4f}<br>Date: %{x|%Y-%m-%d}'
                    ))
                    macd_fig.add_trace(go.Scatter(
                        x=data.index, y=data['Signal'], mode='lines',
                        name='Signal', line=dict(color='purple', width=1.5, dash='dot'),
                        hovertemplate='Signal: %{y:.4f}<br>Date: %{x|%Y-%m-%d}'
                    ))
                    macd_fig.update_layout(template="plotly_dark", height=300, title="MACD (12,26) & Signal (9)",
                                           margin=dict(l=10,r=10,t=40,b=20), autosize=True)
                    st.plotly_chart(macd_fig, use_container_width=True, config={"responsive": True})

                    # ===== RSI CHART RESPONSIVE =====
                    st.subheader("📈 RSI Indicator (14)")
                    rsi_fig = go.Figure()
                    rsi_fig.add_trace(go.Scatter(
                        x=data.index, y=data['RSI'], mode='lines+markers',
                        name='RSI', line=dict(color='cyan', width=2), marker=dict(size=3),
                        hovertemplate='RSI: %{y:.2f}<br>Date: %{x|%Y-%m-%d}'
                    ))
                    rsi_fig.add_trace(go.Scatter(
                        x=data.index, y=[70]*len(data), mode='lines', name='Overbought (70)',
                        line=dict(color='red', width=1, dash='dash'), hoverinfo='skip'
                    ))
                    rsi_fig.add_trace(go.Scatter(
                        x=data.index, y=[30]*len(data), mode='lines', name='Oversold (30)',
                        line=dict(color='green', width=1, dash='dash'), hoverinfo='skip'
                    ))
                    rsi_fig.update_layout(template="plotly_dark", height=300, title="RSI (14) with Overbought/Oversold Levels",
                                          yaxis=dict(range=[0,100], automargin=True), margin=dict(l=10,r=10,t=40,b=20), autosize=True)
                    st.plotly_chart(rsi_fig, use_container_width=True, config={"responsive": True})

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
else:
    st.info("👈 Enter a stock symbol and click **Predict Trend**.")























