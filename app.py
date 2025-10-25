+ # YENİ FONKSİYONLAR
+ def get_1d_data(symbol, days=120):
+ def map_regime_to_4h(df_4h, df_1d):
+ def donchian(df, n=20):
+ def bollinger(df, n=20, k=2):
+ def chandelier_exit(df, period=22, mult=3):
+ def calculate_advanced_indicators(df):
+ def get_regime(symbol, df_4h):
+ def can_trade(last_signal_time, current_time, cooldown_bars=3):
+ def generate_signals_v2(df, regime_col='REGIME', min_rr_ratio=1.5, cooldown_bars=3, bb_width_pct=2.5, donchian_len=20):

# DEĞİŞTİRİLEN FONKSİYONLAR
- def calculate_indicators(data, ema_period=50, rsi_period=14):
+ def calculate_indicators(data, ema_period=50, rsi_period=14, donchian_len=20, bb_width_pct=2.5):

# YENİ SIDEBAR PARAMETRELERİ
+ cooldown_bars = st.slider("Cooldown Bars", 1, 10, 3)
+ bb_width_pct = st.number_input("BB Width (%)", 1.0, 5.0, 2.5, 0.1)
+ donchian_len = st.slider("Donchian Length", 10, 50, 20)

# BACKTEST GÜNCELLEMESİ
- sig = generate_signal_at_bar(...)
+ sig = generate_signals_v2(...)