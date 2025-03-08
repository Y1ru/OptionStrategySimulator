import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import interp1d
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime

# Black-Scholes function
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

# Intrinsic value function
def intrinsic_value(S, K, option_type):
    if option_type.lower() == 'call':
        return np.maximum(S - K, 0)
    elif option_type.lower() == 'put':
        return np.maximum(K - S, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

# Main Streamlit app
def main():
    st.title("Options Strategy Analyzer")

    # User input for stock symbol
    ticker = st.text_input("Enter Stock Symbol", value="TSLA").upper()
    stock = yf.Ticker(ticker)

    # Display stock logo
    logo_url = stock.info.get('logo_url', '')
    if logo_url:
        st.image(logo_url, width=100)

    # Fetch data
    try:
        spot_price = stock.history(period='1d')['Close'].iloc[0]
    except IndexError:
        st.error(f"No data available for {ticker}")
        return

    r = 0.05

    if len(stock.options) == 0:
        st.error(f"No options data available for {ticker}")
        return

    # Fetch available expiration dates
    expiration_dates = stock.options
    st.sidebar.subheader("Select Expiration Dates for Each Leg")

    # Strategy definitions with expiration date selection
    strategies = {
        # Basic Directional Strategies
        'Long Call': {
            'legs': [{'option_type': 'call', 'strike_offset': 0, 'amount': 1000}]
        },
        'Long Put': {
            'legs': [{'option_type': 'put', 'strike_offset': 0, 'amount': 1000}]
        },
        'Long Put (OTM)': {
            'legs': [{'option_type': 'put', 'strike_offset': -5, 'amount': 1000}]
        },
        'Short Put': {
            'legs': [{'option_type': 'put', 'strike_offset': 0, 'amount': -1000}]
        },
        
        # Vertical Spreads
        'Long Call Spread': {
            'legs': [
                {'option_type': 'call', 'strike_offset': 0, 'amount': 1000},
                {'option_type': 'call', 'strike_offset': 5, 'amount': -1000}
            ]
        },
        'Long Put Spread': {
            'legs': [
                {'option_type': 'put', 'strike_offset': 0, 'amount': 1000},
                {'option_type': 'put', 'strike_offset': -5, 'amount': -1000}
            ]
        },
        
        # Straddles and Strangles
        'Long Straddle': {
            'legs': [
                {'option_type': 'call', 'strike_offset': 0, 'amount': 1000},
                {'option_type': 'put', 'strike_offset': 0, 'amount': 1000}
            ]
        },
        'Long Strangle': {
            'legs': [
                {'option_type': 'call', 'strike_offset': 5, 'amount': 1000},
                {'option_type': 'put', 'strike_offset': -5, 'amount': 1000}
            ]
        },
        
        # Ratio Spreads
        'Back Spread with Calls': {
            'legs': [
                {'option_type': 'call', 'strike_offset': 0, 'amount': -1000},
                {'option_type': 'call', 'strike_offset': 10, 'amount': 2000}
            ]
        },
        'Back Spread with Puts': {
            'legs': [
                {'option_type': 'put', 'strike_offset': 0, 'amount': -1000},
                {'option_type': 'put', 'strike_offset': -10, 'amount': 2000}
            ]
        },
        
        # Butterflies
        'Skip Strike Butterfly with Calls': {
            'legs': [
                {'option_type': 'call', 'strike_offset': 0, 'amount': 1000},
                {'option_type': 'call', 'strike_offset': 5, 'amount': -2000},
                {'option_type': 'call', 'strike_offset': 10, 'amount': 1000}
            ]
        },
        'Skip Strike Butterfly with Puts': {
            'legs': [
                {'option_type': 'put', 'strike_offset': -10, 'amount': 1000},
                {'option_type': 'put', 'strike_offset': -5, 'amount': -2000},
                {'option_type': 'put', 'strike_offset': 0, 'amount': 1000}
            ]
        },
        
        # Advanced Strategies
        'Iron Condor': {
            'legs': [
                {'option_type': 'put', 'strike_offset': -10, 'amount': 1000},
                {'option_type': 'put', 'strike_offset': -5, 'amount': -1000},
                {'option_type': 'call', 'strike_offset': 5, 'amount': -1000},
                {'option_type': 'call', 'strike_offset': 10, 'amount': 1000}
            ]
        },
        'Christmas Tree Butterfly': {
            'legs': [
                {'option_type': 'call', 'strike_offset': 0, 'amount': 1000},
                {'option_type': 'call', 'strike_offset': 5, 'amount': -3000},
                {'option_type': 'call', 'strike_offset': 10, 'amount': 2000}
            ]
        },
        'Double Calendar': {
            'legs': [
                {'option_type': 'put', 'strike_offset': 0, 'amount': -1000},
                {'option_type': 'put', 'strike_offset': 0, 'amount': 1000},
                {'option_type': 'call', 'strike_offset': 0, 'amount': -1000},
                {'option_type': 'call', 'strike_offset': 0, 'amount': 1000}
            ]
        }
    }

    # Dropdown menu for strategy selection
    selected_strategy = st.selectbox("Select Strategy", list(strategies.keys()))

    # Get the legs for the selected strategy
    legs = strategies[selected_strategy]['legs']

    # Create a dictionary to store selected expiration dates for each leg
    expiration_selections = {}
    for i, leg in enumerate(legs):
        expiration_selections[f'leg_{i}'] = st.sidebar.selectbox(
            f"Expiration for Leg {i + 1} ({leg['option_type'].capitalize()})",
            expiration_dates,
            index=min(i, len(expiration_dates) - 1)  # Default to the first few expirations
        )

    # Fetch options chains for selected expirations
    options_chains = {}
    for i, leg in enumerate(legs):
        expiration = expiration_selections[f'leg_{i}']
        if expiration not in options_chains:
            options_chains[expiration] = stock.option_chain(expiration)

    # Build strategy options DataFrame
    strategy_options = []
    for i, leg in enumerate(legs):
        expiration = expiration_selections[f'leg_{i}']
        options_chain = options_chains[expiration]
        if leg['option_type'] == 'call':
            options = options_chain.calls.sort_values('strike')
        else:
            options = options_chain.puts.sort_values('strike')

        atm_index = (options['strike'] - spot_price).abs().argsort()[0]
        selected_option = options.iloc[atm_index + leg['strike_offset']]

        strategy_options.append({
            'contractSymbol': selected_option['contractSymbol'],
            'option_type': leg['option_type'],
            'strike': selected_option['strike'],
            'lastPrice': selected_option['lastPrice'],
            'impliedVolatility': selected_option['impliedVolatility'],
            'expiration': expiration,
            'amount': leg['amount']
        })

    strategy_options = pd.DataFrame(strategy_options)
    
    # Add strike price adjustment sliders
    st.sidebar.subheader("Adjust Strike Prices")
    adjusted_strikes = {}
    for i, option in strategy_options.iterrows():
        default_strike = option['strike']
        adjusted_strikes[i] = st.sidebar.slider(
            f"Leg {i+1} ({option['option_type'].capitalize()}) Strike",
            min_value=default_strike * 0.7,
            max_value=default_strike * 1.3,
            value=default_strike,
            step=0.5,
            format="$%.2f"
        )
    
    # Update strike prices in strategy_options with adjusted values
    strategy_options['strike'] = strategy_options.index.map(adjusted_strikes)
    
    # Continue with Monte Carlo simulation...
    earliest_expiration = min(expiration_selections.values())
    expiration_date = datetime.strptime(earliest_expiration, '%Y-%m-%d')
    trade_date = datetime.now()
    
    # Calculate historical volatility
    hist = stock.history(period='1y')
    log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    vol_periods = [5, 10, 20, 30, 60, 90]
    vols = [log_returns[-n:].std() * np.sqrt(252) for n in vol_periods]
    d = (expiration_date - trade_date).days
    y_interp = interp1d(vol_periods, vols, fill_value='extrapolate')
    volaV = y_interp(d)
    
    # Generate price distribution
    num_simulations = 5000
    sigma_T = (volaV / np.sqrt(252)) * np.sqrt(d)
    Distrib = np.random.lognormal(0, sigma_T, size=num_simulations) * spot_price
    Distrib = Distrib[(Distrib > spot_price * 0.8) & (Distrib < spot_price * 1.2)]
    Price = np.sort(Distrib)

    # Calculate P&L for selected strategy
    payoff = np.zeros_like(Price)
    current_pnl = np.zeros_like(Price)
    for _, option in strategy_options.iterrows():
        strike = option['strike']
        option_type = option['option_type']
        last_price = option['lastPrice']
        iv = option['impliedVolatility']
        amount = option['amount']
        expiration_date = datetime.strptime(option['expiration'], '%Y-%m-%d')
        T = (expiration_date - datetime.now()).days / 365.0
        intrinsic = intrinsic_value(Price, strike, option_type)
        payoff += (intrinsic - last_price) * amount
        current_price = black_scholes(Price, strike, T, r, iv, option_type)
        current_pnl += (current_price - last_price) * amount

    # Create plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=Price, y=payoff, name='Payoff at Expiration', line=dict(color='blue')), secondary_y=False)
    fig.add_trace(go.Scatter(x=Price, y=current_pnl, name='Current P&L', line=dict(color='green')), secondary_y=False)
    fig.add_trace(go.Histogram(x=Distrib, nbinsx=50, histnorm='probability density', name='Probability Distribution', opacity=0.5), secondary_y=True)
    fig.update_layout(
        title=f'{selected_strategy} P&L and Probability Distribution',
        xaxis_title='Stock Price',
        yaxis_title='P&L',
        yaxis2_title='Probability Density'
    )

    # Calculate probability of winning
    win_probability = (payoff > 0).mean() * 100
    max_loss = payoff.min()
    max_profit = payoff.max()
    
    # Display probability metrics and strike prices
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Probability of Profit", f"{win_probability:.1f}%")
        
        # Display strike prices for each leg
        st.write("Strike Prices:")
        for i, option in strategy_options.iterrows():
            st.write(f"Leg {i+1} ({option['option_type'].capitalize()}): ${option['strike']:.2f}")
    with col2:
        st.metric("Maximum Loss", f"${max_loss:.2f}")
    with col3:
        st.metric("Maximum Profit", f"${max_profit:.2f}")

    # Display plot
    st.plotly_chart(fig)
    
    # Display strategy details
    st.subheader("Strategy Details")
    st.dataframe(strategy_options)

if __name__ == "__main__":
    main()