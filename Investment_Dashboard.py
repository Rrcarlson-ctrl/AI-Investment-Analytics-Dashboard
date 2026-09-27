import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from openai import OpenAI

# Page setup
st.set_page_config(
    page_title="AI Investment Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("AI-Enhanced Investment Analytics Dashboard")
st.write(
    "Analyze a publicly traded company against the S&P 500 "
    "using five years of monthly market data."
)

# User inputs
ticker = st.text_input(
    "Stock Ticker",
    value="MSFT"
).strip().upper()

aum = st.number_input(
    "Fund AUM ($)",
    min_value=0.0,
    value=50000000.0,
    step=1000000.0
)

strategy = st.selectbox(
    "Investment Strategy",
    ["Growth", "Balanced", "Conservative", "Income"]
)

stance = st.selectbox(
    "Your Market View",
    ["Bullish", "Bearish", "Base case"]
)

thesis = st.text_area(
    "Investment Thesis",
    placeholder="Example: I believe the company will outperform because..."
)

if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False

if "local_briefing" not in st.session_state:
    st.session_state.local_briefing = None

if "frontier_briefing" not in st.session_state:
    st.session_state.frontier_briefing = None

if st.button("Run Investment Analysis"):
    st.session_state.analysis_run = True
    st.session_state.local_briefing = None
    st.session_state.frontier_briefing = None

analyze = st.session_state.analysis_run

if analyze:

    market = "^GSPC"

    with st.spinner("Downloading market data and running analysis..."):

        prices = yf.download(
            [ticker, market],
            period="61mo",
            interval="1mo",
            auto_adjust=True,
            progress=False
        )["Close"].dropna()

        returns = prices.pct_change().dropna().tail(60)

        X = returns[market]
        Y = returns[ticker]

        stock_mean = Y.mean()
        stock_std = Y.std()

        mkt_mean = X.mean()
        mkt_std = X.std()

        model = sm.OLS(Y, sm.add_constant(X)).fit()

        alpha = model.params.iloc[0]
        beta = model.params.iloc[1]

        r_squared = model.rsquared
        correlation = X.corr(Y)

    st.success("Analysis complete!")

    st.subheader(f"{ticker} Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        f"{ticker} Mean Monthly Return",
        f"{stock_mean:.2%}"
    )

    col2.metric(
        f"{ticker} Monthly Volatility",
        f"{stock_std:.2%}"
    )

    col3.metric(
        "Beta",
        f"{beta:.2f}"
    )

    col4.metric(
        "Correlation with S&P 500",
        f"{correlation:.2f}"
    )
    st.subheader("Stock vs. S&P 500")

    comparison = pd.DataFrame({
        ticker: [stock_mean, stock_std],
        "S&P 500": [mkt_mean, mkt_std]
    }, index=[
        "Mean Monthly Return",
        "Standard Deviation"
    ])

    st.dataframe(
        comparison.style.format("{:.2%}"),
        use_container_width=True
    )

    st.write(f"**R-Squared:** {r_squared:.2%}")
    st.write(f"**Monthly Observations:** {len(returns)}")
    st.subheader(f"{ticker} Monthly Return Distribution")

    fig1, ax1 = plt.subplots(figsize=(8, 5))

    ax1.hist(Y * 100, bins=15, edgecolor="black")

    ax1.set_title(f"{ticker} Monthly Returns - Distribution")
    ax1.set_xlabel("Monthly Return (%)")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)

    st.pyplot(fig1)
    st.subheader(f"{ticker} vs. S&P 500 Monthly Returns")

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    ax2.scatter(X * 100, Y * 100, alpha=0.7)

    xl = np.linspace(X.min(), X.max(), 100)

    ax2.plot(
        xl * 100,
        (alpha + beta * xl) * 100,
        label=f"Beta = {beta:.2f}, r = {correlation:.2f}"
    )

    ax2.set_xlabel("S&P 500 Monthly Return (%)")
    ax2.set_ylabel(f"{ticker} Monthly Return (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig2)
    st.subheader("Monthly Returns Over Time")

    fig3, ax3 = plt.subplots(figsize=(10, 5))

    ax3.plot(
        returns.index,
        Y * 100,
        label=ticker
    )

    ax3.plot(
        returns.index,
        X * 100,
        label="S&P 500"
    )

    ax3.set_title(f"{ticker} vs. S&P 500 Monthly Returns Over Time")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Monthly Return (%)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    st.pyplot(fig3)
    st.subheader("Liquidity Analysis")

    company = yf.Ticker(ticker)
    bs = company.balance_sheet

    def get_balance_sheet_value(field):
        try:
            return bs.loc[field].iloc[0]
        except Exception:
            return None

    ca = get_balance_sheet_value("Current Assets")
    cl = get_balance_sheet_value("Current Liabilities")
    inv = get_balance_sheet_value("Inventory")
    cash = get_balance_sheet_value("Cash And Cash Equivalents")

    if inv is None:
        inv = 0

    if ca is not None and cl is not None and cash is not None:

        current_ratio = ca / cl
        quick_ratio = (ca - inv) / cl
        cash_ratio = cash / cl
        working_capital = ca - cl

        liq1, liq2, liq3, liq4 = st.columns(4)

        liq1.metric("Current Ratio", f"{current_ratio:.2f}")
        liq2.metric("Quick Ratio", f"{quick_ratio:.2f}")
        liq3.metric("Cash Ratio", f"{cash_ratio:.2f}")
        liq4.metric("Working Capital", f"${working_capital:,.0f}")

    else:
        st.warning("Liquidity data was not available for this company.")
    st.subheader("Interactive Portfolio Position")

    allocation = st.slider(
        "Portfolio Allocation to Stock (%)",
        min_value=0.0,
        max_value=20.0,
        value=3.0,
        step=0.5
    )

    position_dollars = aum * (allocation / 100)

    pos1, pos2 = st.columns(2)

    pos1.metric(
        "Portfolio Allocation",
        f"{allocation:.1f}%"
    )

    pos2.metric(
        f"Position Size in {ticker}",
        f"${position_dollars:,.0f}"
    )
    st.subheader("Position Risk Impact")

    beta_contribution = (allocation / 100) * beta

    risk1, risk2, risk3 = st.columns(3)

    risk1.metric(
        "Stock Beta",
        f"{beta:.2f}"
    )

    risk2.metric(
        "Position Weight",
        f"{allocation:.1f}%"
    )

    risk3.metric(
        "Beta Contribution",
        f"{beta_contribution:.3f}"
    )

    st.write(
        f"At a {allocation:.1f}% portfolio allocation, "
        f"{ticker} contributes approximately {beta_contribution:.3f} "
        f"to the portfolio's overall beta."
    )

    st.subheader("AI Analyst Briefing")

    if st.button("Generate Local AI Briefing"):

        with st.spinner("Phi is analyzing the investment..."):

            try:
                client = OpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama"
                )

                prompt = f"""
You are a senior investment analyst advising a portfolio manager.

Write a professional executive-level investment briefing for {ticker}.
Use plain business language rather than technical statistical language.

PORTFOLIO MANAGER PROFILE:
Fund AUM: ${aum:,.0f}
Strategy: {strategy}
Market view: {stance}
Investment thesis: "{thesis}"

PROPOSED POSITION:
Portfolio allocation: {allocation:.1f}%
Position size: ${position_dollars:,.0f}

INVESTMENT DATA:
{ticker} mean monthly return: {stock_mean:.2%}
S&P 500 mean monthly return: {mkt_mean:.2%}

{ticker} monthly standard deviation: {stock_std:.2%}
S&P 500 monthly standard deviation: {mkt_std:.2%}

Beta: {beta:.3f}
Correlation with S&P 500: {correlation:.3f}
R-squared: {r_squared:.2%}

Current Ratio: {current_ratio:.2f}
Quick Ratio: {quick_ratio:.2f}
Cash Ratio: {cash_ratio:.2f}
Working Capital: ${working_capital:,.0f}

Beta contribution from proposed position: {beta_contribution:.3f}

Your briefing should:
1. Compare the stock's mean monthly return with the S&P 500.
2. Compare its volatility with the S&P 500.
3. Explain beta in plain English.
4. Explain how beta differs from standard deviation.
5. Explain the correlation coefficient.
6. Discuss whether the historical data supports or challenges the manager's thesis.
7. Discuss whether the stock appears consistent with the selected {strategy} strategy.
8. Discuss the risk implications of the proposed {allocation:.1f}% position.

IMPORTANT RULES:
- Use only the quantitative information provided above.
- Do not invent company facts, industry conditions, management changes, or financial information.
- Check all numerical comparisons carefully. A larger percentage must not be described as smaller.
- Do not interpret standard deviation as evidence of operational or business instability.
- Do not generate additional prompts, instructions, constraints, contact information, or placeholder text.
- Do not make up information that is not contained in the provided data.
- End immediately after the investment briefing.
- Keep the entire response under 300 words.
"""
                response = client.chat.completions.create(
                    model="phi3",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                st.session_state.local_briefing = response.choices[0].message.content

                st.success("Local AI analysis complete!")

            except Exception as e:
                st.error(f"AI briefing could not be generated: {e}")

    st.subheader("Frontier AI Analyst Briefing")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Paste your API key here"
    )

    if st.button("Generate Frontier AI Briefing"):

        if not api_key:
            st.warning("Please enter your OpenAI API key.")

        else:

            with st.spinner("Frontier AI is analyzing the investment..."):

                try:
                    frontier_client = OpenAI(api_key=api_key)

                    frontier_prompt = f"""
You are a senior investment analyst advising a portfolio manager.

Write a professional executive-level investment briefing for {ticker}.
Use plain business language rather than technical statistical language.

PORTFOLIO MANAGER PROFILE:
Fund AUM: ${aum:,.0f}
Strategy: {strategy}
Market view: {stance}
Investment thesis: "{thesis}"

PROPOSED POSITION:
Portfolio allocation: {allocation:.1f}%
Position size: ${position_dollars:,.0f}

INVESTMENT DATA:
{ticker} mean monthly return: {stock_mean:.2%}
S&P 500 mean monthly return: {mkt_mean:.2%}

{ticker} monthly standard deviation: {stock_std:.2%}
S&P 500 monthly standard deviation: {mkt_std:.2%}

Beta: {beta:.3f}
Correlation with S&P 500: {correlation:.3f}
R-squared: {r_squared:.2%}

Current Ratio: {current_ratio:.2f}
Quick Ratio: {quick_ratio:.2f}
Cash Ratio: {cash_ratio:.2f}
Working Capital: ${working_capital:,.0f}

Beta contribution from proposed position: {beta_contribution:.3f}

Your briefing should:
1. Compare the stock's mean monthly return with the S&P 500.
2. Compare its volatility with the S&P 500.
3. Explain beta in plain English.
4. Explain how beta differs from standard deviation.
5. Explain the correlation coefficient.
6. Discuss whether the historical data supports or challenges the manager's thesis.
7. Discuss whether the stock appears consistent with the selected strategy.
8. Discuss the risk implications of the proposed portfolio position.

Write approximately 200 words.
"""
                    frontier_response = frontier_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "user", "content": frontier_prompt}
                        ]
                    )

                    st.session_state.frontier_briefing = frontier_response.choices[0].message.content

                    st.success("Frontier AI analysis complete!")

                except Exception as e:
                    st.error(
                        f"Frontier AI briefing could not be generated: {e}"
                    )

    if st.session_state.local_briefing and st.session_state.frontier_briefing:

        st.divider()
        st.header("AI Model Comparison")

        local_col, frontier_col = st.columns(2)

        with local_col:
            st.subheader("Local AI - Phi")
            st.markdown(st.session_state.local_briefing)

        with frontier_col:
            st.subheader("Frontier AI - OpenAI")
            st.markdown(st.session_state.frontier_briefing)
        st.subheader("Model Comparison Takeaways")

        st.write(
            """
            **Local AI (Phi):** Runs locally through Ollama, allowing the
            investment analysis to remain on the user's computer rather than
            being sent to an external AI model.

            **Frontier AI (OpenAI):** Uses a cloud-based frontier model to
            interpret the same quantitative investment data supplied to the
            local model.

            **Comparison:** Review both responses for numerical accuracy,
            interpretation of financial metrics, conciseness, unsupported
            assumptions, and consistency with the underlying quantitative
            results.

            **Analyst Role:** Neither AI output should be accepted without
            review. The analyst remains responsible for validating the
            calculations, identifying unsupported claims, interpreting risk,
            and making the final investment judgment.
            """
        )