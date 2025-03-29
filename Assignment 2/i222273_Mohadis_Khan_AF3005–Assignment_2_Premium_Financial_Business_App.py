import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import altair as alt


st.set_page_config(page_title="Premium Financial Business App", layout="wide")
st.markdown(
    """
    <style>
        body {background-color: #eef3f7; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
        .main-title {text-align: center; font-size: 48px; font-weight: bold; color: #0D47A1; margin-bottom: 0px; margin-top: 20px;}
        .main-subtitle {text-align: center; font-size: 28px; font-weight: 600; color: #1565C0; margin-bottom: 30px;}
        .section-header {font-size: 26px; color: #1E88E5; margin: 20px 0px; border-bottom: 3px solid #1E88E5; padding-bottom: 8px; }
        .module-title {font-size: 32px; color: #0D47A1; margin-bottom: 10px; font-weight: bold; }
        .stButton>button {background-color: #1E88E5; color: white; border-radius: 8px; font-size: 18px; padding: 10px 20px; margin-top: 10px;}
        .footer {text-align: center; font-size: 14px; color: #777; margin-top: 30px; padding: 10px 0px; border-top: 1px solid #ccc;}
    </style>
    """, unsafe_allow_html=True
)


def check_loan_eligibility(employment_status, income, credit_score):
    """Determine loan eligibility based on employment, income, and credit score."""
    if employment_status == 'Unemployed':
        return False, "❌ Loan Rejected: Applicant must be employed."
    elif income < 50000:
        return False, "❌ Loan Rejected: Income must be at least PKR 50,000."
    elif credit_score >= 750:
        return True, "✅ Eligible: Loan approved at a competitive interest rate of 5%."
    elif 650 <= credit_score < 750:
        return True, "⚠️ Eligible: Loan approved at an interest rate of 8%."
    else:
        return False, "❌ Loan Rejected: Credit Score is below 650."

def calculate_emi(loan_amount, annual_interest_rate, tenure_years):
  
    monthly_interest_rate = annual_interest_rate / (12 * 100)
    months = tenure_years * 12
    emi = loan_amount * monthly_interest_rate * (1 + monthly_interest_rate) ** months / ((1 + monthly_interest_rate) ** months - 1)
    total_payment = emi * months
    total_interest = total_payment - loan_amount

    schedule = []
    remaining_balance = loan_amount
    for month in range(1, months + 1):
        interest_component = remaining_balance * monthly_interest_rate
        principal_component = emi - interest_component
        remaining_balance -= principal_component
        if remaining_balance < 0:
            principal_component += remaining_balance
            remaining_balance = 0
        schedule.append({
            "Month": month,
            "EMI": round(emi, 2),
            "Principal": round(principal_component, 2),
            "Interest": round(interest_component, 2),
            "Remaining Balance": round(remaining_balance, 2)
        })
    schedule_df = pd.DataFrame(schedule)
    return round(emi, 2), round(total_payment, 2), round(total_interest, 2), schedule_df

def investment_risk_analysis(stock_returns):
    stock_returns = np.array(stock_returns)
    avg_return = np.mean(stock_returns)
    std_return = np.std(stock_returns)
    min_return = np.min(stock_returns)
    max_return = np.max(stock_returns)
    if any(r < 0 for r in stock_returns):
        risk_level = "⚠️ High Risk Portfolio"
    elif all(r >= 5 for r in stock_returns):
        risk_level = "✅ Low Risk Portfolio"
    else:
        risk_level = "🟡 Medium Risk Portfolio"
    stats = {
        "Average Return (%)": round(avg_return, 2),
        "Standard Deviation (%)": round(std_return, 2),
        "Min Return (%)": round(min_return, 2),
        "Max Return (%)": round(max_return, 2)
    }
    return risk_level, stats

def track_currency_exchange(start_rate, end_rate):
    days, rates = [], []
    day = 1
    current_rate = start_rate
    while current_rate <= end_rate:
        days.append(f"Day {day}")
        rates.append(current_rate)
        current_rate += np.random.uniform(0.5, 1.5)
        current_rate = round(current_rate, 2)
        day += 1
        if current_rate > end_rate:
            break
    df = pd.DataFrame({"Day": days, "Exchange Rate (PKR/USD)": rates})
    return df

def budget_tracker(income, expenses):
    total_expenses = sum(expenses.values())
    net_savings = income - total_expenses
    return net_savings, total_expenses

def savings_goal_planner(current_savings, monthly_saving, goal):
    if monthly_saving <= 0:
        return None, None
    months_needed = math.ceil((goal - current_savings) / monthly_saving)
    timeline = pd.DataFrame({
        "Month": list(range(1, months_needed + 1)),
        "Projected Savings": [round(current_savings + monthly_saving * i, 2) for i in range(1, months_needed + 1)]
    })
    return months_needed, timeline

def simulate_stock_market(num_stocks=5, days=30):
    dates = pd.date_range(end=datetime.today(), periods=days).to_pydatetime().tolist()
    data = {}
    for stock in [f"Stock {i}" for i in range(1, num_stocks + 1)]:
        price = 100 + np.random.uniform(-5, 5)
        prices = [round(price + np.random.normal(0, 2), 2) for _ in range(days)]
        data[stock] = prices
    df = pd.DataFrame(data, index=[d.strftime("%Y-%m-%d") for d in dates])
    return df

st.sidebar.markdown("<p class='main-subtitle'>Navigation</p>", unsafe_allow_html=True)
module = st.sidebar.radio("Select a Module", 
                          ["Loan Processing", "Investment Risk Analysis", 
                           "Currency Exchange Tracker", "Budget Tracker", 
                           "Savings Goal Planner", "Stock Market Dashboard"])

if module != "Loan Processing":
    for key in ["loan_application_submitted", "loan_eligible", "credit_score"]:
        if key in st.session_state:
            del st.session_state[key]


st.markdown("<p class='main-title'>Premium Financial Business App</p>", unsafe_allow_html=True)
st.markdown("<p class='main-subtitle'>Empowering Your Financial Decisions with Precision & Insight</p>", unsafe_allow_html=True)
st.markdown("---")

if module == "Loan Processing":
    st.markdown("<p class='section-header'>Loan Processing Portal</p>", unsafe_allow_html=True)
    
    if "loan_application_submitted" not in st.session_state:
        st.session_state["loan_application_submitted"] = False
    if "loan_eligible" not in st.session_state:
        st.session_state["loan_eligible"] = False
    if "credit_score" not in st.session_state:
        st.session_state["credit_score"] = 700
    
    st.markdown("### Step 1: Loan Application")
    with st.form("loan_application_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed"],
                                               help="Select your current employment status.")
        with col2:
            income = st.number_input("Monthly Income (PKR)", min_value=0, step=1000, value=60000,
                                     help="Enter your monthly income in PKR.")
        with col3:
            credit_score = st.slider("Credit Score", 300, 1000, 700, step=10,
                                     help="Adjust your credit score.")
        submitted_app = st.form_submit_button("Submit Application")
    
    if submitted_app:
        eligible, message = check_loan_eligibility(employment_status, income, credit_score)
        st.session_state["loan_application_submitted"] = True
        st.session_state["loan_eligible"] = eligible
        st.session_state["credit_score"] = credit_score
        if eligible:
            st.success(message)
        else:
            st.error(message)
    
    if st.session_state.get("loan_application_submitted") and st.session_state.get("loan_eligible"):
        st.markdown("---")
        st.markdown("### Step 2: Loan Calculator & EMI Details")
        with st.form("loan_calculator_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                loan_amount = st.number_input("Loan Amount (PKR)", min_value=10000, step=1000, value=500000,
                                              help="Enter the desired loan amount.")
            with col2:
                default_rate = 5.0 if st.session_state["credit_score"] >= 750 else 8.0 if st.session_state["credit_score"] >= 650 else 10.0
                interest_rate = st.number_input("Annual Interest Rate (%)", min_value=1.0, value=default_rate, step=0.1, format="%.2f",
                                                help="Enter the annual interest rate.")
            with col3:
                tenure_years = st.number_input("Loan Tenure (Years)", min_value=1, max_value=30, value=5, step=1,
                                               help="Select the loan tenure in years.")
            submitted_calc = st.form_submit_button("Calculate EMI")
        
        if submitted_calc:
            emi, total_payment, total_interest, schedule_df = calculate_emi(loan_amount, interest_rate, tenure_years)
            st.markdown(f"**Monthly EMI:** PKR {emi}")
            st.markdown(f"**Total Payment over {tenure_years} Years:** PKR {total_payment}")
            st.markdown(f"**Total Interest Payable:** PKR {total_interest}")
            st.markdown("#### Detailed Repayment Schedule")
            st.dataframe(schedule_df.style.format({
                'EMI': "{:.2f}", 
                'Principal': "{:.2f}", 
                'Interest': "{:.2f}", 
                'Remaining Balance': "{:.2f}"
            }))
            st.markdown("#### Remaining Loan Balance Over Time")
            chart = alt.Chart(schedule_df).mark_line(point=True).encode(
                x=alt.X('Month:Q', title='Month'),
                y=alt.Y('Remaining Balance:Q', title='Remaining Balance (PKR)'),
                tooltip=['Month', 'Remaining Balance']
            ).properties(width=700, height=400)
            st.altair_chart(chart, use_container_width=True)


elif module == "Investment Risk Analysis":
    st.markdown("<p class='section-header'>Investment Risk Analyzer</p>", unsafe_allow_html=True)
    st.markdown("Enter the stock returns (in %) for your portfolio to obtain a detailed risk analysis and statistical summary.")
    with st.form("risk_analysis_form"):
        returns_input = st.text_area("Enter Stock Returns (%) separated by commas", value="6, 7, 8, 5, 4",
                                     help="Example: 6, 7, 8, 5, 4")
        submitted_returns = st.form_submit_button("Analyze Portfolio")
    if submitted_returns:
        try:
            stock_returns = list(map(float, returns_input.split(',')))
            risk_level, stats = investment_risk_analysis(stock_returns)
            st.markdown(f"### Portfolio Risk Level: {risk_level}")
            st.markdown("#### Statistical Summary:")
            st.table(pd.DataFrame([stats]).T.rename(columns={0: "Value"}))
            df_returns = pd.DataFrame({"Stock": list(range(1, len(stock_returns) + 1)), "Return (%)": stock_returns})
            bar_chart = alt.Chart(df_returns).mark_bar(color="#1E88E5").encode(
                x=alt.X("Stock:O", title="Stock Index"),
                y=alt.Y("Return (%):Q", title="Return (%)"),
                tooltip=["Stock", "Return (%)"]
            ).properties(width=700, height=400)
            st.altair_chart(bar_chart, use_container_width=True)
        except Exception as e:
            st.error("❌ Invalid input! Please enter valid numbers separated by commas.")


elif module == "Currency Exchange Tracker":
    st.markdown("<p class='section-header'>Currency Exchange Rate Simulator</p>", unsafe_allow_html=True)
    st.markdown("Simulate the fluctuation of the PKR/USD exchange rate with realistic random variations and analyze the trend.")
    with st.form("currency_exchange_form"):
        col1, col2 = st.columns(2)
        with col1:
            start_rate = st.number_input("Starting Exchange Rate (PKR/USD)", min_value=200.0, value=290.0, step=0.1, format="%.2f",
                                         help="Enter the starting exchange rate.")
        with col2:
            end_rate = st.number_input("Target Exchange Rate (PKR/USD)", min_value=start_rate, value=300.0, step=0.1, format="%.2f",
                                       help="Enter the target exchange rate.")
        submitted_exchange = st.form_submit_button("Simulate Exchange Rates")
    if submitted_exchange:
        df_exchange = track_currency_exchange(start_rate, end_rate)
        st.markdown("#### Simulated Exchange Rate Trend")
        line_chart = alt.Chart(df_exchange).mark_line(point=True).encode(
            x=alt.X('Day', title='Day'),
            y=alt.Y('Exchange Rate (PKR/USD):Q', title='Exchange Rate (PKR/USD)'),
            tooltip=['Day', 'Exchange Rate (PKR/USD)']
        ).properties(width=700, height=400)
        st.altair_chart(line_chart, use_container_width=True)
        st.markdown("#### Detailed Exchange Rate Data")
        st.dataframe(df_exchange)


elif module == "Budget Tracker":
    st.markdown("<p class='section-header'>Budget Tracker</p>", unsafe_allow_html=True)
    st.markdown("Manage your monthly finances by tracking your income and expenses to determine your net savings.")
    
    with st.form("budget_tracker_form"):
        monthly_income = st.number_input("Monthly Income (PKR)", min_value=0, step=1000, value=150000,
                                         help="Enter your total monthly income.")
        st.markdown("#### Enter your monthly expenses for each category:")
        expense_rent = st.number_input("Rent (PKR)", min_value=0, step=500, value=40000)
        expense_food = st.number_input("Food (PKR)", min_value=0, step=500, value=20000)
        expense_transport = st.number_input("Transportation (PKR)", min_value=0, step=500, value=10000)
        expense_utilities = st.number_input("Utilities (PKR)", min_value=0, step=500, value=5000)
        expense_misc = st.number_input("Miscellaneous (PKR)", min_value=0, step=500, value=5000)
        submitted_budget = st.form_submit_button("Calculate Savings")
    
    if submitted_budget:
        expenses = {
            "Rent": expense_rent,
            "Food": expense_food,
            "Transportation": expense_transport,
            "Utilities": expense_utilities,
            "Miscellaneous": expense_misc
        }
        net_savings, total_expenses = budget_tracker(monthly_income, expenses)
        st.markdown(f"**Total Expenses:** PKR {total_expenses}")
        st.markdown(f"**Net Savings:** PKR {net_savings}")
        expense_df = pd.DataFrame(list(expenses.items()), columns=["Category", "Amount (PKR)"])
        pie_chart = alt.Chart(expense_df).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Amount (PKR)", type="quantitative"),
            color=alt.Color(field="Category", type="nominal"),
            tooltip=["Category", "Amount (PKR)"]
        ).properties(width=500, height=400)
        st.altair_chart(pie_chart, use_container_width=True)


elif module == "Savings Goal Planner":
    st.markdown("<p class='section-header'>Savings Goal Planner</p>", unsafe_allow_html=True)
    st.markdown("Plan your savings by setting a goal and tracking your progress over time.")
    
    with st.form("savings_goal_form"):
        current_savings = st.number_input("Current Savings (PKR)", min_value=0, step=1000, value=50000)
        monthly_saving = st.number_input("Monthly Savings (PKR)", min_value=0, step=500, value=15000)
        goal = st.number_input("Savings Goal (PKR)", min_value=0, step=1000, value=500000)
        submitted_goal = st.form_submit_button("Plan Goal")
    
    if submitted_goal:
        months_needed, timeline = savings_goal_planner(current_savings, monthly_saving, goal)
        if months_needed is None:
            st.error("Monthly savings must be greater than zero!")
        else:
            st.markdown(f"**Estimated Time to Reach Goal:** {months_needed} months")
            st.markdown("#### Savings Timeline")
            st.dataframe(timeline)
            line_chart = alt.Chart(timeline).mark_line(point=True).encode(
                x=alt.X('Month', title='Month'),
                y=alt.Y('Projected Savings', title='Projected Savings (PKR)'),
                tooltip=['Month', 'Projected Savings']
            ).properties(width=700, height=400)
            st.altair_chart(line_chart, use_container_width=True)


elif module == "Stock Market Dashboard":
    st.markdown("<p class='section-header'>Stock Market Dashboard</p>", unsafe_allow_html=True)
    st.markdown("Explore simulated stock price data for a range of stocks over a selected period.")
    
    with st.form("stock_market_form"):
        num_stocks = st.number_input("Number of Stocks", min_value=1, max_value=10, value=5, step=1)
        days = st.number_input("Number of Days", min_value=10, max_value=60, value=30, step=1)
        submitted_stock = st.form_submit_button("Simulate Stock Data")
    
    if submitted_stock:
        df_stocks = simulate_stock_market(num_stocks=num_stocks, days=days)
        st.markdown("#### Stock Price Simulation")
        st.dataframe(df_stocks)
        stock_chart = alt.Chart(df_stocks.reset_index().melt(id_vars="index")).mark_line(point=True).encode(
            x=alt.X('index:T', title='Date'),
            y=alt.Y('value:Q', title='Stock Price (PKR)'),
            color=alt.Color('variable:N', title="Stock"),
            tooltip=['index', 'variable', 'value']
        ).properties(width=700, height=400)
        st.altair_chart(stock_chart, use_container_width=True)

st.markdown("<div class='footer'>© {} Premium Financial Business App | All Rights Reserved</div>".format(datetime.now().year), unsafe_allow_html=True)
