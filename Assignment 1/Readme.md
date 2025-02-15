# Smart Financial Management System

## Overview

SecureBank's **Smart Financial Management System** is an interactive Python-based tool that helps customers manage their financial activities. The system is designed to:

✅ Assess customer eligibility for loans. ✅ Classify investment portfolios based on risk. ✅ Automate loan repayment tracking. ✅ Monitor stock market trends and trigger alerts. ✅ Track currency exchange rates and suggest conversions.

The project utilizes **Python** and **ipywidgets** to provide an interactive user experience.

---

## Features & Functionality

### 1. Loan Eligibility & Interest Rate Calculation

**Objective:**

- Determine whether a customer qualifies for a loan.
- Calculate the applicable interest rate based on credit score.

**Logic:**

- The applicant must be **employed**.
- Minimum required income is **PKR 50,000**.
- Credit score evaluation:
  - **750+** → **5% Interest Rate** (Approved)
  - **650 - 749** → **8% Interest Rate** (Approved)
  - **Below 650** → **Loan Rejected**

**Implementation:**

- Uses `ipywidgets` for employment status, income, and credit score input.
- Displays approval/rejection with interest rate if eligible.

---

### 2. Investment Risk Assessment

**Objective:**

- Classify stock portfolios into **High, Medium, or Low Risk**.

**Logic:**

- **High Risk:** If any stock return is negative.
- **Medium Risk:** If all returns are positive, but at least one is below 5%.
- **Low Risk:** If all returns are 5% or above.

**Implementation:**

- Takes user input for stock returns via `ipywidgets`.
- Processes input and categorizes the risk level.

---

### 3. Loan Repayment Tracker

**Objective:**

- Simulate loan repayment and track the outstanding balance after each monthly payment.

**Logic:**

- The user enters an **initial loan balance** and a **fixed monthly payment**.
- The system deducts the payment monthly and displays the remaining balance.
- Stops once the loan is fully repaid.

**Implementation:**

- Uses a `while` loop to decrement the balance.
- Displays real-time tracking of loan payments.

---

### 4. Stock Price Monitoring & Trading Strategy

**Objective:**

- Monitor stock prices and trigger an alert when the price reaches **PKR 200**.

**Logic:**

- Iterates through a **list of stock prices**.
- Skips missing values (`None`) using `continue`.
- Stops tracking when a stock price reaches **PKR 200**.

**Implementation:**

- Uses a `for` loop and handles missing data.
- Displays stock price alerts dynamically.

---

### 5. Currency Exchange Rate Tracker

**Objective:**

- Track the **PKR/USD exchange rate** daily.

**Logic:**

- Starts at **PKR 290/USD**.
- Increases by **1 PKR per day**.
- Stops when it reaches **PKR 300/USD**.

**Implementation:**

- Uses a `while` loop for daily tracking.
- Displays real-time exchange rate updates.

---

## Technologies Used

- **Python 3**
- **ipywidgets** (for user interaction)
- **Jupyter Notebook / Google Colab** (for execution)

---

## How to Run the Project

1. Install dependencies:
   ```bash
   pip install ipywidgets
   ```
2. Open the Jupyter Notebook or Google Colab.
3. Run each section interactively.

---

## Future Enhancements

🔹 Integrate with a **real-time financial API**. 🔹 Add **visual charts** for better data representation. 🔹 Enhance **loan tracking** with **customized repayment plans**.

---

## License

This project is licensed under the **MIT License**.

---

## Contributors

Developed by **Mohadis Khan**. 🚀

