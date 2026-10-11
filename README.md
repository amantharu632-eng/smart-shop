# Shopping Recommendation System
> **A Complete Data Mining Mini Project using Flask, SQLite, and Association Rule Mining (Apriori Algorithm)**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg)](https://flask.palletsprojects.com/)
[![Mlxtend](https://img.shields.io/badge/Mlxtend-0.23%2B-orange.svg)](http://rasbt.github.io/mlxtend/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey.svg)](https://www.sqlite.org/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple.svg)](https://getbootstrap.com/)

---

## 1. Project Objective

**Shopping Recommendation System** demonstrates the practical application of **Data Mining** in modern e-commerce. Rather than displaying arbitrary or static recommendations, the platform implements **Association Rule Mining** powered by the **Apriori Algorithm** to analyze real customer transaction history and uncover hidden product purchase relationships (*Market Basket Analysis*).

### Real-World Example
If customer transaction records frequently contain:
- `Laptop + Wireless Mouse`
- `Laptop + Laptop Bag`
- `SmartPhone + Phone Cover + Screen Protector`

When a customer views or adds a **Laptop** to their cart, the system dynamically calculates metrics and recommends:
- **Wireless Mouse** (Confidence: 82%, Lift: 2.34x)
- **Laptop Bag** (Confidence: 75%, Lift: 2.10x)

The Data Mining engine is **fully functional, dynamic, and integrated with the database**, not a static mock list.

---

## 2. Key Features

- **Full E-Commerce Experience:**
  - Modern, responsive UI with glassmorphic cards and rich animations.
  - 32 curated sample products across 4 categories (*Electronics, Accessories, Home & Living, Fashion*).
  - Live product search, category filtering, price sorting, and rating filters.
  - Interactive shopping cart with AJAX quantity increment/decrement.
  - Simulated checkout with order summary, promo codes (`SMART20`, `MINIPROJECT10`), and receipt generation.

- **Dynamic Data Mining & Apriori Recommendations:**
  - **Live Apriori Engine:** Analyzes 100+ multi-item transaction baskets from SQLite database using `mlxtend` (with educational pure-Python fallback).
  - **"🎯 Recommended for You" Post-Purchase Engine:** After completing an order, the system extracts the purchased products, identifies matching association rules ($A \rightarrow B$), filters out purchased and out-of-stock items, ranks recommendations by **Confidence $\rightarrow$ Lift $\rightarrow$ Support**, logs the recommendation event, and displays cards on the Thank You page with **Add to Cart**, **View Product**, and **Buy Now** buttons.
  - **"🎯 You May Also Need" on Product Pages:** Real-time antecedent matching with badge: `📊 Recommended based on shopping patterns`.
  - **"🎯 Complete Your Purchase" Cart Complements:** Identifies multi-item antecedents in the cart and recommends cross-sell items.
  - **"🛍️ Because You Bought This":** Personalized homepage and profile recommendation section based on the user's actual order history.
  - **Fallbacks:** If no rules match, falls back to highest-frequency products based on actual transaction item sales counts (no arbitrary hardcoded products).

- **Order History & Buy Again (`/orders`):**
  - Dedicated Order History page showing past orders, date, total, status, and item breakdowns.
  - **"Buy Again" Button:** Re-adds all products from any past order back into the shopping cart in one click.

- **Developer REST API Routes:**
  - `GET /api/recommendations/<product_id>`: Returns Apriori rules and recommended products for a single item.
  - `GET /api/recommendations/order/<tx_id>`: Returns post-purchase recommendations for an entire completed order.
  - `POST /api/recommendations/batch`: Accepts a list of product IDs or names and returns unified multi-item antecedent recommendations.

- **Data Mining Analytics Dashboard (`/dashboard`):**
  - 6 Key Performance Metric cards: Total Transactions, Total Products, Total Users, Total Association Rules, **Total Recommendations Generated**, and Top Consequent.
  - **Most Recommended Products Table:** Real-time ranking of products most frequently recommended across all mined rules with their average lift and confidence.
  - Interactive **Chart.js** charts: Most Purchased Products, Catalog Category Distribution, Top Rules by Lift, and Mining Metric Averages.

- **Admin Console & Dynamic Retraining (`/admin`):**
  - Product Catalog CRUD: Add new products, edit price/stock/description, delete products.
  - Order Transaction log and Registered User directory.
  - **"🔄 Regenerate Recommendation Rules" Button (`/admin/retrain`):** Extracts real transaction data directly from SQLite `transactions` and `transaction_items` tables, executes Apriori, updates the `recommendations` table, and immediately alters future recommendation patterns!

---

## 3. Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend** | HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3, Bootstrap Icons, Chart.js 4.4 |
| **Backend** | Python 3.10+, Flask 3.0+, Werkzeug (Security & Password Hashing) |
| **Database** | SQLite (with foreign keys and cascade constraints) |
| **Data Mining** | Pandas, NumPy, Mlxtend (`TransactionEncoder`, `apriori`, `association_rules`) |

---

## 4. How the Data Mining Recommendation System Works

### The 6-Stage Pipeline
```text
Raw Transaction Data (SQLite / CSV)
        ↓
Data Preprocessing (One-Hot Binary Encoding via TransactionEncoder)
        ↓
Candidate Generation & Apriori Pruning (Anti-Monotonicity Principle)
        ↓
Frequent Itemsets Generation (Support ≥ min_support)
        ↓
Association Rule Mining (Confidence ≥ min_confidence, Lift ≥ min_lift)
        ↓
Metric Ranking & Database Persistence (Saved to 'recommendations' table)
```

### Mathematical Definitions

1. **Support:**
   $$\text{Support}(A \rightarrow B) = \frac{\text{count}(A \cup B)}{N}$$
   *Measures the proportion of transactions that contain both itemsets $A$ and $B$.*

2. **Confidence:**
   $$\text{Confidence}(A \rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$$
   *Measures how often itemset $B$ appears in transactions containing itemset $A$.*

3. **Lift:**
   $$\text{Lift}(A \rightarrow B) = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)} = \frac{\text{Support}(A \cup B)}{\text{Support}(A) \times \text{Support}(B)}$$
   - $\text{Lift} = 1$: Items $A$ and $B$ are independent.
   - $\text{Lift} > 1$: Positive correlation (Customers purchasing $A$ are significantly more likely to purchase $B$).

---

## 5. Project Directory Structure

```text
smart shop/
│
├── app.py                      # Main Flask application and routing
├── init_db.py                  # Database schema setup, product seeding & initial mining
├── verify_project.py           # Automated test suite for routes & algorithms
├── requirements.txt            # Python dependencies
├── database.db                 # SQLite database
│
├── data/
│   └── transactions.csv        # 80+ realistic multi-item transaction records
│
├── recommendation/
│   ├── __init__.py
│   └── apriori_model.py        # Apriori algorithm (Mlxtend + Pure Python engine)
│
├── static/
│   ├── css/
│   │   └── style.css           # Modern design system (glassmorphism, badges, animations)
│   └── js/
│       └── script.js           # AJAX cart, toast notifications, live table filtering
│
├── templates/
│   ├── base.html               # Master layout with navbar, footer, search, and flash messages
│   ├── index.html              # Home page with hero, categories, and recommendation sections
│   ├── products.html           # Catalog page with search, category & sort filters
│   ├── product_details.html    # Product specs and "You May Also Like" with Lift badges
│   ├── cart.html               # Shopping cart with basket complement recommendations
│   ├── checkout.html           # Shipping address & demo payment checkout
│   ├── order_success.html      # Order confirmation receipt and SQLite transaction ID
│   ├── login.html              # User authentication with quick demo autofill buttons
│   ├── register.html           # New account registration
│   ├── profile.html            # User profile, past orders, and shopping affinity
│   ├── recommendations.html    # Academic 6-stage pipeline visualizer & rule simulator
│   ├── dashboard.html          # Data Mining KPI dashboard with Chart.js charts
│   ├── rules.html              # Filterable association rules table with CSV export
│   ├── admin.html              # Admin console (product CRUD & model retraining)
│   └── about.html              # Viva Voce Q&A and presentation demonstration steps
│
└── README.md                   # Complete documentation and presentation guide
```

---

## 6. Database Schema Design (SQLite)

- **`users`**: `id`, `name`, `email`, `password` (hashed), `is_admin`, `created_at`
- **`products`**: `id`, `name`, `category`, `price`, `description`, `image`, `rating`, `stock`, `created_at`
- **`transactions`**: `id`, `user_id`, `total_amount`, `payment_method`, `status`, `created_at`
- **`transaction_items`**: `id`, `transaction_id`, `product_id`, `quantity`, `price`
- **`product_views`**: `id`, `user_id`, `product_id`, `viewed_at`
- **`recommendations`**: `id`, `antecedent`, `consequent`, `antecedent_id`, `consequent_id`, `support`, `confidence`, `lift`, `generated_at`

---

## 7. Step-by-Step Installation & Setup

### Prerequisites
- Python 3.10 or higher installed on your computer.

### Step 1: Open Terminal in Project Directory
```bash
cd "c:\Users\aman tharu\OneDrive\Desktop\smart shop"
```

### Step 2: Create and Activate Virtual Environment (Optional but recommended)
**Windows (PowerShell / Command Prompt):**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 4: Initialize the Database & Seed Apriori Rules
*(Note: `app.py` auto-initializes the database on first run, but you can also run it explicitly)*:
```bash
python init_db.py
```

### Step 5: Run the Verification Suite
```bash
python verify_project.py
```
*(All 10 tests should return `OK`)*

### Step 6: Start the Flask Web Server
```bash
python app.py
```

### Step 7: Open the Website in Your Browser
Visit:
```text
http://127.0.0.1:5000
```

---

## 8. Sample Login Credentials

| Role | Email | Password | Access Privileges |
|---|---|---|---|
| **Administrator** | `admin@smartshop.com` | `admin123` | Full Admin Console, Product CRUD, Model Retraining |
| **Customer (John)** | `john@example.com` | `user123` | Shopping, Checkout, Order History, Profile |
| **Customer (Jane)** | `jane@example.com` | `user123` | Shopping, Checkout, Order History, Profile |

*(One-click "Autofill" buttons are provided on the Login page for convenience during college presentations!)*

---

## 9. College Presentation & Viva Voce Guide

### Recommended 5-Minute Demonstration Walkthrough

### Complete End-to-End Test & Demonstration Scenario

| Step | Action | Expected Result & Data Mining Verification |
|---|---|---|
| **1** | Log in as Customer (`john@example.com` / `password123`) | Logged in with profile and personal session loaded. |
| **2** | Search for **"Laptop"** or open **UltraBook Pro 15** | Product detail page displays specs and the **"🎯 You May Also Need"** section with label **`📊 Recommended based on shopping patterns`**. |
| **3** | Observe Item Recommendations | Shows *Wireless Ergonomic Mouse*, *Laptop Sleeve Bag*, and *Mechanical RGB Keyboard* with Confidence and Lift metric badges. |
| **4** | Click **"Add to Cart"** and view **`/cart`** | Cart displays UltraBook Pro 15 and renders **"🎯 Complete Your Purchase"** recommending basket complement items mined from co-occurrence rules. |
| **5** | Click **"Proceed to Checkout"** and place order | Transaction is created: inserts record in SQLite `transactions`, items in `transaction_items`, and updates product stock. |
| **6** | Automatic Redirect to **Order Success / Thank You Page** | Displays `✓ Order Placed Successfully!`, Order ID, Total Amount, and Purchased Products list. |
| **7** | View **"🎯 Recommended for You"** Post-Purchase Section | Apriori recommendation engine runs: finds rules where antecedent matches purchased products, excludes purchased items and out-of-stock products, ranks by Confidence & Lift. Laptop is **not** recommended back to the user! |
| **8** | Click **"Add to Cart"** on a recommended item | Recommended product (*Wireless Ergonomic Mouse*) is instantly added to active cart without page crash. |
| **9** | Navigate to **Order History (`/orders`)** | Lists recent orders with status badges and a **"Buy Again"** button. Clicking "Buy Again" re-adds past order items to cart! |
| **10** | Log in as Admin (`admin@smartshop.com` / `admin123`) and click **"🔄 Regenerate Recommendation Rules"** | System reads all orders from SQLite `transactions` and `transaction_items`, runs Apriori, updates `recommendations` table, and influences future recommendations! |

---

## 10. Frequently Asked Viva Voce Questions

**Q1: What is the Apriori Property and why is it important?**
> *Answer:* The Apriori Property (anti-monotonicity of support) states that any subset of a frequent itemset must also be frequent. If an itemset is infrequent, all of its supersets will also be infrequent. This allows the algorithm to prune vast combinatorial search spaces without scanning the entire database for every combination.

**Q2: Why is Lift more informative than Confidence alone?**
> *Answer:* Confidence measures $P(B|A)$, but if product $B$ is extremely popular across all transactions (high baseline support), confidence might be high simply due to random chance. Lift normalizes confidence by the base probability of $B$ ($\text{Lift} = \frac{\text{Confidence}}{\text{Support}(B)}$). A Lift $> 1$ confirms a genuine positive correlation.

**Q3: How does the Shopping Recommendation System address the Cold Start problem?**
> *Answer:* When a newly added or infrequent product lacks association rules that meet minimum support thresholds, the Shopping Recommendation System gracefully activates a hybrid fallback, serving the highest-rated popular items within the same category.

---

## 11. Future Scope & Enhancements

1. **FP-Growth Algorithm Integration:** Benchmark Apriori against FP-Growth (Frequent Pattern Tree) for memory performance comparisons on millions of transactions.
2. **Hybrid Collaborative Filtering:** Combine Market Basket Analysis with user-user cosine similarity matrix factorization.
3. **Automated Cron Re-Training:** Schedule automatic nightly model retraining as transaction volumes scale.

---

## 12. Authors & Academic Credits

- **Project Title:** Shopping Recommendation System
- **Domain:** Data Mining / Knowledge Discovery in Databases (KDD) / Web Development
- **Course:** Bachelor of Computer Science / Engineering Mini Project
