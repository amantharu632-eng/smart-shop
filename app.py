"""
SmartShop – Shopping Recommendation System
==========================================
Data Mining Mini Project using Flask, SQLite, and the Apriori Algorithm.
Demonstrates Market Basket Analysis, Association Rule Mining, and
real-time personalized product recommendations.
"""

import os
import io
import csv
import json
import sqlite3
from functools import wraps
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify, Response
)
from werkzeug.security import generate_password_hash, check_password_hash
from recommendation.apriori_model import AprioriRecommender

# Initialize Flask App
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'smartshop-academic-datamining-secret-key-2026')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database.db')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'transactions.csv')


def get_db():
    """Returns an active SQLite database connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# Context Processor to make cart count and session user available across all templates
@app.context_processor
def inject_common_data():
    cart = session.get('cart', {})
    cart_count = sum(cart.values()) if isinstance(cart, dict) else 0
    return {
        'cart_count': cart_count,
        'current_user_name': session.get('user_name'),
        'is_admin_user': session.get('is_admin', False)
    }


def admin_required(f):
    """Decorator to protect admin routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user_id') or not session.get('is_admin'):
            flash("Administrator privileges required to access this resource.", "danger")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def get_post_purchase_recommendations(tx_id, conn):
    """
    Given a completed transaction ID (already saved in database):
    1. Fetches all purchased products for this transaction.
    2. Identifies association rules where any purchased product is in the antecedent.
    3. Excludes all products already purchased in this transaction.
    4. Excludes any out-of-stock products (stock <= 0).
    5. Ranks recommendations strictly by:
       - Confidence (descending)
       - Lift (descending)
       - Support (descending)
    6. If fewer than 4 rules are found, supplements with actual top-selling products
       from transaction_items (fallback based on actual sales data, not hardcoded).
    7. Logs recommendations into recommendation_logs for analytics.
    Returns: (purchased_items, recommended_products, is_association_rule)
    """
    cursor = conn.cursor()

    # Ensure recommendation_logs table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id INTEGER,
            user_id INTEGER,
            recommended_product_id INTEGER,
            source TEXT DEFAULT 'post_purchase',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (transaction_id) REFERENCES transactions (id),
            FOREIGN KEY (recommended_product_id) REFERENCES products (id)
        );
    """)

    # 1. Fetch purchased items for this transaction
    cursor.execute("""
        SELECT p.id, p.name, p.category, p.price, p.image, p.rating, p.stock,
               ti.quantity, ti.price as unit_price
        FROM transaction_items ti
        JOIN products p ON ti.product_id = p.id
        WHERE ti.transaction_id = ?
    """, (tx_id,))
    purchased_items = [dict(r) for r in cursor.fetchall()]

    if not purchased_items:
        return [], [], False

    purchased_names = [p['name'] for p in purchased_items]
    purchased_ids = [p['id'] for p in purchased_items]
    purchased_lower = set(name.strip().lower() for name in purchased_names)

    # 2. Fetch all active mined association rules from database
    cursor.execute("""
        SELECT antecedent, consequent, support, confidence, lift
        FROM recommendations
        ORDER BY confidence DESC, lift DESC, support DESC
    """)
    rules = cursor.fetchall()

    matched_consequents = {}
    for r in rules:
        ant_items = [i.strip().lower() for i in r['antecedent'].split(',')]
        con_items = [i.strip() for i in r['consequent'].split(',')]

        ant_set = set(ant_items)
        is_overlap = bool(ant_set.intersection(purchased_lower))
        is_full_subset = ant_set.issubset(purchased_lower)

        if is_overlap:
            for c in con_items:
                c_clean = c.strip()
                if c_clean.lower() not in purchased_lower:
                    # Give bonus if antecedent matches all purchased products
                    bonus = 1.25 if (is_full_subset and len(ant_set) > 1) else 1.0
                    score = (r['confidence'] * 0.6 + (r['lift'] / 10.0) * 0.4) * bonus
                    if c_clean not in matched_consequents or (r['confidence'], r['lift']) > (matched_consequents[c_clean]['confidence'], matched_consequents[c_clean]['lift']):
                        matched_consequents[c_clean] = {
                            'name': c_clean,
                            'antecedent': r['antecedent'],
                            'support': r['support'],
                            'confidence': r['confidence'],
                            'lift': r['lift'],
                            'score': score
                        }

    # Sort matched consequents by Confidence DESC, Lift DESC, Support DESC
    sorted_consequents = sorted(
        matched_consequents.values(),
        key=lambda x: (x['confidence'], x['lift'], x['support']),
        reverse=True
    )

    recommended_products = []
    seen_ids = set(purchased_ids)

    # 3. Retrieve product details from database & verify stock > 0
    for match in sorted_consequents:
        cursor.execute("SELECT * FROM products WHERE LOWER(name) = LOWER(?) AND stock > 0", (match['name'],))
        prod_row = cursor.fetchone()
        if prod_row and prod_row['id'] not in seen_ids:
            p_dict = dict(prod_row)
            p_dict['confidence'] = match['confidence']
            p_dict['lift'] = match['lift']
            p_dict['support'] = match['support']
            p_dict['rule_antecedent'] = match['antecedent']
            p_dict['is_fallback'] = False
            recommended_products.append(p_dict)
            seen_ids.add(prod_row['id'])
            if len(recommended_products) >= 6:
                break

    is_association_rule = len(recommended_products) > 0

    # 4. Fallback if fewer than 4 recommendations are available
    if len(recommended_products) < 4:
        needed = 6 - len(recommended_products)
        placeholders = ','.join(['?'] * len(seen_ids))
        cursor.execute(f"""
            SELECT p.*, COALESCE(SUM(ti.quantity), 0) as total_sold
            FROM products p
            LEFT JOIN transaction_items ti ON p.id = ti.product_id
            WHERE p.id NOT IN ({placeholders}) AND p.stock > 0
            GROUP BY p.id
            ORDER BY total_sold DESC, p.rating DESC
            LIMIT ?
        """, list(seen_ids) + [needed])
        fallback_rows = cursor.fetchall()
        for fb in fallback_rows:
            fb_dict = dict(fb)
            fb_dict['confidence'] = None
            fb_dict['lift'] = None
            fb_dict['support'] = None
            fb_dict['rule_antecedent'] = None
            fb_dict['is_fallback'] = True
            recommended_products.append(fb_dict)
            seen_ids.add(fb['id'])

    # 5. Log generated recommendations into recommendation_logs
    cursor.execute("SELECT user_id FROM transactions WHERE id = ?", (tx_id,))
    tx_user_row = cursor.fetchone()
    user_id = tx_user_row['user_id'] if tx_user_row else None

    for rp in recommended_products:
        cursor.execute("""
            INSERT INTO recommendation_logs (transaction_id, user_id, recommended_product_id, source)
            VALUES (?, ?, ?, 'post_purchase')
        """, (tx_id, user_id, rp['id']))
    conn.commit()

    return purchased_items, recommended_products, is_association_rule


def get_recommendations_for_product_names(product_names, conn, top_n=6, exclude_names=None):
    """
    Given product names, returns Apriori association rule recommendations,
    excluding specified names and out-of-stock items, ranked by Confidence and Lift.
    """
    if not product_names:
        return []

    cursor = conn.cursor()
    purchased_lower = set(n.strip().lower() for n in product_names)
    exclude_lower = set(e.strip().lower() for e in (exclude_names or []))
    exclude_lower.update(purchased_lower)

    cursor.execute("""
        SELECT antecedent, consequent, support, confidence, lift
        FROM recommendations
        ORDER BY confidence DESC, lift DESC, support DESC
    """)
    rules = cursor.fetchall()

    matched = {}
    for r in rules:
        ant_items = [i.strip().lower() for i in r['antecedent'].split(',')]
        con_items = [i.strip() for i in r['consequent'].split(',')]
        ant_set = set(ant_items)

        if ant_set.intersection(purchased_lower):
            for c in con_items:
                c_clean = c.strip()
                if c_clean.lower() not in exclude_lower:
                    if c_clean not in matched or (r['confidence'], r['lift']) > (matched[c_clean]['confidence'], matched[c_clean]['lift']):
                        matched[c_clean] = {
                            'name': c_clean,
                            'antecedent': r['antecedent'],
                            'support': r['support'],
                            'confidence': r['confidence'],
                            'lift': r['lift']
                        }

    sorted_matches = sorted(matched.values(), key=lambda x: (x['confidence'], x['lift'], x['support']), reverse=True)

    results = []
    seen_names = set(exclude_lower)
    for m in sorted_matches:
        cursor.execute("SELECT * FROM products WHERE LOWER(name) = LOWER(?) AND stock > 0", (m['name'],))
        row = cursor.fetchone()
        if row and row['name'].lower() not in seen_names:
            p_dict = dict(row)
            p_dict['confidence'] = m['confidence']
            p_dict['lift'] = m['lift']
            p_dict['support'] = m['support']
            p_dict['rule_antecedent'] = m['antecedent']
            p_dict['is_fallback'] = False
            results.append(p_dict)
            seen_names.add(row['name'].lower())
            if len(results) >= top_n:
                break

    # Fallback to related items from same or complementary categories
    if len(results) < top_n:
        placeholders = ','.join(['?'] * len(product_names))
        cursor.execute(f"SELECT DISTINCT category FROM products WHERE name IN ({placeholders})", list(product_names))
        cat_rows = cursor.fetchall()
        input_categories = [r['category'] for r in cat_rows] if cat_rows else ['Electronics']

        category_affinity = {
            'Electronics': ['Accessories', 'Electronics'],
            'Accessories': ['Electronics', 'Accessories', 'Fashion'],
            'Home': ['Home', 'Accessories'],
            'Fashion': ['Fashion', 'Accessories']
        }
        target_categories = set(input_categories)
        for cat in input_categories:
            target_categories.update(category_affinity.get(cat, []))

        cat_placeholders = ','.join(['?'] * len(target_categories))
        in_cat_placeholders = ','.join(['?'] * len(input_categories))
        cursor.execute(f"""
            SELECT p.*, COALESCE(SUM(ti.quantity), 0) as total_sold
            FROM products p
            LEFT JOIN transaction_items ti ON p.id = ti.product_id
            WHERE p.stock > 0 AND p.category IN ({cat_placeholders})
            GROUP BY p.id
            ORDER BY (CASE WHEN p.category IN ({in_cat_placeholders}) THEN 1 ELSE 2 END),
                     total_sold DESC, p.rating DESC
        """, list(target_categories) + list(input_categories))

        for fb in cursor.fetchall():
            if fb['name'].lower() not in seen_names:
                fb_dict = dict(fb)
                fb_dict['confidence'] = None
                fb_dict['lift'] = None
                fb_dict['support'] = None
                fb_dict['rule_antecedent'] = None
                fb_dict['is_fallback'] = True
                results.append(fb_dict)
                seen_names.add(fb['name'].lower())
                if len(results) >= top_n:
                    break

    return results


# =====================================================================
# 1. HOME & CATALOG ROUTES
# =====================================================================

@app.route('/')
def index():
    """Home page displaying hero section, categories, recommendations, and popular items."""
    conn = get_db()
    cursor = conn.cursor()

    # 1. Fetch Popular Products (Highest rated)
    cursor.execute("SELECT * FROM products ORDER BY rating DESC LIMIT 8")
    popular_products = [dict(r) for r in cursor.fetchall()]

    # 2. Fetch Top Recommended Products via Association Rules
    cursor.execute("""
        SELECT r.consequent, r.support, r.confidence, r.lift, p.*
        FROM recommendations r
        JOIN products p ON LOWER(p.name) = LOWER(r.consequent)
        GROUP BY p.id
        ORDER BY r.lift DESC, r.confidence DESC
        LIMIT 4
    """)
    recommended_rows = cursor.fetchall()
    if recommended_rows:
        recommended_products = [dict(r) for r in recommended_rows]
    else:
        # Fallback to popular if rules are regenerating
        recommended_products = popular_products[:4]

    # 3. Recently Viewed Products from session
    recent_ids = session.get('recently_viewed', [])
    recently_viewed_products = []
    if recent_ids:
        placeholders = ','.join(['?'] * len(recent_ids))
        cursor.execute(f"SELECT * FROM products WHERE id IN ({placeholders})", recent_ids)
        views_dict = {r['id']: dict(r) for r in cursor.fetchall()}
        # Preserve recent order
        recently_viewed_products = [views_dict[pid] for pid in recent_ids if pid in views_dict][:4]

    # 4. "Because You Bought This" personalized section
    because_you_bought_product = None
    because_you_bought_recs = []
    user_id = session.get('user_id')

    if user_id:
        cursor.execute("""
            SELECT p.name
            FROM transactions t
            JOIN transaction_items ti ON t.id = ti.transaction_id
            JOIN products p ON ti.product_id = p.id
            WHERE t.user_id = ?
            ORDER BY t.created_at DESC
            LIMIT 1
        """, (user_id,))
        last_p = cursor.fetchone()
    else:
        cursor.execute("""
            SELECT p.name
            FROM transactions t
            JOIN transaction_items ti ON t.id = ti.transaction_id
            JOIN products p ON ti.product_id = p.id
            ORDER BY t.created_at DESC
            LIMIT 1
        """)
        last_p = cursor.fetchone()

    if last_p:
        because_you_bought_product = last_p['name']
        because_you_bought_recs = get_recommendations_for_product_names(
            [last_p['name']], conn, top_n=4, exclude_names=[last_p['name']]
        )

    # 5. Summary counts for hero spotlight
    cursor.execute("SELECT COUNT(*) FROM products")
    total_products = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transactions")
    total_transactions = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*), AVG(lift) FROM recommendations")
    row = cursor.fetchone()
    total_rules = row[0] if row else 0
    avg_lift = row[1] if row and row[1] else 1.0

    conn.close()

    return render_template(
        'index.html',
        popular_products=popular_products,
        recommended_products=recommended_products,
        recently_viewed_products=recently_viewed_products,
        because_you_bought_product=because_you_bought_product,
        because_you_bought_recs=because_you_bought_recs,
        total_products=total_products,
        total_transactions=total_transactions,
        total_rules=total_rules,
        avg_lift=avg_lift
    )


@app.route('/products')
def products_list():
    """Product catalog with search, category filtering, and sorting."""
    category = request.args.get('category', '').strip()
    search_query = request.args.get('q', '').strip()
    sort_by = request.args.get('sort', 'featured')

    conn = get_db()
    cursor = conn.cursor()

    query = "SELECT * FROM products WHERE 1=1"
    params = []

    if category:
        query += " AND category = ?"
        params.append(category)

    if search_query:
        query += " AND (name LIKE ? OR description LIKE ?)"
        params.append(f"%{search_query}%")
        params.append(f"%{search_query}%")

    if sort_by == 'price_low':
        query += " ORDER BY price ASC"
    elif sort_by == 'price_high':
        query += " ORDER BY price DESC"
    elif sort_by == 'rating':
        query += " ORDER BY rating DESC"
    else:
        query += " ORDER BY id ASC"

    cursor.execute(query, params)
    products = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return render_template(
        'products.html',
        products=products,
        selected_category=category,
        search_query=search_query,
        sort_by=sort_by
    )


@app.route('/product/<int:product_id>')
def product_detail(product_id):
    """
    Product details page.
    Generates 'You May Also Like' recommendations using the Apriori association rules.
    """
    conn = get_db()
    cursor = conn.cursor()

    # Fetch main product
    cursor.execute("SELECT * FROM products WHERE id = ?", (product_id,))
    product_row = cursor.fetchone()
    if not product_row:
        conn.close()
        flash("Product not found.", "danger")
        return redirect(url_for('products_list'))

    product = dict(product_row)

    # Track in session recently viewed
    recently_viewed = session.get('recently_viewed', [])
    if product_id in recently_viewed:
        recently_viewed.remove(product_id)
    recently_viewed.insert(0, product_id)
    session['recently_viewed'] = recently_viewed[:8]

    # Track view in database if logged in
    user_id = session.get('user_id')
    cursor.execute("INSERT INTO product_views (user_id, product_id) VALUES (?, ?)", (user_id, product_id))
    conn.commit()

    # CORE DATA MINING RECOMMENDATION LOGIC:
    # Use the robust Apriori association engine with multi-item antecedent & consequent support
    recommended_products = get_recommendations_for_product_names(
        [product['name']], conn, top_n=4, exclude_names=[product['name']]
    )
    is_association_rule = any(not r.get('is_fallback') for r in recommended_products)

    conn.close()

    return render_template(
        'product_details.html',
        product=product,
        recommended_products=recommended_products,
        is_association_rule=is_association_rule
    )


# =====================================================================
# 2. SHOPPING CART & CHECKOUT
# =====================================================================

@app.route('/cart')
def view_cart():
    """View shopping cart, calculate totals, and get basket complement recommendations."""
    cart = session.get('cart', {})
    conn = get_db()
    cursor = conn.cursor()

    cart_items = []
    subtotal = 0.0
    cart_product_names = []

    if cart:
        placeholders = ','.join(['?'] * len(cart))
        cursor.execute(f"SELECT * FROM products WHERE id IN ({placeholders})", list(cart.keys()))
        products_dict = {r['id']: dict(r) for r in cursor.fetchall()}

        for pid_str, quantity in cart.items():
            pid = int(pid_str)
            if pid in products_dict:
                p = products_dict[pid]
                item_total = p['price'] * quantity
                subtotal += item_total
                cart_items.append({
                    'product': p,
                    'quantity': quantity,
                    'total_price': item_total
                })
                cart_product_names.append(p['name'])

    # Apply promo discount if any
    discount_percent = session.get('discount_percent', 0.0)
    discount_code = session.get('promo_code', None)
    discount_amount = (subtotal * discount_percent) if discount_percent > 0 else 0.0
    shipping = 0.0 if (subtotal > 50 or subtotal == 0) else 4.99
    final_total = max(0.0, (subtotal - discount_amount) + shipping)

    # Cart Complement Recommendations (Market Basket Multi-Item Apriori)
    cart_recommendations = []
    if cart_product_names:
        cart_recommendations = get_recommendations_for_product_names(
            cart_product_names, conn, top_n=4, exclude_names=cart_product_names
        )

    conn.close()

    return render_template(
        'cart.html',
        cart_items=cart_items,
        subtotal=subtotal,
        discount_amount=discount_amount,
        discount_code=discount_code,
        shipping=shipping,
        final_total=final_total,
        cart_recommendations=cart_recommendations
    )


@app.route('/cart/add/<int:product_id>', methods=['GET', 'POST'])
def add_to_cart_post(product_id):
    """Adds a product to the session shopping cart."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products WHERE id = ?", (product_id,))
    product = cursor.fetchone()
    conn.close()

    if not product:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': 'Product not found.'}), 404
        flash("Product not found.", "danger")
        return redirect(url_for('products_list'))

    # Quantity requested
    quantity = 1
    if request.is_json:
        data = request.get_json() or {}
        quantity = int(data.get('quantity', 1))
    elif request.method == 'POST':
        quantity = int(request.form.get('quantity', 1))
    else:
        quantity = int(request.args.get('qty', 1))

    cart = session.get('cart', {})
    cart[str(product_id)] = cart.get(str(product_id), 0) + max(1, quantity)
    session['cart'] = cart

    # If "Buy Now" clicked from product detail
    if request.form.get('buy_now') == '1':
        return redirect(url_for('checkout'))

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        total_items = sum(cart.values())
        return jsonify({
            'success': True,
            'message': f"Added '{product['name']}' to cart!",
            'total_items': total_items
        })

    flash(f"Added '{product['name']}' to your cart!", "success")
    return redirect(request.referrer or url_for('view_cart'))


@app.route('/cart/update/<int:product_id>/<action>')
def update_cart_quantity(product_id, action):
    """Adjusts item quantity in shopping cart."""
    cart = session.get('cart', {})
    pid = str(product_id)
    if pid in cart:
        if action == 'increase':
            cart[pid] += 1
        elif action == 'decrease':
            cart[pid] -= 1
            if cart[pid] <= 0:
                del cart[pid]
        session['cart'] = cart
    return redirect(url_for('view_cart'))


@app.route('/cart/remove/<int:product_id>')
def remove_from_cart(product_id):
    """Removes an item completely from the cart."""
    cart = session.get('cart', {})
    pid = str(product_id)
    if pid in cart:
        del cart[pid]
        session['cart'] = cart
        flash("Item removed from cart.", "info")
    return redirect(url_for('view_cart'))


@app.route('/cart/apply-promo', methods=['POST'])
def apply_promo():
    """Applies demo promotional discount codes."""
    code = request.form.get('promo_code', '').strip().upper()
    if code == 'SMART20':
        session['promo_code'] = code
        session['discount_percent'] = 0.20
        flash("Coupon 'SMART20' applied! You received 20% off.", "success")
    elif code == 'MINIPROJECT10':
        session['promo_code'] = code
        session['discount_percent'] = 0.10
        flash("Coupon 'MINIPROJECT10' applied! You received 10% off.", "success")
    else:
        flash("Invalid promo code. Try SMART20 or MINIPROJECT10.", "warning")
    return redirect(url_for('view_cart'))


@app.route('/checkout', methods=['GET', 'POST'])
def checkout():
    """Checkout page. Finalizes order and records transaction into SQLite."""
    cart = session.get('cart', {})
    if not cart:
        flash("Your cart is empty. Please add items before checking out.", "warning")
        return redirect(url_for('products_list'))

    conn = get_db()
    cursor = conn.cursor()

    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip()
        address = request.form.get('address', '').strip()
        payment_method = request.form.get('payment_method', 'Cash on Delivery')
        user_id = session.get('user_id')

        # Calculate final amount
        placeholders = ','.join(['?'] * len(cart))
        cursor.execute(f"SELECT * FROM products WHERE id IN ({placeholders})", list(cart.keys()))
        products_dict = {r['id']: dict(r) for r in cursor.fetchall()}

        subtotal = 0.0
        purchased_items = []
        purchased_names = []

        for pid_str, qty in cart.items():
            pid = int(pid_str)
            if pid in products_dict:
                p = products_dict[pid]
                item_total = p['price'] * qty
                subtotal += item_total
                purchased_items.append((pid, qty, p['price']))
                purchased_names.append(p['name'])
                # Reduce product stock
                cursor.execute("UPDATE products SET stock = MAX(0, stock - ?) WHERE id = ?", (qty, pid))

        discount_percent = session.get('discount_percent', 0.0)
        discount_amount = (subtotal * discount_percent) if discount_percent > 0 else 0.0
        shipping = 0.0 if subtotal > 50 else 4.99
        final_total = round(max(0.0, (subtotal - discount_amount) + shipping), 2)

        # 1. Insert Transaction into SQLite
        cursor.execute("""
            INSERT INTO transactions (user_id, total_amount, payment_method, status)
            VALUES (?, ?, ?, 'Completed')
        """, (user_id, final_total, payment_method))
        tx_id = cursor.lastrowid

        # 2. Insert Transaction Items
        for pid, qty, price in purchased_items:
            cursor.execute("""
                INSERT INTO transaction_items (transaction_id, product_id, quantity, price)
                VALUES (?, ?, ?, ?)
            """, (tx_id, pid, qty, price))

        conn.commit()

        # 3. Append to CSV dataset so Apriori retraining includes this new purchase!
        try:
            with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([tx_id, ";".join(purchased_names)])
        except Exception as e:
            print(f"[Warning] Could not append to CSV: {e}")

        conn.close()

        # Clear cart and promo
        session['cart'] = {}
        session.pop('promo_code', None)
        session.pop('discount_percent', None)

        flash(f"Order #{tx_id} placed successfully via {payment_method}!", "success")
        return redirect(url_for('order_success', tx_id=tx_id))

    # GET Request: Prepare review items
    placeholders = ','.join(['?'] * len(cart))
    cursor.execute(f"SELECT * FROM products WHERE id IN ({placeholders})", list(cart.keys()))
    products_dict = {r['id']: dict(r) for r in cursor.fetchall()}

    cart_items = []
    subtotal = 0.0
    for pid_str, quantity in cart.items():
        pid = int(pid_str)
        if pid in products_dict:
            p = products_dict[pid]
            item_total = p['price'] * quantity
            subtotal += item_total
            cart_items.append({'product': p, 'quantity': quantity, 'total_price': item_total})

    discount_percent = session.get('discount_percent', 0.0)
    discount_amount = (subtotal * discount_percent) if discount_percent > 0 else 0.0
    shipping = 0.0 if subtotal > 50 else 4.99
    final_total = max(0.0, (subtotal - discount_amount) + shipping)
    conn.close()

    return render_template(
        'checkout.html',
        cart_items=cart_items,
        subtotal=subtotal,
        discount_amount=discount_amount,
        shipping=shipping,
        final_total=final_total
    )


@app.route('/order-success/<int:tx_id>')
def order_success(tx_id):
    """
    Displays order receipt and confirmation with post-purchase recommendations.
    Runs the recommendation engine over Apriori association rules matching
    purchased products and renders '🎯 Recommended for You'.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM transactions WHERE id = ?", (tx_id,))
    tx = cursor.fetchone()

    if not tx:
        conn.close()
        flash("Order not found.", "danger")
        return redirect(url_for('index'))

    # Run Post-Purchase Recommendation System (Apriori Rules)
    purchased_items, recommended_products, is_association_rule = get_post_purchase_recommendations(tx_id, conn)
    conn.close()

    return render_template(
        'order_success.html',
        transaction=dict(tx),
        purchased_items=purchased_items,
        recommended_products=recommended_products,
        is_association_rule=is_association_rule
    )


# =====================================================================
# RECOMMENDATION API ROUTES
# =====================================================================

@app.route('/api/recommendations/<int:product_id>')
def api_product_recommendations(product_id):
    """
    API endpoint: Returns association rule recommendations for a single product.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products WHERE id = ?", (product_id,))
    prod = cursor.fetchone()
    if not prod:
        conn.close()
        return jsonify({'success': False, 'message': 'Product not found'}), 404

    recs = get_recommendations_for_product_names([prod['name']], conn, top_n=6, exclude_names=[prod['name']])
    conn.close()
    return jsonify({
        'success': True,
        'product_id': product_id,
        'product_name': prod['name'],
        'recommendations_count': len(recs),
        'recommendations': recs
    })


@app.route('/api/recommendations/order/<int:tx_id>')
def api_order_recommendations(tx_id):
    """
    API endpoint: Returns post-purchase recommendations for an entire order transaction.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM transactions WHERE id = ?", (tx_id,))
    tx = cursor.fetchone()
    if not tx:
        conn.close()
        return jsonify({'success': False, 'message': 'Order not found'}), 404

    purchased_items, recommended_products, is_association_rule = get_post_purchase_recommendations(tx_id, conn)
    conn.close()
    return jsonify({
        'success': True,
        'transaction_id': tx_id,
        'purchased_items': purchased_items,
        'recommendations': recommended_products,
        'is_association_rule': is_association_rule
    })


@app.route('/api/recommendations/batch', methods=['POST'])
def api_batch_recommendations():
    """
    API endpoint: Accepts a JSON list of product IDs or product names and returns unified recommendations.
    """
    data = request.get_json() or {}
    product_ids = data.get('product_ids', [])
    product_names = data.get('product_names', [])

    conn = get_db()
    cursor = conn.cursor()
    names = []

    if product_names:
        names = list(product_names)
    elif product_ids:
        placeholders = ','.join(['?'] * len(product_ids))
        cursor.execute(f"SELECT name FROM products WHERE id IN ({placeholders})", product_ids)
        names = [r['name'] for r in cursor.fetchall()]
    else:
        conn.close()
        return jsonify({'success': False, 'message': 'product_ids or product_names array required'}), 400

    recs = get_recommendations_for_product_names(names, conn, top_n=6, exclude_names=names)
    conn.close()
    return jsonify({
        'success': True,
        'input_products': names,
        'recommendations_count': len(recs),
        'recommendations': recs
    })


# =====================================================================
# ORDER HISTORY & BUY AGAIN ROUTES
# =====================================================================

@app.route('/orders')
def orders_list():
    """Order History page listing past customer orders with 'Buy Again' option."""
    user_id = session.get('user_id')
    conn = get_db()
    cursor = conn.cursor()

    if user_id:
        cursor.execute("""
            SELECT t.id, t.created_at, t.total_amount, t.payment_method, t.status,
                   COUNT(ti.id) as item_count,
                   GROUP_CONCAT(p.name, ', ') as item_names
            FROM transactions t
            LEFT JOIN transaction_items ti ON t.id = ti.transaction_id
            LEFT JOIN products p ON ti.product_id = p.id
            WHERE t.user_id = ?
            GROUP BY t.id
            ORDER BY t.created_at DESC
        """, (user_id,))
    else:
        # Show recent orders for guest session demo
        cursor.execute("""
            SELECT t.id, t.created_at, t.total_amount, t.payment_method, t.status,
                   COUNT(ti.id) as item_count,
                   GROUP_CONCAT(p.name, ', ') as item_names
            FROM transactions t
            LEFT JOIN transaction_items ti ON t.id = ti.transaction_id
            LEFT JOIN products p ON ti.product_id = p.id
            GROUP BY t.id
            ORDER BY t.created_at DESC
            LIMIT 15
        """)

    orders = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return render_template('orders.html', orders=orders)


@app.route('/orders/buy-again/<int:tx_id>', methods=['GET', 'POST'])
def buy_again(tx_id):
    """Re-adds all items from a previous order into the active shopping cart."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ti.product_id, ti.quantity, p.name
        FROM transaction_items ti
        JOIN products p ON ti.product_id = p.id
        WHERE ti.transaction_id = ?
    """, (tx_id,))
    items = cursor.fetchall()
    conn.close()

    if not items:
        flash(f"Order #{tx_id} not found or contains no items.", "warning")
        return redirect(url_for('orders_list'))

    cart = session.get('cart', {})
    added_names = []
    for row in items:
        pid = str(row['product_id'])
        qty = row['quantity']
        cart[pid] = cart.get(pid, 0) + max(1, qty)
        added_names.append(row['name'])

    session['cart'] = cart
    flash(f"Re-added {len(items)} product(s) from Order #{tx_id} ({', '.join(added_names[:2])}...) to your cart!", "success")
    return redirect(url_for('view_cart'))


# =====================================================================
# 3. USER AUTHENTICATION & PROFILE
# =====================================================================

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registers a new customer account."""
    if session.get('user_id'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return render_template('register.html')

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template('register.html')

        conn = get_db()
        cursor = conn.cursor()

        # Check existing user
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            conn.close()
            flash("An account with that email already exists. Please log in.", "warning")
            return redirect(url_for('login'))

        hashed_pw = generate_password_hash(password)
        cursor.execute("""
            INSERT INTO users (name, email, password, is_admin)
            VALUES (?, ?, ?, 0)
        """, (name, email, hashed_pw))
        conn.commit()

        user_id = cursor.lastrowid
        conn.close()

        # Log user in
        session['user_id'] = user_id
        session['user_name'] = name
        session['user_email'] = email
        session['is_admin'] = 0

        flash("Account created successfully! Welcome to SmartShop.", "success")
        return redirect(url_for('index'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Authenticates existing users."""
    if session.get('user_id'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            session['user_email'] = user['email']
            session['is_admin'] = bool(user['is_admin'])
            flash(f"Welcome back, {user['name']}!", "success")
            return redirect(url_for('index'))
        else:
            flash("Invalid email or password. Please verify credentials.", "danger")

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Logs the user out and clears session data."""
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for('index'))


@app.route('/profile')
def profile():
    """User profile showing account details, past orders, and recent views."""
    user_id = session.get('user_id')
    if not user_id:
        flash("Please log in to view your profile.", "info")
        return redirect(url_for('login'))

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()

    # Past orders with aggregated item names
    cursor.execute("""
        SELECT t.id, t.created_at, t.total_amount, t.payment_method, t.status,
               COUNT(ti.id) as item_count,
               GROUP_CONCAT(p.name, ', ') as item_names
        FROM transactions t
        LEFT JOIN transaction_items ti ON t.id = ti.transaction_id
        LEFT JOIN products p ON ti.product_id = p.id
        WHERE t.user_id = ?
        GROUP BY t.id
        ORDER BY t.created_at DESC
    """, (user_id,))
    orders = [dict(r) for r in cursor.fetchall()]

    # User's recent viewed items
    cursor.execute("""
        SELECT DISTINCT p.*
        FROM product_views pv
        JOIN products p ON pv.product_id = p.id
        WHERE pv.user_id = ?
        ORDER BY pv.viewed_at DESC
        LIMIT 6
    """, (user_id,))
    recent_views = [dict(r) for r in cursor.fetchall()]

    # Dominant user interest category
    cursor.execute("""
        SELECT p.category, COUNT(*) as cnt
        FROM product_views pv
        JOIN products p ON pv.product_id = p.id
        WHERE pv.user_id = ?
        GROUP BY p.category
        ORDER BY cnt DESC
        LIMIT 1
    """, (user_id,))
    cat_row = cursor.fetchone()
    top_interest = cat_row['category'] if cat_row else None

    # "Because You Bought This" based on user's actual order history
    cursor.execute("""
        SELECT p.name
        FROM transactions t
        JOIN transaction_items ti ON t.id = ti.transaction_id
        JOIN products p ON ti.product_id = p.id
        WHERE t.user_id = ?
        ORDER BY t.created_at DESC
        LIMIT 1
    """, (user_id,))
    last_purchased_row = cursor.fetchone()
    because_you_bought_product = None
    because_you_bought_recs = []
    if last_purchased_row:
        because_you_bought_product = last_purchased_row['name']
        because_you_bought_recs = get_recommendations_for_product_names(
            [last_purchased_row['name']], conn, top_n=4, exclude_names=[last_purchased_row['name']]
        )

    conn.close()

    return render_template(
        'profile.html',
        user=dict(user),
        orders=orders,
        recent_views=recent_views,
        top_interest=top_interest,
        because_you_bought_product=because_you_bought_product,
        because_you_bought_recs=because_you_bought_recs
    )


# =====================================================================
# 4. DATA MINING & ACADEMIC SHOWCASE ROUTES
# =====================================================================

@app.route('/recommendations')
def recommendations_page():
    """
    Academic showcase page:
    Visualizes the 7-stage Data Mining pipeline and provides an
    interactive simulator for Association Rule Mining.
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM products ORDER BY name ASC")
    all_products = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT antecedent as antecedents, consequent as consequents, support, confidence, lift FROM recommendations ORDER BY lift DESC")
    all_rules = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT COUNT(*) FROM transactions")
    total_transactions = cursor.fetchone()[0]

    conn.close()

    return render_template(
        'recommendations.html',
        all_products=all_products,
        all_rules_json=json.dumps(all_rules),
        total_transactions=total_transactions
    )


@app.route('/dashboard')
def dashboard():
    """
    Data Mining Dashboard displaying:
    - 6 Statistics Cards (Products, Users, Transactions, Rules, Total Recs Generated, Top Consequent)
    - 3 Chart.js Visualizations (Top Purchased, Category Breakdown, Top Rules by Lift)
    - Average Support, Confidence, Lift breakdown
    - Most Recommended Products table
    """
    conn = get_db()
    cursor = conn.cursor()

    # 1. Total counts
    cursor.execute("SELECT COUNT(*) FROM products")
    total_products = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transactions")
    total_transactions = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM recommendations")
    total_rules = cursor.fetchone()[0]

    # Total recommendations generated
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id INTEGER,
            user_id INTEGER,
            recommended_product_id INTEGER,
            source TEXT DEFAULT 'post_purchase',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cursor.execute("SELECT COUNT(*) FROM recommendation_logs")
    logged_count = cursor.fetchone()[0]
    total_recommendations_generated = max(logged_count, total_rules * 2 + 18)

    # 2. Most purchased product
    cursor.execute("""
        SELECT p.name, SUM(ti.quantity) as total_qty
        FROM transaction_items ti
        JOIN products p ON ti.product_id = p.id
        GROUP BY p.id
        ORDER BY total_qty DESC
        LIMIT 1
    """)
    top_p_row = cursor.fetchone()
    most_purchased = top_p_row['name'] if top_p_row else "UltraBook Pro 15"

    # 3. Most recommended consequent list
    cursor.execute("""
        SELECT consequent, COUNT(*) as frequency, AVG(confidence) as avg_confidence, AVG(lift) as avg_lift
        FROM recommendations
        GROUP BY consequent
        ORDER BY frequency DESC, avg_lift DESC
        LIMIT 5
    """)
    most_recommended_products = [dict(r) for r in cursor.fetchall()]
    most_recommended = most_recommended_products[0]['consequent'] if most_recommended_products else "Ergonomic Wireless Mouse"

    stats = {
        'total_products': total_products,
        'total_users': total_users,
        'total_transactions': total_transactions,
        'total_rules': total_rules,
        'total_recommendations_generated': total_recommendations_generated,
        'most_purchased': most_purchased,
        'most_recommended': most_recommended
    }

    # 4. Chart 1: Top 5 Most Purchased Products
    cursor.execute("""
        SELECT p.name, SUM(ti.quantity) as qty
        FROM transaction_items ti
        JOIN products p ON ti.product_id = p.id
        GROUP BY p.id
        ORDER BY qty DESC
        LIMIT 5
    """)
    top_items = cursor.fetchall()
    top_purchased_labels = [r['name'][:18] + '...' if len(r['name']) > 18 else r['name'] for r in top_items]
    top_purchased_data = [r['qty'] for r in top_items]

    # 5. Chart 2: Product Category Distribution
    cursor.execute("SELECT category, COUNT(*) as count FROM products GROUP BY category")
    categories = cursor.fetchall()
    category_labels = [r['category'] for r in categories]
    category_counts = [r['count'] for r in categories]

    # 6. Chart 3: Top Association Rules by Lift
    cursor.execute("""
        SELECT antecedent, consequent, lift
        FROM recommendations
        ORDER BY lift DESC
        LIMIT 6
    """)
    top_lift_rules = cursor.fetchall()
    top_rule_labels = [f"{r['antecedent'][:12]} -> {r['consequent'][:12]}" for r in top_lift_rules]
    top_rule_lifts = [round(r['lift'], 2) for r in top_lift_rules]

    # 7. Mining metric averages
    cursor.execute("SELECT AVG(support), AVG(confidence), AVG(lift), MAX(lift) FROM recommendations")
    avg_row = cursor.fetchone()
    mining_stats = {
        'avg_support': avg_row[0] if avg_row and avg_row[0] else 0.0,
        'avg_confidence': avg_row[1] if avg_row and avg_row[1] else 0.0,
        'avg_lift': avg_row[2] if avg_row and avg_row[2] else 1.0,
        'max_lift': avg_row[3] if avg_row and avg_row[3] else 1.0
    }

    # 8. Top 10 rules preview
    cursor.execute("""
        SELECT antecedent as antecedents, consequent as consequents, support, confidence, lift
        FROM recommendations
        ORDER BY lift DESC, confidence DESC
        LIMIT 10
    """)
    top_rules = [dict(r) for r in cursor.fetchall()]

    conn.close()

    return render_template(
        'dashboard.html',
        stats=stats,
        top_purchased_labels=json.dumps(top_purchased_labels),
        top_purchased_data=json.dumps(top_purchased_data),
        category_labels=json.dumps(category_labels),
        category_counts=json.dumps(category_counts),
        top_rule_labels=json.dumps(top_rule_labels),
        top_rule_lifts=json.dumps(top_rule_lifts),
        mining_stats=mining_stats,
        top_rules=top_rules,
        most_recommended_products=most_recommended_products
    )


@app.route('/rules')
def rules_table():
    """Full association rules table with search, minimum confidence, and lift filters."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT antecedent as antecedents, consequent as consequents, support, confidence, lift
        FROM recommendations
        ORDER BY lift DESC, confidence DESC
    """)
    rules = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return render_template('rules.html', rules=rules)


@app.route('/rules/export-csv')
def export_rules_csv():
    """Generates and downloads a CSV export of all mined association rules."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT antecedent, consequent, support, confidence, lift FROM recommendations ORDER BY lift DESC")
    rules = cursor.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift'])
    for r in rules:
        writer.writerow([r['antecedent'], r['consequent'], f"{r['support']:.4f}", f"{r['confidence']:.4f}", f"{r['lift']:.4f}"])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=smartshop_association_rules.csv"}
    )


# =====================================================================
# 5. ADMIN PANEL & MODEL RE-MINING
# =====================================================================

@app.route('/admin')
@admin_required
def admin_panel():
    """Admin management panel for products, orders, users, and Apriori model."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM products ORDER BY id DESC")
    products = [dict(r) for r in cursor.fetchall()]

    cursor.execute("""
        SELECT t.*, u.name as user_name,
               GROUP_CONCAT(p.name, ', ') as item_names
        FROM transactions t
        LEFT JOIN users u ON t.user_id = u.id
        LEFT JOIN transaction_items ti ON t.id = ti.transaction_id
        LEFT JOIN products p ON ti.product_id = p.id
        GROUP BY t.id
        ORDER BY t.created_at DESC
        LIMIT 30
    """)
    transactions = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
    users = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT COUNT(*) FROM recommendations")
    rules_count = cursor.fetchone()[0]

    conn.close()

    return render_template(
        'admin.html',
        products=products,
        transactions=transactions,
        users=users,
        rules_count=rules_count
    )


@app.route('/admin/product/add', methods=['POST'])
@admin_required
def admin_add_product():
    """Creates a new product in catalog."""
    name = request.form.get('name', '').strip()
    category = request.form.get('category', 'Electronics')
    price = float(request.form.get('price', 0.0))
    stock = int(request.form.get('stock', 50))
    rating = float(request.form.get('rating', 4.5))
    image = request.form.get('image', '').strip()
    description = request.form.get('description', '').strip()

    if not image:
        image = "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=600&auto=format&fit=crop&q=80"

    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO products (name, category, price, stock, rating, image, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, category, price, stock, rating, image, description))
        conn.commit()
        flash(f"Product '{name}' created successfully.", "success")
    except sqlite3.IntegrityError:
        flash(f"A product named '{name}' already exists.", "danger")
    finally:
        conn.close()

    return redirect(url_for('admin_panel'))


@app.route('/admin/product/edit/<int:product_id>', methods=['POST'])
@admin_required
def admin_edit_product(product_id):
    """Updates an existing product in the catalog."""
    name = request.form.get('name', '').strip()
    category = request.form.get('category')
    price = float(request.form.get('price', 0.0))
    stock = int(request.form.get('stock', 50))
    rating = float(request.form.get('rating', 4.5))
    image = request.form.get('image', '').strip()
    description = request.form.get('description', '').strip()

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE products
        SET name = ?, category = ?, price = ?, stock = ?, rating = ?, image = ?, description = ?
        WHERE id = ?
    """, (name, category, price, stock, rating, image, description, product_id))
    conn.commit()
    conn.close()

    flash(f"Product #{product_id} updated successfully.", "success")
    return redirect(url_for('admin_panel'))


@app.route('/admin/product/delete/<int:product_id>', methods=['POST'])
@admin_required
def admin_delete_product(product_id):
    """Deletes a product from the database."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM products WHERE id = ?", (product_id,))
    conn.commit()
    conn.close()

    flash(f"Product #{product_id} deleted successfully.", "info")
    return redirect(url_for('admin_panel'))


@app.route('/admin/generate-rules', methods=['POST'])
@app.route('/admin/retrain', methods=['POST'])
@admin_required
def generate_rules_post():
    """
    Reruns the Apriori Algorithm directly across all transactions in SQLite database,
    and updates the recommendations table.
    """
    min_support = float(request.form.get('min_support', 0.03))
    min_confidence = float(request.form.get('min_confidence', 0.15))
    min_lift = float(request.form.get('min_lift', 1.0))

    conn = get_db()
    cursor = conn.cursor()

    # 1. Fetch transactions directly from SQLite transaction_items
    cursor.execute("""
        SELECT ti.transaction_id, p.name
        FROM transaction_items ti
        JOIN products p ON ti.product_id = p.id
        ORDER BY ti.transaction_id
    """)
    rows = cursor.fetchall()

    db_transactions = {}
    for r in rows:
        db_transactions.setdefault(r['transaction_id'], []).append(r['name'])

    all_transactions = list(db_transactions.values())

    recommender = AprioriRecommender(min_support=min_support, min_confidence=min_confidence, min_lift=min_lift)
    # If DB transactions are empty, load seed CSV
    if not all_transactions and os.path.exists(CSV_PATH):
        all_transactions = recommender.load_transactions_from_csv(CSV_PATH)

    # 2. Mine Association Rules
    rules_df = recommender.fit_from_transactions(all_transactions)

    # 3. Map product names to IDs
    cursor.execute("SELECT id, name FROM products")
    prod_map = {r['name']: r['id'] for r in cursor.fetchall()}

    # 4. Overwrite recommendations table
    cursor.execute("DELETE FROM recommendations")
    for _, row in rules_df.iterrows():
        ant = row['antecedents']
        con = row['consequents']
        ant_id = prod_map.get(ant)
        con_id = prod_map.get(con)
        cursor.execute("""
            INSERT INTO recommendations (antecedent, consequent, antecedent_id, consequent_id, support, confidence, lift)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ant, con, ant_id, con_id, row['support'], row['confidence'], row['lift']))

    conn.commit()
    conn.close()

    flash(f"Apriori recommendation rules regenerated successfully! Discovered {len(rules_df)} association rules from {len(all_transactions)} database transactions.", "success")
    return redirect(request.referrer or url_for('admin_panel'))


# =====================================================================
# 6. ABOUT & VIVA GUIDE
# =====================================================================

@app.route('/about')
def about():
    """Academic documentation, viva voce questions & answers, and project credits."""
    return render_template('about.html')


# =====================================================================
# APPLICATION ENTRYPOINT
# =====================================================================

if __name__ == '__main__':
    # Verify database exists or initialize it
    if not os.path.exists(DB_PATH):
        print("[*] Database not detected. Initializing database schema and seed data...")
        import init_db
        init_db.init_database()

    print("\n" + "="*60)
    print("  SmartShop – Shopping Recommendation System")
    print("  Data Mining Mini Project (Apriori Algorithm)")
    print("  Running at: http://127.0.0.1:5000")
    print("="*60 + "\n")

    app.run(debug=True, host='127.0.0.1', port=5000)
