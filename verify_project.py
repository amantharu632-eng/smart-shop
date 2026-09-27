"""
Comprehensive Verification Script for SmartShop
Tests Flask routes, Apriori mining, database transactions, post-purchase recommendations,
API endpoints, Order History, Buy Again, and Admin model retraining.
"""

import sys
import json
import unittest
from app import app, get_db

class SmartShopTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

    def test_01_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'SmartShop', response.data)
        self.assertIn(b'Apriori', response.data)
        print("[PASS] 01. Home page loaded successfully.")

    def test_02_products_page(self):
        response = self.client.get('/products')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Product Catalog', response.data)
        self.assertIn(b'UltraBook Pro 15', response.data)
        print("[PASS] 02. Product catalog page loaded with products.")

    def test_03_category_filter(self):
        response = self.client.get('/products?category=Electronics')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Electronics', response.data)
        print("[PASS] 03. Category filtering works.")

    def test_04_product_detail_and_recommendations(self):
        response = self.client.get('/product/1')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'UltraBook Pro 15', response.data)
        self.assertIn(b'You May Also Need', response.data)
        self.assertIn(b'Recommended based on shopping patterns', response.data)
        print("[PASS] 04. Product detail and pattern-based recommendations verified.")

    def test_05_cart_and_post_purchase_flow(self):
        # 1. Add Laptop (ID: 1) to cart
        response = self.client.post('/cart/add/1', json={'quantity': 1}, headers={'X-Requested-With': 'XMLHttpRequest'})
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data['success'])

        # 2. Check cart page shows 'Complete Your Purchase'
        cart_resp = self.client.get('/cart')
        self.assertEqual(cart_resp.status_code, 200)
        self.assertIn(b'UltraBook Pro 15', cart_resp.data)
        self.assertIn(b'Complete Your Purchase', cart_resp.data)
        self.assertIn(b'Recommended based on shopping patterns', cart_resp.data)

        # 3. Checkout and place order
        order_resp = self.client.post('/checkout', data={
            'full_name': 'Aman Buyer',
            'email': 'aman@example.com',
            'address': '456 Tech Park Ave',
            'city': 'Metropolis',
            'postal_code': '560001',
            'payment_method': 'Credit Card'
        }, follow_redirects=True)
        self.assertEqual(order_resp.status_code, 200)

        # 4. Verify Order Success page content
        self.assertIn(b'Order Placed Successfully', order_resp.data)
        self.assertIn(b'Recommended for You', order_resp.data)
        self.assertIn(b'Confidence', order_resp.data)
        self.assertIn(b'Lift', order_resp.data)
        self.assertIn(b'Add to Cart', order_resp.data)
        self.assertIn(b'Buy Now', order_resp.data)
        self.assertIn(b'View Product', order_resp.data)

        # Verify that purchased item (Laptop) is NOT in the recommendations list
        # Extract recommendations section from order_resp.data
        html_text = order_resp.data.decode('utf-8')
        recs_start = html_text.find('Recommended for You')
        self.assertTrue(recs_start != -1)
        recs_section = html_text[recs_start:]
        # UltraBook Pro 15 should not be recommended for someone who just bought UltraBook Pro 15
        # (It appears in the purchased summary, but not as a recommended product card title)
        self.assertNotIn('UltraBook Pro 15</h4>', recs_section)
        self.assertNotIn('UltraBook Pro 15</h5>', recs_section)
        print("[PASS] 05. Cart recommendations and Post-Purchase recommendations verified.")

    def test_06_multi_product_purchase_recommendations(self):
        with self.client.session_transaction() as sess:
            sess['cart'] = {'1': 1, '2': 1}  # UltraBook Pro 15 and Wireless Mouse

        order_resp = self.client.post('/checkout', data={
            'full_name': 'Multi Item Buyer',
            'email': 'multi@example.com',
            'address': '789 Commerce Way',
            'city': 'Silicon Valley',
            'postal_code': '94016',
            'payment_method': 'UPI'
        }, follow_redirects=True)
        self.assertEqual(order_resp.status_code, 200)
        self.assertIn(b'Order Placed Successfully', order_resp.data)
        html_text = order_resp.data.decode('utf-8')
        recs_start = html_text.find('Recommended for You')
        recs_section = html_text[recs_start:]
        
        # Neither UltraBook Pro 15 nor Wireless Mouse should be in the recommended items
        self.assertNotIn('UltraBook Pro 15</h5>', recs_section)
        self.assertNotIn('Wireless Ergonomic Mouse</h5>', recs_section)
        print("[PASS] 06. Multi-product purchase filters purchased items from recommendations.")

    def test_07_api_endpoints(self):
        # 1. Product recommendation API
        resp1 = self.client.get('/api/recommendations/1')
        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.get_json()
        self.assertTrue(data1['success'])
        self.assertEqual(data1['product_name'], 'UltraBook Pro 15')
        self.assertTrue(len(data1['recommendations']) > 0)
        self.assertIn('confidence', data1['recommendations'][0])
        self.assertIn('lift', data1['recommendations'][0])

        # 2. Batch recommendation API
        resp2 = self.client.post('/api/recommendations/batch', json={
            'product_names': ['UltraBook Pro 15', 'Wireless Ergonomic Mouse']
        })
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.get_json()
        self.assertTrue(data2['success'])
        # Ensure excluded items are not in recommendations
        rec_names = [r['name'] for r in data2['recommendations']]
        self.assertNotIn('UltraBook Pro 15', rec_names)
        self.assertNotIn('Wireless Ergonomic Mouse', rec_names)

        # 3. Order recommendations API
        # Fetch the latest transaction id from db
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM transactions ORDER BY id DESC LIMIT 1")
        last_tx = cursor.fetchone()
        conn.close()
        if last_tx:
            resp3 = self.client.get(f'/api/recommendations/order/{last_tx[0]}')
            self.assertEqual(resp3.status_code, 200)
            data3 = resp3.get_json()
            self.assertTrue(data3['success'])
            self.assertEqual(data3['transaction_id'], last_tx[0])

        print("[PASS] 07. REST API endpoints (/api/recommendations/*) verified.")

    def test_08_order_history_and_buy_again(self):
        # Login user
        self.client.post('/login', data={'email': 'john@example.com', 'password': 'password123'}, follow_redirects=True)
        
        # Add item and checkout as logged-in user
        self.client.post('/cart/add/3', json={'quantity': 1}, headers={'X-Requested-With': 'XMLHttpRequest'})
        self.client.post('/checkout', data={
            'full_name': 'John Doe',
            'email': 'john@example.com',
            'address': '456 Tech Park',
            'city': 'San Jose',
            'postal_code': '95112',
            'payment_method': 'Cash on Delivery'
        }, follow_redirects=True)

        # Visit /orders
        orders_resp = self.client.get('/orders')
        self.assertEqual(orders_resp.status_code, 200)
        self.assertIn(b'My Orders', orders_resp.data)
        self.assertIn(b'Buy Again', orders_resp.data)

        # Test Buy Again
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM transactions WHERE user_id = 2 ORDER BY id DESC LIMIT 1")
        user_tx = cursor.fetchone()
        conn.close()

        if user_tx:
            buy_again_resp = self.client.get(f'/orders/buy-again/{user_tx[0]}', follow_redirects=True)
            self.assertEqual(buy_again_resp.status_code, 200)
            self.assertIn(b'from Order #', buy_again_resp.data)
            self.assertIn(b'Shopping Cart', buy_again_resp.data)

        print("[PASS] 08. Order History page and Buy Again flow verified.")

    def test_09_dashboard_and_recommendation_kpis(self):
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Total Recommendations Generated', response.data)
        self.assertIn(b'Most Recommended Products', response.data)
        self.assertIn(b'Data Mining &amp; Store Analytics Dashboard', response.data)
        print("[PASS] 09. Data Mining dashboard with recommendation KPIs verified.")

    def test_10_admin_retrain_from_database(self):
        # Login as admin
        self.client.post('/login', data={'email': 'admin@smartshop.com', 'password': 'admin123'}, follow_redirects=True)
        
        # Trigger Retrain
        retrain_resp = self.client.post('/admin/retrain', follow_redirects=True)
        self.assertEqual(retrain_resp.status_code, 200)
        self.assertIn(b'Apriori recommendation rules regenerated successfully', retrain_resp.data)
        print("[PASS] 10. Admin database retraining pipeline verified.")

if __name__ == '__main__':
    unittest.main()
