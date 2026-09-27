/**
 * SmartShop – Frontend Interactions & Dynamic Recommendations
 */

document.addEventListener('DOMContentLoaded', function () {
  // Initialize Bootstrap Tooltips if available
  if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
      return new bootstrap.Tooltip(tooltipTriggerEl);
    });
  }

  // Auto-dismiss alerts after 5 seconds
  setTimeout(function () {
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(function (alert) {
      if (typeof bootstrap !== 'undefined' && bootstrap.Alert) {
        const bsAlert = new bootstrap.Alert(alert);
        bsAlert.close();
      }
    });
  }, 5000);
});

/**
 * Add Product to Cart via AJAX with Toast notification
 */
function addToCart(productId, quantity = 1, event = null) {
  if (event) {
    event.preventDefault();
    event.stopPropagation();
  }

  fetch(`/cart/add/${productId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Requested-With': 'XMLHttpRequest'
    },
    body: JSON.stringify({ quantity: quantity })
  })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        // Update cart count badge in navbar (both top nav-item and right button)
        const cartBadge = document.getElementById('navbar-cart-count');
        if (cartBadge) {
          cartBadge.innerText = data.total_items;
          cartBadge.classList.remove('d-none');
        }
        const topCartBadge = document.getElementById('nav-top-cart-count');
        if (topCartBadge) {
          topCartBadge.innerText = data.total_items;
          topCartBadge.classList.remove('d-none');
        }

        showToast(data.message || 'Item added to cart!', 'success');
      } else {
        showToast(data.message || 'Could not add to cart.', 'danger');
      }
    })
    .catch(error => {
      console.error('Error adding to cart:', error);
      // Fallback form submit
      window.location.href = `/cart/add/${productId}?qty=${quantity}`;
    });
}

/**
 * Toast Notification Popup
 */
function showToast(message, type = 'success') {
  let toastContainer = document.getElementById('toast-container');
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.id = 'toast-container';
    toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
    toastContainer.style.zIndex = '1100';
    document.body.appendChild(toastContainer);
  }

  const toastId = 'toast-' + Date.now();
  const bgClass = type === 'success' ? 'bg-success' : (type === 'danger' ? 'bg-danger' : 'bg-primary');
  
  const toastHtml = `
    <div id="${toastId}" class="toast align-items-center text-white ${bgClass} border-0 shadow-lg" role="alert" aria-live="assertive" aria-atomic="true">
      <div class="d-flex">
        <div class="toast-body d-flex align-items-center gap-2">
          <i class="bi ${type === 'success' ? 'bi-check-circle-fill' : 'bi-exclamation-circle-fill'}"></i>
          <span>${message}</span>
        </div>
        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
      </div>
    </div>
  `;

  toastContainer.insertAdjacentHTML('beforeend', toastHtml);
  const toastElement = document.getElementById(toastId);
  if (typeof bootstrap !== 'undefined' && bootstrap.Toast) {
    const toast = new bootstrap.Toast(toastElement, { delay: 3500 });
    toast.show();
    toastElement.addEventListener('hidden.bs.toast', function () {
      toastElement.remove();
    });
  }
}

/**
 * Live Table Filter for Association Rules Page
 */
function filterRulesTable() {
  const searchInput = document.getElementById('ruleSearch');
  const minConfInput = document.getElementById('minConfFilter');
  const minLiftInput = document.getElementById('minLiftFilter');
  const table = document.getElementById('rulesTable');

  if (!table) return;

  const searchQuery = searchInput ? searchInput.value.toLowerCase() : '';
  const minConf = minConfInput ? parseFloat(minConfInput.value) || 0 : 0;
  const minLift = minLiftInput ? parseFloat(minLiftInput.value) || 0 : 0;

  const rows = table.querySelectorAll('tbody tr');
  let visibleCount = 0;

  rows.forEach(row => {
    const antText = (row.cells[0]?.textContent || '').toLowerCase();
    const conText = (row.cells[1]?.textContent || '').toLowerCase();
    const confVal = parseFloat(row.getAttribute('data-conf') || '0');
    const liftVal = parseFloat(row.getAttribute('data-lift') || '0');

    const matchesSearch = antText.includes(searchQuery) || conText.includes(searchQuery);
    const matchesConf = confVal >= minConf;
    const matchesLift = liftVal >= minLift;

    if (matchesSearch && matchesConf && matchesLift) {
      row.style.display = '';
      visibleCount++;
    } else {
      row.style.display = 'none';
    }
  });

  const countBadge = document.getElementById('visibleRulesCount');
  if (countBadge) {
    countBadge.innerText = visibleCount;
  }
}
