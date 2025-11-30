document.addEventListener('DOMContentLoaded', () => {
    const tableBody = document.getElementById('tableBody');
    const searchInput = document.getElementById('searchInput');
    const headers = document.querySelectorAll('th');

    let benchmarkData = [];
    let currentSort = { column: 'model', direction: 'asc' };

    // Fetch data
    fetch('data.json')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            benchmarkData = data;
            renderTable(benchmarkData);
        })
        .catch(error => {
            console.error('Error loading data:', error);
            tableBody.innerHTML = '<tr><td colspan="4" style="text-align:center; color: red;">Error loading benchmark data. Please ensure data.json exists.</td></tr>';
        });

    // Render table function
    function renderTable(data) {
        tableBody.innerHTML = '';

        if (data.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="4" style="text-align:center;">No results found</td></tr>';
            return;
        }

        data.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-weight: 500;">${row.model}</td>
                <td>${row.scenario}</td>
                <td style="font-family: monospace;">${row.throughput_tokens_per_sec.toFixed(2)}</td>
                <td style="font-family: monospace;">${row.requests_per_sec.toFixed(2)}</td>
            `;
            tableBody.appendChild(tr);
        });
    }

    // Sort function
    function sortData(column) {
        const direction = currentSort.column === column && currentSort.direction === 'asc' ? 'desc' : 'asc';
        currentSort = { column, direction };

        benchmarkData.sort((a, b) => {
            let valA = a[column];
            let valB = b[column];

            // Handle string comparison
            if (typeof valA === 'string') {
                valA = valA.toLowerCase();
                valB = valB.toLowerCase();
            }

            if (valA < valB) return direction === 'asc' ? -1 : 1;
            if (valA > valB) return direction === 'asc' ? 1 : -1;
            return 0;
        });

        // Update sort icons (visual only)
        headers.forEach(th => {
            const icon = th.querySelector('.sort-icon');
            if (th.dataset.sort === column) {
                icon.textContent = direction === 'asc' ? '↑' : '↓';
                icon.style.opacity = '1';
            } else {
                icon.textContent = '↕';
                icon.style.opacity = '0.5';
            }
        });

        filterAndRender();
    }

    // Filter function
    function filterAndRender() {
        const searchTerm = searchInput.value.toLowerCase();
        const filteredData = benchmarkData.filter(item =>
            item.model.toLowerCase().includes(searchTerm) ||
            item.scenario.toLowerCase().includes(searchTerm)
        );
        renderTable(filteredData);
    }

    // Event listeners
    headers.forEach(th => {
        th.addEventListener('click', () => {
            sortData(th.dataset.sort);
        });
    });

    searchInput.addEventListener('input', filterAndRender);
});
