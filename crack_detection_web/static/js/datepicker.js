/* =============================================
   DATEPICKER.JS — Bộ chọn ngày sinh tùy chỉnh
   Chế độ "ngày": lưới ngày trong tháng, mũi tên trái/phải đổi tháng.
   Chế độ "tháng": ấn vào nhãn tháng/năm để chuyển sang lưới 12 tháng,
                   mũi tên trái/phải lúc này đổi năm.
   ============================================= */
(function () {
    const MONTHS_VI = [
        'Tháng 1', 'Tháng 2', 'Tháng 3', 'Tháng 4', 'Tháng 5', 'Tháng 6',
        'Tháng 7', 'Tháng 8', 'Tháng 9', 'Tháng 10', 'Tháng 11', 'Tháng 12',
    ];
    const MONTHS_SHORT_VI = [
        'Th 1', 'Th 2', 'Th 3', 'Th 4', 'Th 5', 'Th 6',
        'Th 7', 'Th 8', 'Th 9', 'Th 10', 'Th 11', 'Th 12',
    ];
    const WEEKDAYS_VI = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'CN'];

    function pad(n) { return n < 10 ? '0' + n : '' + n; }
    function daysInMonth(year, month) { return new Date(year, month + 1, 0).getDate(); }

    function initDatePicker(root) {
        const displayInput = root.querySelector('.dp-display-input');
        const hiddenInput  = root.querySelector('.dp-hidden-input');
        const calendarBtn  = root.querySelector('.dp-calendar-btn');
        const panel        = root.querySelector('.dp-panel');

        const minYear = parseInt(root.dataset.minYear || '1900', 10);
        const maxYear = parseInt(root.dataset.maxYear || String(new Date().getFullYear()), 10);

        let selectedDate = null;   // { y, m (0-11), d }
        let viewYear, viewMonth;
        let mode = 'day';          // 'day' | 'month'
        let isOpen = false;

        const initVal = (hiddenInput.value || '').trim();
        if (initVal) {
            const parts = initVal.split('-').map(Number);
            if (parts.length === 3 && !parts.some(isNaN)) {
                const [y, m, d] = parts;
                selectedDate = { y, m: m - 1, d };
                viewYear = y;
                viewMonth = m - 1;
                displayInput.value = `${pad(d)}/${pad(m)}/${y}`;
            }
        }
        if (viewYear === undefined) {
            const now = new Date();
            viewYear = Math.min(Math.max(now.getFullYear(), minYear), maxYear);
            viewMonth = now.getMonth();
        }

        function render() {
            panel.innerHTML = '';

            const header = document.createElement('div');
            header.className = 'dp-header';

            const prevBtn = document.createElement('button');
            prevBtn.type = 'button';
            prevBtn.className = 'dp-nav-btn';
            prevBtn.innerHTML = '&#8249;';
            prevBtn.setAttribute('aria-label', 'Trước');

            const nextBtn = document.createElement('button');
            nextBtn.type = 'button';
            nextBtn.className = 'dp-nav-btn';
            nextBtn.innerHTML = '&#8250;';
            nextBtn.setAttribute('aria-label', 'Sau');

            const label = document.createElement('button');
            label.type = 'button';
            label.className = 'dp-label-btn';

            if (mode === 'day') {
                label.textContent = `${MONTHS_VI[viewMonth]} năm ${viewYear}`;
                prevBtn.disabled = (viewYear === minYear && viewMonth === 0);
                nextBtn.disabled = (viewYear === maxYear && viewMonth === 11);
                prevBtn.addEventListener('click', (e) => { e.stopPropagation(); shiftMonth(-1); });
                nextBtn.addEventListener('click', (e) => { e.stopPropagation(); shiftMonth(1); });
            } else {
                label.textContent = `${viewYear}`;
                prevBtn.disabled = (viewYear <= minYear);
                nextBtn.disabled = (viewYear >= maxYear);
                prevBtn.addEventListener('click', (e) => { e.stopPropagation(); shiftYear(-1); });
                nextBtn.addEventListener('click', (e) => { e.stopPropagation(); shiftYear(1); });
            }
            label.addEventListener('click', (e) => {
                e.stopPropagation();
                mode = (mode === 'day') ? 'month' : 'day';
                render();
            });

            header.appendChild(prevBtn);
            header.appendChild(label);
            header.appendChild(nextBtn);
            panel.appendChild(header);

            if (mode === 'day') renderDayGrid(); else renderMonthGrid();
        }

        function renderDayGrid() {
            const weekRow = document.createElement('div');
            weekRow.className = 'dp-weekday-row';
            WEEKDAYS_VI.forEach((w) => {
                const el = document.createElement('span');
                el.className = 'dp-weekday';
                el.textContent = w;
                weekRow.appendChild(el);
            });
            panel.appendChild(weekRow);

            const grid = document.createElement('div');
            grid.className = 'dp-day-grid';

            const firstDay = new Date(viewYear, viewMonth, 1).getDay(); // 0 = CN
            const offset = (firstDay === 0) ? 6 : firstDay - 1;         // đưa về Thứ 2 đầu tuần
            const totalDays = daysInMonth(viewYear, viewMonth);
            const today = new Date();

            for (let i = 0; i < offset; i++) {
                const empty = document.createElement('span');
                empty.className = 'dp-day dp-day-empty';
                grid.appendChild(empty);
            }

            for (let d = 1; d <= totalDays; d++) {
                const cell = document.createElement('button');
                cell.type = 'button';
                cell.className = 'dp-day';
                cell.textContent = d;

                if (selectedDate && selectedDate.y === viewYear &&
                    selectedDate.m === viewMonth && selectedDate.d === d) {
                    cell.classList.add('dp-day-selected');
                }
                if (viewYear === today.getFullYear() && viewMonth === today.getMonth() && d === today.getDate()) {
                    cell.classList.add('dp-day-today');
                }

                cell.addEventListener('click', (e) => {
                    e.stopPropagation();
                    selectedDate = { y: viewYear, m: viewMonth, d };
                    hiddenInput.value = `${viewYear}-${pad(viewMonth + 1)}-${pad(d)}`;
                    displayInput.value = `${pad(d)}/${pad(viewMonth + 1)}/${viewYear}`;
                    hiddenInput.dispatchEvent(new Event('change', { bubbles: true }));
                    closePanel();
                });
                grid.appendChild(cell);
            }

            panel.appendChild(grid);
        }

        function renderMonthGrid() {
            const grid = document.createElement('div');
            grid.className = 'dp-month-grid';
            MONTHS_SHORT_VI.forEach((label, idx) => {
                const cell = document.createElement('button');
                cell.type = 'button';
                cell.className = 'dp-month-cell';
                cell.textContent = label;
                if (idx === viewMonth && viewYear === (selectedDate ? selectedDate.y : viewYear)) {
                    cell.classList.add('dp-month-selected');
                }
                cell.addEventListener('click', (e) => {
                    e.stopPropagation();
                    viewMonth = idx;
                    mode = 'day';
                    render();
                });
                grid.appendChild(cell);
            });
            panel.appendChild(grid);
        }

        function shiftMonth(delta) {
            let m = viewMonth + delta;
            let y = viewYear;
            if (m < 0) { m = 11; y -= 1; }
            if (m > 11) { m = 0; y += 1; }
            if (y < minYear || y > maxYear) return;
            viewMonth = m; viewYear = y;
            render();
        }

        function shiftYear(delta) {
            const y = viewYear + delta;
            if (y < minYear || y > maxYear) return;
            viewYear = y;
            render();
        }

        function openPanel() {
            if (isOpen) return;
            isOpen = true;
            mode = 'day';
            panel.hidden = false;
            root.classList.add('dp-open');
            render();
            document.addEventListener('click', onDocClick, true);
        }
        function closePanel() {
            isOpen = false;
            panel.hidden = true;
            root.classList.remove('dp-open');
            document.removeEventListener('click', onDocClick, true);
        }
        function onDocClick(e) {
            if (!root.contains(e.target)) closePanel();
        }

        displayInput.addEventListener('click', (e) => {
            e.stopPropagation();
            isOpen ? closePanel() : openPanel();
        });
        calendarBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            isOpen ? closePanel() : openPanel();
        });
    }

    document.addEventListener('DOMContentLoaded', function () {
        document.querySelectorAll('[data-datepicker]').forEach(initDatePicker);
    });
})();