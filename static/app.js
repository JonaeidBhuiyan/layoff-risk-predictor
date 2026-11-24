/* =========================================================
   WORKFORCE INTELLIGENCE — LAYOFF RISK PREDICTOR
   Frontend aligned with the supplied FastAPI backend.
   API: /api/health, /api/options, /api/predict
   ========================================================= */

'use strict';

/* ---------------------------------------------------------
   1. CONFIGURATION
--------------------------------------------------------- */

// Leave empty when the frontend is served by the same FastAPI app.
// For a separately hosted frontend, set this to the backend origin,
// for example: https://your-api.onrender.com (without a trailing slash).
const API_BASE_URL = 'https://layoff-risk-predictor.onrender.com';
const HISTORY_KEY = 'workforce-intelligence-layoff-history-v1';
const MAX_HISTORY_ITEMS = 12;

const apiUrl = (path) => `${API_BASE_URL}${path}`;


/* ---------------------------------------------------------
   2. FORM SCHEMA — MATCHES CompanyData IN main.py
--------------------------------------------------------- */

const FIELDS = [
    {
        title: 'Company & operating environment',
        items: [
            { key: 'year', label: 'Year', type: 'number', min: 2005, max: 2026, step: 1, default: 2024 },
            { key: 'quarter', label: 'Quarter', type: 'select', option: 'quarter', default: 'Q1' },
            { key: 'industry', label: 'Industry', type: 'select', option: 'industry', default: 'Software' },
            { key: 'region', label: 'Region', type: 'select', option: 'region', default: 'North America' },
            { key: 'department', label: 'Department', type: 'select', option: 'department', default: 'Engineering' },
            { key: 'economic_condition', label: 'Economic condition', type: 'select', option: 'economic_condition', default: 'Stable' },
            { key: 'work_model', label: 'Work model', type: 'select', option: 'work_model', default: 'Hybrid' }
        ]
    },
    {
        title: 'Economic indicators',
        items: [
            { key: 'inflation_rate', label: 'Inflation rate', suffix: '%', type: 'range', min: 0, max: 20, step: 0.1, default: 4 },
            { key: 'interest_rate', label: 'Interest rate', suffix: '%', type: 'range', min: 0, max: 15, step: 0.1, default: 3.5 },
            { key: 'gdp_growth', label: 'GDP growth', suffix: '%', type: 'range', min: -10, max: 15, step: 0.1, default: 3 }
        ]
    },
    {
        title: 'AI adoption & workforce',
        items: [
            { key: 'ai_adoption_score', label: 'AI adoption score', type: 'range', min: 0, max: 100, step: 1, default: 30 },
            { key: 'automation_level', label: 'Automation level', type: 'range', min: 0, max: 100, step: 1, default: 30 },
            { key: 'employees_hired', label: 'Employees hired', type: 'range', min: 100, max: 12000, step: 50, default: 6000 },
            { key: 'attrition_rate', label: 'Attrition rate', suffix: '%', type: 'range', min: 0, max: 50, step: 0.1, default: 16 },
            { key: 'employee_satisfaction', label: 'Employee satisfaction', type: 'range', min: 0, max: 100, step: 0.1, default: 65, optional: true },
            { key: 'productivity_index', label: 'Productivity index', type: 'range', min: 0, max: 100, step: 0.1, default: 68 },
            { key: 'innovation_score', label: 'Innovation score', type: 'range', min: 0, max: 100, step: 0.1, default: 71 }
        ]
    },
    {
        title: 'Financial performance',
        items: [
            { key: 'avg_salary', label: 'Average salary', prefix: '$', type: 'range', min: 0, max: 250000, step: 500, default: 78000, optional: true },
            { key: 'revenue_growth', label: 'Revenue growth', suffix: '%', type: 'range', min: -50, max: 80, step: 0.1, default: 5, optional: true },
            { key: 'stock_growth', label: 'Stock growth', suffix: '%', type: 'range', min: -70, max: 100, step: 0.1, default: 5 }
        ]
    }
];


/* ---------------------------------------------------------
   3. SAMPLE SCENARIOS
--------------------------------------------------------- */

const PRESETS = {
    healthy: {
        year: 2024,
        quarter: 'Q2',
        industry: 'Cloud Computing',
        region: 'North America',
        department: 'Engineering',
        economic_condition: 'Stable',
        work_model: 'Hybrid',
        inflation_rate: 2.5,
        interest_rate: 3,
        gdp_growth: 4.5,
        ai_adoption_score: 25,
        automation_level: 20,
        employees_hired: 7000,
        attrition_rate: 8,
        employee_satisfaction: 85,
        productivity_index: 82,
        innovation_score: 88,
        avg_salary: 95000,
        revenue_growth: 18,
        stock_growth: 14
    },
    pressure: {
        year: 2024,
        quarter: 'Q3',
        industry: 'E-Commerce',
        region: 'North America',
        department: 'Operations',
        economic_condition: 'Recession',
        work_model: 'Onsite',
        inflation_rate: 8,
        interest_rate: 8,
        gdp_growth: -2,
        ai_adoption_score: 55,
        automation_level: 60,
        employees_hired: 2500,
        attrition_rate: 28,
        employee_satisfaction: 42,
        productivity_index: 48,
        innovation_score: 40,
        avg_salary: 65000,
        revenue_growth: -18,
        stock_growth: -25
    },
    disrupted: {
        year: 2024,
        quarter: 'Q4',
        industry: 'Software',
        region: 'North America',
        department: 'Engineering',
        economic_condition: 'Slowdown',
        work_model: 'Remote',
        inflation_rate: 5,
        interest_rate: 5.5,
        gdp_growth: 1,
        ai_adoption_score: 92,
        automation_level: 88,
        employees_hired: 1200,
        attrition_rate: 22,
        employee_satisfaction: 55,
        productivity_index: 70,
        innovation_score: 76,
        avg_salary: 88000,
        revenue_growth: 2,
        stock_growth: -12
    }
};


/* ---------------------------------------------------------
   4. APPLICATION STATE & HELPERS
--------------------------------------------------------- */

let modelOptions = {};
let latestResult = null;
let latestPayload = null;
let formReady = false;

const byId = (id) => document.getElementById(id);

function formatNumber(value, maximumFractionDigits = 1) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '—';
    return new Intl.NumberFormat(undefined, {
        maximumFractionDigits,
        minimumFractionDigits: 0
    }).format(number);
}

function showError(message) {
    const error = byId('form-error');
    error.textContent = message;
    error.hidden = false;
}

function clearError() {
    const error = byId('form-error');
    error.textContent = '';
    error.hidden = true;
}

function setConnectionStatus(state, message) {
    const label = byId('connection-label');
    const status = byId('connection-dot')?.parentElement;
    if (!label || !status) return;

    label.textContent = message;
    status.classList.remove('is-online', 'is-offline');
    if (state === 'online') status.classList.add('is-online');
    if (state === 'offline') status.classList.add('is-offline');
}

function getOptionalToggle(key) {
    return document.querySelector(`[data-optional="${key}"]`);
}

function getFieldControl(key) {
    return byId(`field-${key}`);
}

function getFieldDefinition(key) {
    for (const section of FIELDS) {
        const found = section.items.find((item) => item.key === key);
        if (found) return found;
    }
    return null;
}


/* ---------------------------------------------------------
   5. BUILD THE FORM FROM THE SCHEMA
--------------------------------------------------------- */

function buildForm() {
    const host = byId('form-sections');
    host.replaceChildren();

    FIELDS.forEach((section, sectionIndex) => {
        const sectionElement = document.createElement('section');
        sectionElement.className = 'field-section';

        const heading = document.createElement('div');
        heading.className = 'section-title';

        const number = document.createElement('span');
        number.textContent = String(sectionIndex + 1).padStart(2, '0');

        const title = document.createElement('h3');
        title.textContent = section.title;

        heading.append(number, title);

        const grid = document.createElement('div');
        grid.className = 'field-grid';

        section.items.forEach((field) => {
            grid.appendChild(buildField(field));
        });

        sectionElement.append(heading, grid);
        host.appendChild(sectionElement);
    });
}

function buildField(field) {
    const wrapper = document.createElement('div');
    wrapper.className = `field field-${field.key}`;

    const label = document.createElement('label');
    label.htmlFor = `field-${field.key}`;
    label.textContent = field.label;
    wrapper.appendChild(label);

    if (field.type === 'select') {
        const select = document.createElement('select');
        select.id = `field-${field.key}`;
        select.name = field.key;
        select.required = true;
        select.setAttribute('aria-label', field.label);

        const values = modelOptions[field.option] || [];
        values.forEach((value) => {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = value;
            select.appendChild(option);
        });

        if (values.includes(field.default)) {
            select.value = field.default;
        }

        wrapper.appendChild(select);
    } else if (field.type === 'range') {
        const controls = document.createElement('div');
        controls.className = 'range-controls';

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.id = `field-${field.key}`;
        slider.name = field.key;
        slider.min = String(field.min);
        slider.max = String(field.max);
        slider.step = String(field.step);
        slider.value = String(field.default);
        slider.setAttribute('aria-label', `${field.label} slider`);

        const exact = document.createElement('input');
        exact.type = 'number';
        exact.className = 'range-number';
        exact.min = String(field.min);
        exact.max = String(field.max);
        exact.step = String(field.step);
        exact.value = String(field.default);
        exact.inputMode = 'decimal';
        exact.setAttribute('aria-label', `${field.label} exact value`);

        slider.addEventListener('input', () => {
            exact.value = slider.value;
            updateRangeFill(slider);
        });

        exact.addEventListener('input', () => {
            if (exact.value === '') return;
            const value = Number(exact.value);
            if (Number.isFinite(value) && value >= field.min && value <= field.max) {
                slider.value = String(value);
                updateRangeFill(slider);
            }
        });

        exact.addEventListener('change', () => {
            if (exact.value === '') return;
            const raw = Number(exact.value);
            if (!Number.isFinite(raw)) {
                exact.value = slider.value;
                return;
            }
            const bounded = Math.min(field.max, Math.max(field.min, raw));
            exact.value = String(bounded);
            slider.value = String(bounded);
            updateRangeFill(slider);
        });

        controls.append(slider, exact);
        wrapper.appendChild(controls);
        updateRangeFill(slider);
    } else {
        const input = document.createElement('input');
        input.type = 'number';
        input.id = `field-${field.key}`;
        input.name = field.key;
        input.min = String(field.min);
        input.max = String(field.max);
        input.step = String(field.step);
        input.value = String(field.default);
        input.required = true;
        wrapper.appendChild(input);
    }

    if (field.optional) {
        const optionalLabel = document.createElement('label');
        optionalLabel.className = 'optional-toggle';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = true;
        checkbox.dataset.optional = field.key;
        checkbox.setAttribute('aria-label', `Include ${field.label}`);

        checkbox.addEventListener('change', () => {
            setOptionalFieldEnabled(field.key, checkbox.checked);
        });

        const helper = document.createElement('span');
        helper.textContent = 'Include value';
        optionalLabel.append(checkbox, helper);
        wrapper.appendChild(optionalLabel);
    }

    return wrapper;
}

function updateRangeFill(slider) {
    const min = Number(slider.min);
    const max = Number(slider.max);
    const value = Number(slider.value);
    const percentage = max === min ? 0 : ((value - min) / (max - min)) * 100;
    slider.style.setProperty('--range-progress', `${Math.min(100, Math.max(0, percentage))}%`);
}

function setOptionalFieldEnabled(key, enabled) {
    const control = getFieldControl(key);
    const wrapper = control?.closest('.field');
    const pairedNumber = wrapper?.querySelector('.range-number');
    if (control) control.disabled = !enabled;
    if (pairedNumber) pairedNumber.disabled = !enabled;
}


/* ---------------------------------------------------------
   6. READ AND VALIDATE THE FORM
--------------------------------------------------------- */

function readForm() {
    const payload = {};

    for (const section of FIELDS) {
        for (const field of section.items) {
            const optionalToggle = field.optional ? getOptionalToggle(field.key) : null;

            if (field.optional && optionalToggle && !optionalToggle.checked) {
                payload[field.key] = null;
                continue;
            }

            const control = getFieldControl(field.key);
            if (!control) {
                throw new Error(`The form is missing the ${field.label} field. Reload the page and try again.`);
            }

            if (field.type === 'select') {
                const selected = control.value;
                if (!selected) throw new Error(`Please select ${field.label.toLowerCase()}.`);
                payload[field.key] = selected;
                continue;
            }

            const wrapper = control.closest('.field');
            const exactInput = field.type === 'range' ? wrapper?.querySelector('.range-number') : null;
            const rawValue = exactInput ? exactInput.value : control.value;

            if (rawValue === '') {
                throw new Error(`Please enter a value for ${field.label.toLowerCase()}.`);
            }

            const value = Number(rawValue);
            if (!Number.isFinite(value)) {
                throw new Error(`${field.label} must be a valid number.`);
            }
            if (value < field.min || value > field.max) {
                throw new Error(`${field.label} must be between ${field.min} and ${field.max}.`);
            }
            if (Number.isInteger(field.step) && !Number.isInteger(value)) {
                throw new Error(`${field.label} must be a whole number.`);
            }

            payload[field.key] = value;
        }
    }

    return payload;
}


/* ---------------------------------------------------------
   7. SET FORM VALUES / PRESETS / RESET
--------------------------------------------------------- */

function setForm(values) {
    for (const section of FIELDS) {
        for (const field of section.items) {
            const control = getFieldControl(field.key);
            if (!control) continue;

            const value = values[field.key];
            const wrapper = control.closest('.field');
            const exactInput = field.type === 'range' ? wrapper?.querySelector('.range-number') : null;
            const optionalToggle = field.optional ? getOptionalToggle(field.key) : null;

            if (field.optional && value === null) {
                if (optionalToggle) optionalToggle.checked = false;
                setOptionalFieldEnabled(field.key, false);
                continue;
            }

            if (value === undefined || value === null) continue;

            if (field.type === 'select') {
                const hasOption = Array.from(control.options).some((option) => option.value === String(value));
                if (hasOption) control.value = String(value);
                continue;
            }

            control.value = String(value);
            if (exactInput) exactInput.value = String(value);
            if (field.type === 'range') updateRangeFill(control);

            if (field.optional && optionalToggle) {
                optionalToggle.checked = true;
                setOptionalFieldEnabled(field.key, true);
            }
        }
    }
}

function getDefaultValues() {
    const defaults = {};
    for (const section of FIELDS) {
        for (const field of section.items) defaults[field.key] = field.default;
    }
    return defaults;
}

function resetForm() {
    if (!formReady) return;
    setForm(getDefaultValues());
    clearError();
    document.querySelectorAll('[data-preset]').forEach((button) => button.classList.remove('active'));
    latestResult = null;
    latestPayload = null;
    byId('result-empty').hidden = false;
    byId('result-body').hidden = true;
    byId('gauge')?.style.setProperty('--score-angle', '0deg');
    byId('gauge')?.setAttribute('aria-label', 'Risk score not yet available');
}


/* ---------------------------------------------------------
   8. API REQUESTS
--------------------------------------------------------- */

async function fetchJson(path, options = {}) {
    const response = await fetch(apiUrl(path), {
        cache: 'no-store',
        ...options,
        headers: {
            ...(options.headers || {})
        }
    });

    let body;
    try {
        body = await response.json();
    } catch {
        body = null;
    }

    if (!response.ok) {
        const detail = body?.detail;
        throw new Error(formatApiError(detail) || `Request failed (${response.status}).`);
    }

    if (body === null) throw new Error('The API returned an unreadable response.');
    return body;
}

function formatApiError(detail) {
    if (Array.isArray(detail)) {
        return detail.map((item) => {
            const location = Array.isArray(item.loc)
                ? item.loc.filter((part) => part !== 'body').join(' → ')
                : 'Input';
            return `${location}: ${item.msg || 'Invalid value'}`;
        }).join('\n');
    }
    if (typeof detail === 'string') return detail;
    if (detail && typeof detail === 'object') return detail.message || detail.error || JSON.stringify(detail);
    return '';
}

async function loadModelOptions() {
    const data = await fetchJson('/api/options');
    if (!data || typeof data !== 'object') throw new Error('The model options response is invalid.');

    for (const section of FIELDS) {
        for (const field of section.items) {
            if (field.type === 'select' && !Array.isArray(data[field.option])) {
                throw new Error(`The API did not provide the '${field.option}' dropdown options.`);
            }
            if (field.type === 'select' && data[field.option].length === 0) {
                throw new Error(`The '${field.option}' dropdown has no available values in the trained encoder.`);
            }
        }
    }
    return data;
}

async function requestPrediction(payload) {
    return fetchJson('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
}


/* ---------------------------------------------------------
   9. LOADING STATE
--------------------------------------------------------- */

function setLoading(isLoading) {
    const button = byId('submit-btn');
    button.disabled = isLoading || !formReady;
    button.classList.toggle('loading', isLoading);
    byId('submit-btn').querySelector('.btn-label').textContent = isLoading ? 'Analyzing…' : 'Predict risk';
}


/* ---------------------------------------------------------
   10. DISPLAY MODEL RESULT
--------------------------------------------------------- */

function showResult(result, payload) {
    if (!result || typeof result !== 'object') throw new Error('The API returned an invalid prediction response.');

    const rawScore = Number(result.score);
    if (!Number.isFinite(rawScore)) throw new Error('The API response does not contain a valid numeric score.');

    const score = Math.min(100, Math.max(0, rawScore));
    const level = String(result.level || 'Unknown');
    latestResult = { ...result, score };
    latestPayload = payload;

    byId('result-empty').hidden = true;
    byId('result-body').hidden = false;
    byId('score-value').textContent = score.toFixed(1);
    byId('risk-level').textContent = level;
    byId('risk-level').className = `level ${level.toLowerCase().replace(/[^a-z0-9_-]/g, '-')}`;
    byId('gauge').style.setProperty('--score-angle', `${score * 3.6}deg`);
    byId('gauge').setAttribute('aria-label', `Risk score ${score.toFixed(1)} out of 100`);
    byId('result-body').dataset.score = score.toFixed(1);

    const caption = document.querySelector('.result-caption');
    caption.textContent = result.note || 'An exploratory model score on a 0–100 scale. It is not a probability or guaranteed forecast.';

    renderDrivers(Array.isArray(result.drivers) ? result.drivers : []);
    saveHistory(result, payload);
}

function renderDrivers(drivers) {
    const list = byId('drivers-list');
    list.replaceChildren();

    if (!drivers.length) {
        const item = document.createElement('li');
        item.textContent = 'No what-if comparison data was returned.';
        list.appendChild(item);
        return;
    }

    drivers.forEach((driver) => {
        const change = Number(driver.change ?? driver.delta ?? driver.impact);
        if (!Number.isFinite(change)) return;

        const item = document.createElement('li');
        const label = document.createElement('span');
        label.textContent = driver.label || driver.feature || driver.name || 'Indicator';

        const value = document.createElement('strong');
        value.textContent = `${change > 0 ? '+' : ''}${change.toFixed(2)}`;
        if (change > 0.0001) value.classList.add('up');
        if (change < -0.0001) value.classList.add('down');

        const unit = document.createElement('span');
        unit.className = 'driver-unit';
        unit.textContent = 'pts';
        value.appendChild(unit);
        item.append(label, value);
        list.appendChild(item);
    });

    if (!list.children.length) {
        const item = document.createElement('li');
        item.textContent = 'No numeric what-if comparison data was returned.';
        list.appendChild(item);
    }
}


/* ---------------------------------------------------------
   11. LOCAL PREDICTION HISTORY
--------------------------------------------------------- */

function getHistory() {
    try {
        const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
        return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
        console.warn('Could not read local prediction history:', error);
        return [];
    }
}

function saveHistory(result, payload) {
    try {
        const history = getHistory();
        history.unshift({
            id: `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
            date: new Date().toISOString(),
            score: result.score,
            level: result.level,
            payload
        });
        localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, MAX_HISTORY_ITEMS)));
        renderHistory();
    } catch (error) {
        console.warn('Could not save prediction history:', error);
    }
}

function renderHistory() {
    const host = byId('history-list');
    const history = getHistory();
    host.replaceChildren();

    if (!history.length) {
        const empty = document.createElement('div');
        empty.className = 'history-empty';
        const icon = document.createElement('span');
        icon.setAttribute('aria-hidden', 'true');
        icon.textContent = '◷';
        const copy = document.createElement('p');
        copy.textContent = 'No predictions yet. Your recent analyses will be saved here.';
        empty.append(icon, copy);
        host.appendChild(empty);
        return;
    }

    history.forEach((entry) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'history-item';
        button.title = 'Restore these inputs in the predictor';

        const main = document.createElement('span');
        main.className = 'history-item-main';

        const score = document.createElement('strong');
        const numericScore = Number(entry.score);
        score.textContent = Number.isFinite(numericScore) ? `${numericScore.toFixed(1)} / 100` : 'Score unavailable';

        const date = document.createElement('small');
        const parsedDate = new Date(entry.date);
        date.textContent = Number.isNaN(parsedDate.getTime()) ? 'Date unavailable' : parsedDate.toLocaleString();
        main.append(score, date);

        const level = document.createElement('span');
        level.className = 'history-item-level';
        level.textContent = entry.level || 'Unknown';

        button.append(main, level);
        button.addEventListener('click', () => {
            if (!formReady || !entry.payload) return;
            setForm(entry.payload);
            clearError();
            document.querySelectorAll('[data-preset]').forEach((chip) => chip.classList.remove('active'));
            byId('predictor').scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
        host.appendChild(button);
    });
}

function clearHistory() {
    try {
        localStorage.removeItem(HISTORY_KEY);
        renderHistory();
    } catch (error) {
        console.error('Could not clear history:', error);
        showError('The browser could not clear local history. Check your browser storage settings.');
    }
}


/* ---------------------------------------------------------
   12. CSV EXPORT
--------------------------------------------------------- */

function exportHistoryCsv() {
    const history = getHistory();
    if (!history.length) {
        showError('There is no prediction history to export yet.');
        return;
    }

    const fieldKeys = FIELDS.flatMap((section) => section.items.map((field) => field.key));
    const headers = ['date', 'score', 'risk_level', ...fieldKeys];
    const quoteCsv = (value) => `"${String(value ?? '').replace(/"/g, '""')}"`;

    const rows = history.map((entry) => [
        entry.date,
        entry.score,
        entry.level,
        ...fieldKeys.map((key) => entry.payload?.[key])
    ]);

    const csv = [headers, ...rows].map((row) => row.map(quoteCsv).join(',')).join('\r\n');
    const blob = new Blob(['\uFEFF', csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'workforce-layoff-risk-history.csv';
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}


/* ---------------------------------------------------------
   13. COPY SUMMARY
--------------------------------------------------------- */

async function copyResultSummary() {
    if (!latestResult) {
        showError('Run a prediction before copying a result summary.');
        return;
    }

    const lines = [
        'WORKFORCE INTELLIGENCE — LAYOFF RISK PREDICTOR',
        '',
        `Estimated model score: ${Number(latestResult.score).toFixed(1)} / 100`,
        `Application risk band: ${latestResult.level || 'Unknown'}`,
        '',
        'What-if comparisons (score-point change from baseline):',
        ...(Array.isArray(latestResult.drivers)
            ? latestResult.drivers.map((driver) => `${driver.label || driver.feature || 'Indicator'}: ${Number(driver.change) > 0 ? '+' : ''}${Number(driver.change).toFixed(2)} pts`)
            : []),
        '',
        'This is an exploratory regression score, not a probability or guaranteed forecast. What-if comparisons are not causal effects.'
    ];
    const text = lines.join('\n');

    try {
        if (navigator.clipboard?.writeText && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
            showTemporaryButtonLabel(byId('copy-btn'), 'Copied');
        } else {
            window.prompt('Copy your prediction summary:', text);
        }
    } catch (error) {
        console.warn('Clipboard operation failed:', error);
        window.prompt('Copy your prediction summary:', text);
    }
}

function showTemporaryButtonLabel(button, message) {
    const original = button.textContent;
    button.textContent = message;
    window.setTimeout(() => { button.textContent = original; }, 1500);
}


/* ---------------------------------------------------------
   14. EVENT HANDLERS
--------------------------------------------------------- */

function attachEventHandlers() {
    byId('form').addEventListener('submit', async (event) => {
        event.preventDefault();
        clearError();
        if (!formReady) return;

        let payload;
        try {
            payload = readForm();
        } catch (error) {
            showError(error.message || 'Please check the form values.');
            return;
        }

        setLoading(true);
        try {
            const result = await requestPrediction(payload);
            showResult(result, payload);
            if (window.matchMedia('(max-width: 760px)').matches) {
                byId('result-body').scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        } catch (error) {
            console.error('Prediction request failed:', error);
            showError(error.message || 'Prediction failed. Check the API server and try again.');
        } finally {
            setLoading(false);
        }
    });

    document.querySelectorAll('[data-preset]').forEach((button) => {
        button.addEventListener('click', () => {
            if (!formReady) return;
            const preset = PRESETS[button.dataset.preset];
            if (!preset) return;
            setForm(preset);
            clearError();
            document.querySelectorAll('[data-preset]').forEach((chip) => {
                chip.classList.toggle('active', chip === button);
            });
        });
    });

    byId('reset-btn').addEventListener('click', resetForm);
    byId('clear-history').addEventListener('click', clearHistory);
    byId('export-btn').addEventListener('click', exportHistoryCsv);
    byId('copy-btn').addEventListener('click', copyResultSummary);
    byId('print-btn').addEventListener('click', () => window.print());
}


/* ---------------------------------------------------------
   15. INITIALIZE APPLICATION
--------------------------------------------------------- */

async function initializeApp() {
    attachEventHandlers();
    renderHistory();

    // Check API health independently, so options can still load even
    // if the health endpoint is temporarily unavailable.
    const healthPromise = fetchJson('/api/health')
        .then((health) => {
            const isHealthy = health?.status === 'ok';
            setConnectionStatus(isHealthy ? 'online' : 'offline', isHealthy ? 'Model connected' : 'Model status unknown');
        })
        .catch((error) => {
            console.warn('API health check failed:', error);
            setConnectionStatus('offline', 'API unavailable');
        });

    try {
        modelOptions = await loadModelOptions();
        buildForm();
        formReady = true;
        setLoading(false);
        clearError();
    } catch (error) {
        console.error('Could not initialize the predictor:', error);
        const host = byId('form-sections');
        host.replaceChildren();

        const message = document.createElement('div');
        message.className = 'error';
        message.textContent = `${error.message} Make sure FastAPI is running and that the trained model files are beside main.py.`;
        host.appendChild(message);

        byId('submit-btn').disabled = true;
        setConnectionStatus('offline', 'Model unavailable');
    }

    await healthPromise;
}

document.addEventListener('DOMContentLoaded', initializeApp);
