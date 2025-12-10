const K_SIGMA = 1.0;
const MIN_TOL = 0.25;
const MODEL_COL_WIDTH = 300;
// Winner column removed

const state = {
    envs: ["TP1", "TP2"],
    backendOrder: ["TP1", "TP2"],
    columnWidths: { "TP1": 260, "TP2": 260 },
    filters: {
        search: "",
        quant: "",
        backends: new Set(["TP1", "TP2"]),
        sizeLo: null,
        sizeHi: null,
    },
    ui: {},
    sizeStats: { min: Infinity, max: -Infinity },
    draggingEnv: null,
    quantOptions: [],
};

document.addEventListener("DOMContentLoaded", async () => {
    cacheUI();
    setupModals();
    try {
        const res = await fetch("results.json");
        const data = await res.json();
        prepareData(data?.runs || []);
        initializeControls();
        renderTables();
    } catch (err) {
        console.error("Failed to load results.json", err);
        state.ui.stats.textContent = "Failed to load results.json";
    }
});

function cacheUI() {
    state.ui = {
        search: document.getElementById("filter-search"),
        quant: document.getElementById("filter-quant"),
        backendList: document.getElementById("backend-list"),
        backendAll: document.getElementById("backend-all"),
        backendNone: document.getElementById("backend-none"),
        sizeLo: document.getElementById("sizeLo"),
        sizeHi: document.getElementById("sizeHi"),
        sizeTrack: document.getElementById("sizeTrack"),
        sizeLoVal: document.getElementById("sizeLoVal"),
        sizeHiVal: document.getElementById("sizeHiVal"),
        stats: document.getElementById("stats-line"),
        resetBtn: document.getElementById("reset-layout"),
        tables: document.getElementById("tables"),
        // Modal hooks
        tp1ModalOpen: document.getElementById("tp1-modal-open"),
        tp2ModalOpen: document.getElementById("tp2-modal-open"),
        tp1Modal: document.getElementById("tp1-modal"),
        tp2Modal: document.getElementById("tp2-modal"),
        tp1ModalClose: document.getElementById("tp1-modal-close"),
        tp2ModalClose: document.getElementById("tp2-modal-close"),
    };
}

function setupModals() {
    const modalConfigs = [
        { open: state.ui.tp1ModalOpen, modal: state.ui.tp1Modal, close: state.ui.tp1ModalClose },
        { open: state.ui.tp2ModalOpen, modal: state.ui.tp2Modal, close: state.ui.tp2ModalClose },
    ];

    modalConfigs.forEach(({ open, modal, close }) => {
        if (!open || !modal) return;
        const openModal = () => modal.classList.remove("hidden");
        const closeModal = () => modal.classList.add("hidden");
        open.addEventListener("click", openModal);
        close?.addEventListener("click", closeModal);
        modal.addEventListener("click", (e) => {
            if (e.target === modal) closeModal();
        });
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && !modal.classList.contains("hidden")) {
                closeModal();
            }
        });
    });
}

function prepareData(runs) {
    const quantSet = new Set();
    // Tests map: TestName -> { name: ..., models: Map(ModelName -> Row) }
    const testsMap = new Map();

    for (const run of runs) {
        if (!run.test) continue;
        const testKey = run.test;

        if (run.quant) quantSet.add(run.quant.toUpperCase());

        if (!testsMap.has(testKey)) {
            testsMap.set(testKey, { name: testKey, models: new Map() });
        }
        const testEntry = testsMap.get(testKey);

        const modelName = run.model_clean || run.model;

        if (!testEntry.models.has(modelName)) {
            testEntry.models.set(modelName, {
                model: modelName,
                quant: (run.quant || "Unknown").toUpperCase(),
                sizeB: run.name_params_b ?? run.params_b ?? null,
                backends: {},
                search_blob: [modelName, run.quant, run.env, run.test]
                    .filter(Boolean)
                    .map((s) => s.toString().toLowerCase())
                    .join(" "),
            });
        }

        const row = testEntry.models.get(modelName);

        // Update stats
        if (row.sizeB != null) {
            state.sizeStats.min = Math.min(state.sizeStats.min, row.sizeB);
            state.sizeStats.max = Math.max(state.sizeStats.max, row.sizeB);
        }

        // Add backend data
        // run.env comes from python script as "TP1" or "TP2"
        const env = run.env;
        row.backends[env] = {
            mean: typeof run.tps_mean === "number" ? run.tps_mean : null,
            std: 0, // Not currently parsed
            error: Boolean(run.error),
            error_type: run.error_type || null,
        };
    }

    state.tests = [...testsMap.values()].sort((a, b) => a.name.localeCompare(b.name));
    state.quantOptions = [...quantSet].sort();
}

function initializeControls() {
    const { quant, backendList, search, resetBtn, sizeLo, sizeHi } = state.ui;

    quant.innerHTML = "";
    const anyOpt = document.createElement("option");
    anyOpt.value = "";
    anyOpt.textContent = "Any";
    quant.appendChild(anyOpt);
    state.quantOptions.forEach((q) => {
        const opt = document.createElement("option");
        opt.value = q;
        opt.textContent = q;
        quant.appendChild(opt);
    });

    renderBackendList();
    setupSizeSlider();

    search.addEventListener("input", (e) => {
        state.filters.search = (e.target.value || "").trim().toLowerCase();
        renderTables();
    });

    quant.addEventListener("change", (e) => {
        state.filters.quant = e.target.value;
        renderTables();
    });

    backendList.addEventListener("change", (e) => {
        const checkbox = e.target.closest("input[data-env]");
        if (!checkbox) return;
        const env = checkbox.dataset.env;
        if (checkbox.checked) {
            state.filters.backends.add(env);
        } else {
            state.filters.backends.delete(env);
        }
        renderTables();
    });

    state.ui.backendAll.addEventListener("click", () => {
        state.filters.backends = new Set(state.envs);
        renderBackendList();
        renderTables();
    });

    state.ui.backendNone.addEventListener("click", () => {
        state.filters.backends = new Set();
        renderBackendList();
        renderTables();
    });

    sizeLo.addEventListener("input", () => updateSizeUI(true));
    sizeHi.addEventListener("input", () => updateSizeUI(true));

    resetBtn.addEventListener("click", () => {
        state.filters.search = "";
        state.filters.quant = "";
        state.filters.backends = new Set(state.envs);
        search.value = "";
        quant.value = "";
        renderBackendList();
        setupSizeSlider();
        renderTables();
    });
}

function renderBackendList() {
    const container = state.ui.backendList;
    container.innerHTML = "";
    state.backendOrder.forEach((env) => {
        const label = document.createElement("label");
        label.className = "backend-item";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.dataset.env = env;
        checkbox.checked = state.filters.backends.has(env);
        label.appendChild(checkbox);

        const baseSpan = document.createElement("span");
        baseSpan.textContent = env;
        label.appendChild(baseSpan);

        container.appendChild(label);
    });
}

function setupSizeSlider() {
    const { sizeLo, sizeHi } = state.ui;
    const minRaw = state.sizeStats.min === Infinity ? 0 : Math.floor(state.sizeStats.min || 0);
    const maxRaw = state.sizeStats.max === -Infinity ? 0 : Math.ceil(state.sizeStats.max || 0);
    const minB = Math.max(0, minRaw);
    const maxB = Math.max(minB, maxRaw);

    [sizeLo, sizeHi].forEach((inp) => {
        inp.min = minB;
        inp.max = maxB;
        inp.step = 1;
    });

    sizeLo.value = minB;
    sizeHi.value = maxB;
    sizeLo.style.zIndex = 2;
    sizeHi.style.zIndex = 1;
    updateSizeUI(false);
}

function updateSizeUI(triggerRender) {
    const { sizeLo, sizeHi, sizeLoVal, sizeHiVal, sizeTrack } = state.ui;
    if (+sizeLo.value > +sizeHi.value) {
        if (document.activeElement === sizeLo) {
            sizeHi.value = sizeLo.value;
        } else {
            sizeLo.value = sizeHi.value;
        }
    }
    sizeLo.style.zIndex = +sizeLo.value >= +sizeHi.max - 1 ? 4 : 2;
    sizeHi.style.zIndex = +sizeHi.value <= +sizeLo.min + 1 ? 3 : 1;
    state.filters.sizeLo = +sizeLo.value;
    state.filters.sizeHi = +sizeHi.value;
    sizeLoVal.textContent = formatSizeLabel(state.filters.sizeLo);
    sizeHiVal.textContent = formatSizeLabel(state.filters.sizeHi);
    const range = (sizeHi.max - sizeLo.min) || 1;
    const minB = +sizeLo.min;
    const start = ((state.filters.sizeLo - minB) / range) * 100;
    const end = ((state.filters.sizeHi - minB) / range) * 100;
    sizeTrack.style.background = `linear-gradient(to right, #e3e7f1 ${start}%, var(--accent) ${start}%, var(--accent) ${end}%, #e3e7f1 ${end}%)`;
    if (triggerRender) renderTables();
}

function renderTables() {
    const backendList = state.backendOrder.filter((env) => state.filters.backends.has(env));
    const frag = document.createDocumentFragment();
    let totalRows = 0;

    for (const test of state.tests) {
        const models = filterModels(test.models);
        if (!models.length) continue;
        totalRows += models.length;

        const block = document.createElement("div");
        block.className = "test-block";
        const heading = document.createElement("h2");
        heading.textContent = test.name;
        block.appendChild(heading);

        const tableWrap = document.createElement("div");
        tableWrap.className = "table-wrap";
        const scroller = document.createElement("div");
        scroller.className = "table-scroll";

        const table = buildSingleTable(models, backendList);
        scroller.appendChild(table);
        tableWrap.appendChild(scroller);
        block.appendChild(tableWrap);
        setupResizeOverlay(scroller, backendList, table);
        frag.appendChild(block);
    }

    state.ui.tables.innerHTML = "";
    if (frag.childNodes.length) {
        state.ui.tables.appendChild(frag);
    } else {
        state.ui.tables.innerHTML = "<p>No models match the current filters.</p>";
    }
    state.ui.stats.textContent = `Showing ${totalRows.toLocaleString()} model rows across ${backendList.length} configurations`;
}

function buildSingleTable(models, backendList) {
    const table = document.createElement("table");
    const colgroup = document.createElement("colgroup");
    const colModel = document.createElement("col");
    colModel.style.width = `${MODEL_COL_WIDTH}px`;
    colgroup.appendChild(colModel);
    // Winner colGroup removed

    backendList.forEach((env) => {
        const col = document.createElement("col");
        col.style.width = `${state.columnWidths[env] || 200}px`;
        col.dataset.env = env;
        colgroup.appendChild(col);
    });
    table.appendChild(colgroup);

    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    headRow.appendChild(makeHeaderCell("Model", "model"));
    // Winner header removed

    backendList.forEach((env) => {
        const th = makeHeaderCell(env, ""); // REMOVED "backend-header" class
        attachHeaderInteractions(th, env);
        headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    models.forEach((model) => {
        const tr = document.createElement("tr");
        const tdModel = document.createElement("td");
        tdModel.className = "model";
        const head = document.createElement("div");
        head.className = "model-head";
        const nameSpan = document.createElement("span");
        nameSpan.className = "model-name";
        nameSpan.textContent = model.model;
        head.appendChild(nameSpan);
        tdModel.appendChild(head);

        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = `${model.quant} · ${formatSize(model.sizeB)}`;
        tdModel.appendChild(meta);
        tr.appendChild(tdModel);

        // Winner cell removed

        backendList.forEach((env) => {
            const td = document.createElement("td");
            td.className = "data-cell";
            td.dataset.env = env;
            const cell = model.backends[env];
            if (!cell) {
                td.innerHTML = `<span class="cell-empty">N/A</span>`;
            } else if (cell.error || cell.mean == null) {
                td.innerHTML = `<span class="cell-error">FAIL</span>`;
            } else {
                td.innerHTML = `<div class="measure">${cell.mean.toFixed(2)}</div>`;
            }
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
}

function makeHeaderCell(label, extra = "") {
    const th = document.createElement("th");
    th.textContent = label;
    if (extra) th.className = extra;
    return th;
}

function attachHeaderInteractions(th, env) {
    const width = state.columnWidths[env] || 200;
    th.style.width = `${width}px`;
    th.style.minWidth = `${width}px`;
    th.draggable = true;
    th.addEventListener("dragstart", (e) => {
        state.draggingEnv = env;
        th.classList.add("dragging");
        e.dataTransfer.effectAllowed = "move";
    });
    th.addEventListener("dragend", () => {
        state.draggingEnv = null;
        th.classList.remove("dragging");
        document.querySelectorAll("th.drop-target").forEach((el) => el.classList.remove("drop-target"));
    });
    th.addEventListener("dragover", (e) => {
        if (!state.draggingEnv || state.draggingEnv === env) return;
        e.preventDefault();
        th.classList.add("drop-target");
    });
    th.addEventListener("dragleave", () => th.classList.remove("drop-target"));
    th.addEventListener("drop", (e) => {
        if (!state.draggingEnv || state.draggingEnv === env) return;
        e.preventDefault();
        moveBackend(state.draggingEnv, env);
        th.classList.remove("drop-target");
    });

    const handle = document.createElement("span");
    handle.className = "resize-handle";
    handle.addEventListener("mousedown", (e) => startResize(e, env));
    th.appendChild(handle);
}

function moveBackend(from, to) {
    const order = state.backendOrder;
    const fromIdx = order.indexOf(from);
    const toIdx = order.indexOf(to);
    if (fromIdx === -1 || toIdx === -1) return;
    const [col] = order.splice(fromIdx, 1);
    order.splice(toIdx, 0, col);
    renderBackendList();
    renderTables();
}

function filterModels(modelsMap) {
    const models = [];
    for (const model of modelsMap.values()) {
        if (state.filters.search && !model.search_blob.includes(state.filters.search)) continue;
        if (state.filters.quant && model.quant !== state.filters.quant) continue;
        if (model.sizeB != null) {
            if (state.filters.sizeLo != null && model.sizeB < state.filters.sizeLo - 1e-6) continue;
            if (state.filters.sizeHi != null && model.sizeB > state.filters.sizeHi + 1e-6) continue;
        }
        models.push(model);
    }
    models.sort((a, b) => a.model.localeCompare(b.model));
    return models;
}

function formatSize(size) {
    if (size == null) return "—";
    return `${Number(size).toFixed(1)}B`;
}

function formatSizeLabel(size) {
    if (size >= 1000) return `${(size / 1000).toFixed(1)}kB`;
    return `${Math.round(size)}B`;
}

function startResize(event, env) {
    event.preventDefault();
    event.stopPropagation();
    const column = state.columnWidths[env] || 200;
    const startX = event.clientX;
    const shellRect = state.ui.tables.getBoundingClientRect();
    const guide = document.createElement("div");
    guide.className = "resize-line";
    guide.style.position = "fixed";
    guide.style.top = `${shellRect.top}px`;
    guide.style.bottom = `${window.innerHeight - shellRect.bottom}px`;
    guide.style.left = `${startX}px`;
    guide.style.width = "2px";
    guide.style.background = "var(--accent)";
    guide.style.zIndex = "10";
    document.body.appendChild(guide);
    let nextWidth = column;

    const onMove = (e) => {
        const delta = e.clientX - startX;
        nextWidth = Math.max(80, column + delta);
        guide.style.left = `${e.clientX}px`;
    };

    const onUp = () => {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        guide.remove();
        state.columnWidths[env] = nextWidth;
        renderTables();
    };

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
}

function setupResizeOverlay(tableWrap, backendList, table) {
    let overlay = tableWrap.querySelector(".resize-overlay");
    if (!overlay) {
        overlay = document.createElement("div");
        overlay.className = "resize-overlay";
        tableWrap.appendChild(overlay);
    } else {
        overlay.innerHTML = "";
    }

    overlay.style.width = `${tableWrap.clientWidth}px`;
    overlay.style.height = `${table.offsetHeight}px`;

    const bars = [];
    let offset = MODEL_COL_WIDTH; // Winner column width removed
    backendList.forEach((env) => {
        const width = state.columnWidths[env] || 200;
        const bar = document.createElement("div");
        bar.className = "resize-bar";
        bar.dataset.env = env;
        bar.addEventListener("mousedown", (e) => startResize(e, env));
        overlay.appendChild(bar);
        bars.push({ bar, offset, width, env });
        offset += width;
    });

    const positionBars = () => {
        bars.forEach(({ bar, offset, width }) => {
            const left = offset + width - 3 - tableWrap.scrollLeft;
            bar.style.left = `${left}px`;
        });
    };
    positionBars();

    if (tableWrap._overlayScroll) {
        tableWrap.removeEventListener("scroll", tableWrap._overlayScroll);
    }
    const onScroll = () => positionBars();
    tableWrap.addEventListener("scroll", onScroll);
    tableWrap._overlayScroll = onScroll;

    if (tableWrap._overlayResize) {
        tableWrap._overlayResize.disconnect();
    }
    const resizeObserver = new ResizeObserver(() => {
        overlay.style.width = `${tableWrap.clientWidth}px`;
        overlay.style.height = `${table.offsetHeight}px`;
        positionBars();
    });
    resizeObserver.observe(tableWrap);
    tableWrap._overlayResize = resizeObserver;
}
