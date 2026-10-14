/* ==========================================================================
   50 Startups Feature Selection Dashboard — Client-Side Application Logic
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  const data = window.STARTUP_DATA || {};

  // ------------------------------------------------------------------------
  // 1. Tab Navigation
  // ------------------------------------------------------------------------
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabContents.forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add("active");
      }
    });
  });

  // ------------------------------------------------------------------------
  // 2. Algorithm Filter Chips & Dynamic Table / Chart Swapping
  // ------------------------------------------------------------------------
  const chips = document.querySelectorAll(".chip");
  const mainChart = document.getElementById("main-feature-chart");
  const chartTitle = document.getElementById("current-chart-title");
  const chartBadge = document.getElementById("current-chart-badge");
  const downloadBtn = document.getElementById("chart-download-btn");
  const tableTitle = document.getElementById("current-table-title");
  const tableBody = document.getElementById("feature-table-body");

  const chartConfigs = {
    combined: {
      title: "RMSE & R-squared by Number of Features — All 5 Algorithms Combined",
      badge: "1 of Together",
      img: "assets/fs_combined/combined_algorithms_k_features.png",
      filterAlgo: null,
    },
    correlation: {
      title: "RMSE & R-squared by Number of Features — Correlation Filter (|r|)",
      badge: "Filter Method",
      img: "assets/fs_individual/correlation_k_features.png",
      filterAlgo: "Correlation",
    },
    mutual_info: {
      title: "RMSE & R-squared by Number of Features — Mutual Information (MI)",
      badge: "Filter Method",
      img: "assets/fs_individual/mutual_information_k_features.png",
      filterAlgo: "Mutual Information",
    },
    rfe: {
      title: "RMSE & R-squared by Number of Features — Recursive Feature Elimination (RFE)",
      badge: "Wrapper Method",
      img: "assets/fs_individual/rfe_k_features.png",
      filterAlgo: "RFE",
    },
    rfecv: {
      title: "RMSE & R-squared by Number of Features — RFECV (Cross-Validated)",
      badge: "Wrapper (CV)",
      img: "assets/fs_individual/rfecv_k_features.png",
      filterAlgo: "RFECV",
    },
    lasso: {
      title: "RMSE & R-squared by Number of Features — LASSO Regularization Path",
      badge: "Embedded (L1)",
      img: "assets/fs_individual/lasso_k_features.png",
      filterAlgo: "LASSO",
    },
  };

  function renderFeatureTable(filterAlgo = null) {
    if (!data.algorithmResults) return;
    tableBody.innerHTML = "";

    let rows = data.algorithmResults;
    if (filterAlgo) {
      rows = rows.filter((r) => r.Algorithm === filterAlgo);
    }

    rows.forEach((row) => {
      const tr = document.createElement("tr");
      const isOptimal = row["Number of Features"] === 2 && row.RMSE < 8200;

      tr.innerHTML = `
        <td><strong>${row.Algorithm}</strong></td>
        <td><span class="badge-tag">${row["Number of Features"]}</span></td>
        <td><code>${row["Selected Features"]}</code></td>
        <td style="font-weight: 700; color: ${isOptimal ? '#34d399' : 'inherit'};">$${row.RMSE.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
        <td style="font-weight: 700; color: ${isOptimal ? '#34d399' : 'inherit'};">${row["R-squared"].toFixed(6)}</td>
        <td>${isOptimal ? '<span class="badge-optimal">Global Optimum</span>' : '<span class="badge-suboptimal">Sub-optimal</span>'}</td>
      `;
      tableBody.appendChild(tr);
    });
  }

  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      chips.forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");

      const key = chip.getAttribute("data-algo");
      const cfg = chartConfigs[key] || chartConfigs.combined;

      chartTitle.textContent = cfg.title;
      chartBadge.textContent = cfg.badge;
      mainChart.src = cfg.img;
      downloadBtn.href = cfg.img;

      if (cfg.filterAlgo) {
        tableTitle.textContent = `${cfg.filterAlgo} Feature Progression Table (k = 1 to 5)`;
      } else {
        tableTitle.textContent = "All 5 Algorithms Performance Comparison (k = 1 to 5)";
      }

      renderFeatureTable(cfg.filterAlgo);
    });
  });

  // Initial table render
  renderFeatureTable(null);

  // ------------------------------------------------------------------------
  // 3. Live Interactive Profit Predictor
  // ------------------------------------------------------------------------
  const sliderRd = document.getElementById("slider-rd");
  const inputRd = document.getElementById("input-rd");
  const valRd = document.getElementById("val-rd");

  const sliderAdmin = document.getElementById("slider-admin");
  const inputAdmin = document.getElementById("input-admin");
  const valAdmin = document.getElementById("val-admin");

  const sliderMarketing = document.getElementById("slider-marketing");
  const inputMarketing = document.getElementById("input-marketing");
  const valMarketing = document.getElementById("val-marketing");

  const inputState = document.getElementById("input-state");
  const resetBtn = document.getElementById("reset-inputs-btn");

  const predBestVal = document.getElementById("pred-best-val");
  const predBaseVal = document.getElementById("pred-base-val");
  const predDiffVal = document.getElementById("pred-diff-val");

  // Exact Scaler Parameters from Training
  const SCALER_MEAN = {
    rd: 71663.358,
    admin: 123391.4115,
    marketing: 202141.028,
  };
  const SCALER_SCALE = {
    rd: 46189.79069592,
    admin: 25887.54913413,
    marketing: 123460.24539264,
  };

  // Exact Model Coefficients from Fitted Pipeline
  const BEST_MODEL = {
    intercept: 109446.44725,
    coef_rd: 36426.11421332,
    coef_mkt: 4021.91840262,
  };

  const BASE_MODEL = {
    intercept: 109441.48912163253,
    coef_rd: 35726.28774249,
    coef_admin: 851.30163448,
    coef_mkt: 4519.88277698,
    coef_fl: -959.28416006,
    coef_ny: 699.36905252,
  };

  function updatePredictions() {
    const rd = parseFloat(inputRd.value) || 0;
    const admin = parseFloat(inputAdmin.value) || 0;
    const marketing = parseFloat(inputMarketing.value) || 0;
    const state = inputState.value;

    valRd.textContent = `$${rd.toLocaleString()}`;
    valAdmin.textContent = `$${admin.toLocaleString()}`;
    valMarketing.textContent = `$${marketing.toLocaleString()}`;

    // Standardize continuous inputs
    const z_rd = (rd - SCALER_MEAN.rd) / SCALER_SCALE.rd;
    const z_admin = (admin - SCALER_MEAN.admin) / SCALER_SCALE.admin;
    const z_mkt = (marketing - SCALER_MEAN.marketing) / SCALER_SCALE.marketing;

    // One-hot dummy encodings for State (drop California)
    const isFlorida = state === "Florida" ? 1 : 0;
    const isNewYork = state === "New York" ? 1 : 0;

    // 1. Calculate Best Model Prediction (R&D + Marketing Spend)
    const predBest = BEST_MODEL.intercept + BEST_MODEL.coef_rd * z_rd + BEST_MODEL.coef_mkt * z_mkt;

    // 2. Calculate Baseline Model Prediction (All 5 features)
    const predBase =
      BASE_MODEL.intercept +
      BASE_MODEL.coef_rd * z_rd +
      BASE_MODEL.coef_admin * z_admin +
      BASE_MODEL.coef_mkt * z_mkt +
      BASE_MODEL.coef_fl * isFlorida +
      BASE_MODEL.coef_ny * isNewYork;

    const diff = predBest - predBase;

    predBestVal.textContent = `$${predBest.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    predBaseVal.textContent = `$${predBase.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

    const sign = diff >= 0 ? "+" : "-";
    predDiffVal.textContent = `${sign}$${Math.abs(diff).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }

  function syncInputs(slider, input) {
    slider.addEventListener("input", () => {
      input.value = slider.value;
      updatePredictions();
    });
    input.addEventListener("input", () => {
      slider.value = input.value;
      updatePredictions();
    });
  }

  syncInputs(sliderRd, inputRd);
  syncInputs(sliderAdmin, inputAdmin);
  syncInputs(sliderMarketing, inputMarketing);
  inputState.addEventListener("change", updatePredictions);

  resetBtn.addEventListener("click", () => {
    sliderRd.value = 100000;
    inputRd.value = 100000;
    sliderAdmin.value = 120000;
    inputAdmin.value = 120000;
    sliderMarketing.value = 250000;
    inputMarketing.value = 250000;
    inputState.value = "New York";
    updatePredictions();
  });

  updatePredictions();

  // ------------------------------------------------------------------------
  // 4. Raw Dataset Table
  // ------------------------------------------------------------------------
  const rawTableBody = document.getElementById("raw-table-body");
  const datasetSearch = document.getElementById("dataset-search");

  function renderRawTable(filterText = "") {
    if (!data.rawDataset) return;
    rawTableBody.innerHTML = "";

    const lower = filterText.toLowerCase();
    const filtered = data.rawDataset.filter((row) => {
      return (
        row.State.toLowerCase().includes(lower) ||
        row["R&D Spend"].toString().includes(lower) ||
        row.Profit.toString().includes(lower)
      );
    });

    filtered.forEach((row, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="color: var(--text-dim);">${i + 1}</td>
        <td>$${row["R&D Spend"].toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
        <td>$${row.Administration.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
        <td>$${row["Marketing Spend"].toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
        <td><span class="badge-tag">${row.State}</span></td>
        <td style="font-weight: 700; color: #34d399;">$${row.Profit.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
      `;
      rawTableBody.appendChild(tr);
    });
  }

  if (datasetSearch) {
    datasetSearch.addEventListener("input", (e) => {
      renderRawTable(e.target.value);
    });
  }

  renderRawTable();

  // ------------------------------------------------------------------------
  // 5. Diagnostics Table
  // ------------------------------------------------------------------------
  const diagTableBody = document.getElementById("diag-table-body");
  if (diagTableBody && data.featureImportance) {
    diagTableBody.innerHTML = "";
    data.featureImportance.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${row.Feature}</strong></td>
        <td>${row.Pearson_r.toFixed(4)}</td>
        <td style="font-weight: 700;">${row.Abs_Pearson_r.toFixed(4)}</td>
        <td>${row.Mutual_Information.toFixed(4)}</td>
        <td><span class="badge-tag">Rank ${row.RFE_Ranking}</span></td>
        <td>${row.Lasso_Coefficient.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
        <td>${row.Selected_by_Lasso ? '<span class="badge-optimal">True (Retained)</span>' : '<span class="badge-suboptimal">False (Zeroed)</span>'}</td>
      `;
      diagTableBody.appendChild(tr);
    });
  }
});
