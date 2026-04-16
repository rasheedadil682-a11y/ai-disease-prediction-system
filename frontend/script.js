(function () {
  "use strict";

  const API_BASE = "https://ai-disease-prediction-system-production.up.railway.app";
  const API_URL = `${API_BASE}/predict`;
  const TIMEOUT_MS = 12000;
  const THEME_KEY = "nexus-predict-theme";

  const loading = document.getElementById("loading-overlay");
  const themeToggle = document.getElementById("theme-toggle");
  const heartForm = document.getElementById("heart-form");
  const diabetesForm = document.getElementById("diabetes-form");
  const heartResult = document.getElementById("heart-result");
  const diabetesResult = document.getElementById("diabetes-result");
  const heartMetrics = document.getElementById("heart-metrics");
  const diabetesMetrics = document.getElementById("diabetes-metrics");
  const heartInsights = document.getElementById("heart-insights");
  const diabetesInsights = document.getElementById("diabetes-insights");
  const heartRiskFactors = document.getElementById("heart-risk-factors");
  const diabetesRiskFactors = document.getElementById("diabetes-risk-factors");
  const heartDiet = document.getElementById("heart-diet");
  const diabetesDiet = document.getElementById("diabetes-diet");
  const heartExercise = document.getElementById("heart-exercise");
  const diabetesExercise = document.getElementById("diabetes-exercise");
  const tabs = document.querySelectorAll(".tab");
  const heartPanel = document.getElementById("heart-panel");
  const diabetesPanel = document.getElementById("diabetes-panel");
  const charts = {};

  function setLoading(isLoading) {
    loading.classList.toggle("hidden", !isLoading);
  }

  function setTheme(theme) {
    document.body.classList.toggle("light", theme === "light");
    localStorage.setItem(THEME_KEY, theme);
  }

  function initTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === "light") setTheme("light");
    else setTheme("dark");
  }

  function readForm(form) {
    const data = {};
    form.querySelectorAll("input, select").forEach((field) => {
      if (!field.name) return;
      if (field.tagName === "SELECT") {
        data[field.name] = field.value;
      } else {
        data[field.name] = field.value === "" ? "" : Number(field.value);
      }
    });
    return data;
  }

  function normalizeScore(value, min, max) {
    const clamped = Math.max(min, Math.min(max, value));
    return ((clamped - min) / (max - min)) * 100;
  }

  function weightedRiskScore(model, payload) {
    if (model === "heart") {
      const score =
        normalizeScore(payload.age || 0, 18, 85) * 0.2 +
        normalizeScore(payload.resting_bp || 0, 90, 200) * 0.18 +
        normalizeScore(payload.cholesterol || 0, 120, 420) * 0.2 +
        normalizeScore(payload.oldpeak || 0, 0, 6) * 0.16 +
        normalizeScore((payload.maxhr || 0) * -1, -210, -70) * 0.1 +
        ((payload.exercise_angina === "Y" ? 100 : 20) * 0.09) +
        ((payload.st_slope === "Down" ? 95 : payload.st_slope === "Flat" ? 75 : 35) * 0.07);
      return Math.max(0, Math.min(100, Math.round(score)));
    }
    const score =
      normalizeScore(payload.glucose || 0, 70, 220) * 0.24 +
      normalizeScore(payload.bmi || 0, 16, 50) * 0.2 +
      normalizeScore(payload.bp || 0, 55, 130) * 0.14 +
      normalizeScore(payload.insulin || 0, 0, 400) * 0.14 +
      normalizeScore(payload.age || 0, 18, 85) * 0.1 +
      normalizeScore(payload.dpf || 0, 0.05, 2.2) * 0.1 +
      normalizeScore(payload.pregnancies || 0, 0, 15) * 0.08;
    return Math.max(0, Math.min(100, Math.round(score)));
  }

  function featureContributions(model, payload) {
    if (model === "heart") {
      const entries = [
        { key: "Age", value: normalizeScore(payload.age || 0, 18, 85) * 0.2 },
        { key: "Resting BP", value: normalizeScore(payload.resting_bp || 0, 90, 200) * 0.18 },
        { key: "Cholesterol", value: normalizeScore(payload.cholesterol || 0, 120, 420) * 0.2 },
        { key: "Oldpeak", value: normalizeScore(payload.oldpeak || 0, 0, 6) * 0.16 },
        { key: "Max HR", value: normalizeScore((payload.maxhr || 0) * -1, -210, -70) * 0.1 },
        { key: "Exercise Angina", value: (payload.exercise_angina === "Y" ? 100 : 20) * 0.09 },
        { key: "ST Slope", value: (payload.st_slope === "Down" ? 95 : payload.st_slope === "Flat" ? 75 : 35) * 0.07 },
      ];
      return entries.map((entry) => ({ ...entry, value: Number(entry.value.toFixed(2)) }));
    }

    const entries = [
      { key: "Glucose", value: normalizeScore(payload.glucose || 0, 70, 220) * 0.24 },
      { key: "BMI", value: normalizeScore(payload.bmi || 0, 16, 50) * 0.2 },
      { key: "Blood Pressure", value: normalizeScore(payload.bp || 0, 55, 130) * 0.14 },
      { key: "Insulin", value: normalizeScore(payload.insulin || 0, 0, 400) * 0.14 },
      { key: "Age", value: normalizeScore(payload.age || 0, 18, 85) * 0.1 },
      { key: "DPF", value: normalizeScore(payload.dpf || 0, 0.05, 2.2) * 0.1 },
      { key: "Pregnancies", value: normalizeScore(payload.pregnancies || 0, 0, 15) * 0.08 },
    ];
    return entries.map((entry) => ({ ...entry, value: Number(entry.value.toFixed(2)) }));
  }

  function resolveRiskPercent(model, payload, apiResult) {
    const probability = typeof apiResult.probability === "number" ? apiResult.probability : null;
    const isHigh = String(apiResult.result || "").toLowerCase().includes("high");

    if (probability !== null) {
      const percent = Math.round((isHigh ? probability : 1 - probability) * 100);
      return Math.max(1, Math.min(99, percent));
    }

    const weighted = weightedRiskScore(model, payload);
    if (isHigh) return Math.max(70, weighted);
    return Math.min(30, Math.max(1, 100 - weighted));
  }

  function riskLevel(percent) {
    if (percent >= 70) return { key: "high", label: "High Risk", icon: "⚠️" };
    if (percent >= 40) return { key: "medium", label: "Medium Risk", icon: "🟠" };
    return { key: "low", label: "Low Risk", icon: "✅" };
  }

  function confidenceLevel(percent, hasProbability) {
    if (hasProbability) return percent >= 75 || percent <= 25 ? "High" : percent >= 60 || percent <= 40 ? "Medium" : "Low";
    return percent >= 80 || percent <= 20 ? "Medium" : "Low";
  }

  function buildAiExplanation(model, payload, levelLabel, rankedFactors) {
    const top = rankedFactors.slice(0, 3).map((f) => f.key).join(", ");
    const protective = rankedFactors[rankedFactors.length - 1]?.key || "baseline stability markers";
    if (model === "heart") {
      return {
        summary: `Summary: Overall profile indicates ${levelLabel.toLowerCase()} for heart disease screening.`,
        drivers: `Risk drivers: ${top} are currently the strongest contributors based on weighted feature analysis.`,
        protective: `Protective factors: ${protective} appears comparatively less aggressive, which moderates total severity.`,
        conclusion: `Conclusion: Your values suggest cardiovascular strain is present, so preventive action should focus first on blood pressure, lipids, and stress-adjusted activity pacing.`,
      };
    }
    return {
      summary: `Summary: Overall profile indicates ${levelLabel.toLowerCase()} for diabetes screening.`,
      drivers: `Risk drivers: ${top} dominate the current metabolic risk pattern from your submitted values.`,
      protective: `Protective factors: ${protective} contributes less relative pressure, helping offset some risk load.`,
      conclusion: `Conclusion: Prioritize glycemic control, weight strategy, and consistent activity to shift this trajectory in a healthier direction.`,
    };
  }

  function buildRecommendations(model, payload, levelKey) {
    const dietDo = [];
    const dietAvoid = [];
    const exercise = [];

    if (model === "heart") {
      if ((payload.resting_bp || 0) >= 140) {
        dietDo.push("Add potassium-rich foods (spinach, banana, beans) and keep sodium under 1500 mg/day.");
        dietAvoid.push("Avoid packaged soups, chips, and high-sodium fast food.");
      }
      if ((payload.cholesterol || 0) >= 240) {
        dietDo.push("Use oats, flaxseed, lentils, and fatty fish 3x/week for lipid support.");
        dietAvoid.push("Avoid fried foods, processed meats, and butter-heavy meals.");
      }
      if (dietDo.length === 0) dietDo.push("Maintain Mediterranean-style meals with olive oil, vegetables, whole grains, and lean protein.");

      if (levelKey === "high") exercise.push("Begin with supervised 20-30 min brisk walking, 5 days/week.");
      else if (levelKey === "medium") exercise.push("Target 150 min/week moderate cardio + 2 strength sessions.");
      else exercise.push("Continue maintenance routine: cardio + mobility + light resistance.");
      if (payload.exercise_angina === "Y") exercise.push("Avoid sudden high-intensity bursts; progress gradually.");
    } else {
      if ((payload.glucose || 0) >= 140) {
        dietDo.push("Choose low-glycemic carbs (millets, oats, legumes) and add fiber in each meal.");
        dietAvoid.push("Avoid sugary beverages, sweets, and frequent refined flour snacks.");
      }
      if ((payload.bmi || 0) >= 30) {
        dietDo.push("Use portion-controlled, high-protein meals with vegetables in a mild calorie deficit.");
        dietAvoid.push("Avoid late-night heavy meals and repetitive high-calorie takeout.");
      }
      if ((payload.insulin || 0) >= 180) {
        dietDo.push("Keep meal timing consistent and include post-meal light walking.");
        dietAvoid.push("Avoid large carb-only meals that spike insulin demand.");
      }
      if (dietDo.length === 0) dietDo.push("Maintain balanced carb intake, hydration, and regular mealtime consistency.");

      if (levelKey === "high") exercise.push("Start with 30 min daily walking + post-meal 10 min light activity.");
      else if (levelKey === "medium") exercise.push("Build toward 180 min/week aerobic activity with 2 resistance sessions.");
      else exercise.push("Maintain 150 min/week mixed cardio and strength training.");
      if ((payload.bp || 0) >= 90) exercise.push("Include low-impact cardio and controlled breathing cooldowns.");
    }

    return { dietDo, dietAvoid, exercise };
  }

  function renderMetrics(container, percent, healthScore, levelLabel, confidence) {
    container.classList.remove("hidden");
    container.innerHTML = `
      <div class="metrics-grid">
        <div class="metric-item">
          <div class="metric-label">Risk Percentage</div>
          <div class="metric-value">${percent}%</div>
        </div>
        <div class="metric-item">
          <div class="metric-label">Health Score</div>
          <div class="metric-value">${healthScore}/100</div>
        </div>
        <div class="metric-item">
          <div class="metric-label">Risk Tier</div>
          <div class="metric-value">${levelLabel}</div>
        </div>
        <div class="metric-item">
          <div class="metric-label">Confidence</div>
          <div class="metric-value">${confidence}</div>
        </div>
      </div>
    `;
  }

  function renderInsights(container, explanation) {
    container.classList.remove("hidden");
    container.innerHTML = `
      <strong>AI Explanation</strong>
      <p>${explanation.summary}</p>
      <p>${explanation.drivers}</p>
      <p>${explanation.protective}</p>
      <p>${explanation.conclusion}</p>
    `;
  }

  function renderPlan(container, title, list) {
    container.classList.remove("hidden");
    container.innerHTML = `<strong>${title}</strong><ul>${list.map((item) => `<li>${item}</li>`).join("")}</ul>`;
  }

  function renderRiskFactors(container, factors) {
    container.classList.remove("hidden");
    const top = factors.slice(0, 5);
    container.innerHTML = `
      <strong>Key Risk Factors</strong>
      <ul>${top.map((item) => `<li>${item.key} (+${Math.round(item.value)}%)</li>`).join("")}</ul>
    `;
  }

  function renderChart(canvasId, highPercent) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === "undefined") return;
    const ctx = canvas.getContext("2d");
    if (charts[canvasId]) charts[canvasId].destroy();
    const lowPercent = 100 - highPercent;
    charts[canvasId] = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: ["High Risk", "Low Risk"],
        datasets: [{
          data: [highPercent, lowPercent],
          backgroundColor: ["#ef5b73", "#28c179"],
          borderColor: ["#ef5b73", "#28c179"],
          borderWidth: 1,
          hoverOffset: 8,
        }],
      },
      options: {
        animation: { duration: 700 },
        plugins: {
          legend: { labels: { color: getComputedStyle(document.body).getPropertyValue("--text-muted") } },
        },
      },
    });
  }

  function renderImportanceChart(canvasId, factors) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === "undefined") return;
    const ctx = canvas.getContext("2d");
    if (charts[canvasId]) charts[canvasId].destroy();
    charts[canvasId] = new Chart(ctx, {
      type: "bar",
      data: {
        labels: factors.map((f) => f.key),
        datasets: [{
          label: "Importance",
          data: factors.map((f) => Number(f.value.toFixed(2))),
          backgroundColor: "rgba(124, 137, 255, 0.72)",
          borderColor: "rgba(159, 103, 255, 0.95)",
          borderWidth: 1.2,
          borderRadius: 8,
        }],
      },
      options: {
        animation: { duration: 700 },
        scales: {
          x: { ticks: { color: getComputedStyle(document.body).getPropertyValue("--text-muted") } },
          y: { ticks: { color: getComputedStyle(document.body).getPropertyValue("--text-muted") }, beginAtZero: true },
        },
        plugins: {
          legend: { labels: { color: getComputedStyle(document.body).getPropertyValue("--text-muted") } },
        },
      },
    });
  }

  function showPremiumResult(container, level, riskPercent) {
    const title =
      level.key === "high" ? "High Risk Detected" :
      level.key === "medium" ? "Medium Risk Detected" :
      "Low Risk";
    container.classList.remove("hidden", "high", "low", "error");
    container.classList.add(level.key === "high" ? "high" : level.key === "medium" ? "error" : "low");
    container.innerHTML = `
      <span class="risk-badge badge-${level.key}">${level.icon} ${level.label}</span>
      <h3>${title}</h3>
      <p>${level.key === "high" ? "The model detected elevated clinical risk signals." : level.key === "medium" ? "The model indicates moderate concern and needs proactive management." : "The model indicates a favorable risk profile for this screening."}</p>
      <div class="risk-percent">${riskPercent}%</div>
      <div class="progress"><span style="width:${riskPercent}%"></span></div>
    `;
  }

  function showError(container, message) {
    container.classList.remove("hidden", "high", "low");
    container.classList.add("error");
    container.innerHTML = `<h3>Prediction unavailable</h3><p>${message}</p>`;
  }

  async function postPredict(payload) {
    console.log("[frontend] request payload:", payload);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      const text = await response.text();
      let json;
      try {
        json = text ? JSON.parse(text) : {};
      } catch (_err) {
        throw new Error("Invalid server response format.");
      }
      console.log("[frontend] response:", json);
      if (!response.ok || json.ok === false) {
        throw new Error(json.error || "Prediction request failed");
      }
      return json;
    } catch (error) {
      if (error.name === "AbortError") {
        throw new Error("Request timed out after 12 seconds. Please retry.");
      }
      if (String(error.message || "").includes("Failed to fetch")) {
        throw new Error(`Unable to reach API at ${API_BASE}. Ensure backend is running on port 5001.`);
      }
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  function validatePayload(payload) {
    for (const [key, val] of Object.entries(payload)) {
      if (key === "model") continue;
      if (val === "" || val === null || val === undefined || Number.isNaN(val)) {
        throw new Error(`Please provide a valid value for "${key}".`);
      }
    }
  }

  async function handleSubmit(form, modelName, ui) {
    const payload = readForm(form);
    payload.model = modelName;
    const button = form.querySelector("button[type='submit']");

    try {
      validatePayload(payload);
    } catch (error) {
      showError(ui.resultEl, error.message || "Invalid form input.");
      return;
    }

    setLoading(true);
    button.disabled = true;
    ui.resultEl.classList.add("hidden");
    ui.metricsEl.classList.add("hidden");
    ui.insightsEl.classList.add("hidden");
    ui.riskFactorsEl.classList.add("hidden");
    ui.dietEl.classList.add("hidden");
    ui.exerciseEl.classList.add("hidden");

    try {
      const result = await postPredict(payload);
      const riskPercent = resolveRiskPercent(modelName, payload, result);
      const healthScore = Math.max(1, 100 - riskPercent);
      const level = riskLevel(riskPercent);
      const factors = featureContributions(modelName, payload).sort((a, b) => b.value - a.value);
      const explanation = buildAiExplanation(modelName, payload, level.label, factors);
      const recs = buildRecommendations(modelName, payload, level.key);
      const confidence = confidenceLevel(riskPercent, typeof result.probability === "number");

      showPremiumResult(ui.resultEl, level, riskPercent);
      renderMetrics(ui.metricsEl, riskPercent, healthScore, level.label, confidence);
      renderChart(ui.chartId, riskPercent);
      renderRiskFactors(ui.riskFactorsEl, factors);
      renderImportanceChart(ui.importanceChartId, factors);
      renderInsights(ui.insightsEl, explanation);
      renderPlan(ui.dietEl, "🥗 Diet Plan - DO", recs.dietDo);
      renderPlan(ui.exerciseEl, "🏃 Exercise Plan", recs.exercise);
      if (recs.dietAvoid.length) {
        ui.dietEl.innerHTML += `<strong>🚫 Diet - Avoid</strong><ul>${recs.dietAvoid.map((item) => `<li>${item}</li>`).join("")}</ul>`;
      }
      ui.resultEl.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      showError(ui.resultEl, error.message || "Unexpected error");
    } finally {
      setLoading(false);
      button.disabled = false;
    }
  }

  heartForm.addEventListener("submit", (event) => {
    event.preventDefault();
    handleSubmit(heartForm, "heart", {
      resultEl: heartResult,
      metricsEl: heartMetrics,
      insightsEl: heartInsights,
      riskFactorsEl: heartRiskFactors,
      dietEl: heartDiet,
      exerciseEl: heartExercise,
      chartId: "heart-chart",
      importanceChartId: "heart-importance-chart",
    });
  });

  diabetesForm.addEventListener("submit", (event) => {
    event.preventDefault();
    handleSubmit(diabetesForm, "diabetes", {
      resultEl: diabetesResult,
      metricsEl: diabetesMetrics,
      insightsEl: diabetesInsights,
      riskFactorsEl: diabetesRiskFactors,
      dietEl: diabetesDiet,
      exerciseEl: diabetesExercise,
      chartId: "diabetes-chart",
      importanceChartId: "diabetes-importance-chart",
    });
  });

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const key = tab.dataset.tab;
      tabs.forEach((button) => button.classList.remove("active"));
      tab.classList.add("active");

      const showHeart = key === "heart";
      heartPanel.classList.toggle("active", showHeart);
      diabetesPanel.classList.toggle("active", !showHeart);
    });
  });

  themeToggle.addEventListener("click", () => {
    const nextTheme = document.body.classList.contains("light") ? "dark" : "light";
    setTheme(nextTheme);
  });

  initTheme();
})();
