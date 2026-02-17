(function () {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const uploadSection = document.getElementById("uploadSection");
  const progressSection = document.getElementById("progressSection");
  const progressTitle = document.getElementById("progressTitle");
  const progressText = document.getElementById("progressText");
  const progressFill = document.getElementById("progressFill");
  const resultSection = document.getElementById("resultSection");
  const downloadBtn = document.getElementById("downloadBtn");
  const uploadAnotherBtn = document.getElementById("uploadAnotherBtn");
  const errorSection = document.getElementById("errorSection");
  const errorText = document.getElementById("errorText");
  const retryBtn = document.getElementById("retryBtn");
  const liveFeed = document.getElementById("liveFeed");
  const liveStartBtn = document.getElementById("liveStartBtn");
  const liveStopBtn = document.getElementById("liveStopBtn");
  const liveMonitorFeed = document.getElementById("liveMonitorFeed");
  const liveStatusText = document.getElementById("liveStatusText");
  const liveMetricsEl = document.getElementById("liveMetrics");
  const liveSourceInput = document.getElementById("liveSource");
  var liveStatsInterval = null;
  var liveFrameLoopActive = false;
  var liveFrameTimeout = null;
  var liveFrameUrl = null;
  const tabs = document.querySelectorAll(".tab");
  const panelUpload = document.getElementById("panelUpload");
  const panelLive = document.getElementById("panelLive");

  const POLL_INTERVAL_MS = 2000;
  const MAX_FILE_SIZE_MB = 500;
  const LIVE_FRAME_POLL_MS = 12;

  function show(section) {
    [uploadSection, progressSection, resultSection, errorSection].forEach(function (el) {
      el.classList.add("hidden");
    });
    if (section) section.classList.remove("hidden");
  }

  function setProgress(percent, title, text) {
    progressFill.style.width = (percent || 0) + "%";
    if (title) progressTitle.textContent = title;
    if (text) progressText.textContent = text;
  }

  function showError(message) {
    errorText.textContent = message || "An error occurred.";
    show(errorSection);
  }

  function scheduleLiveFramePoll(delayMs) {
    if (liveFrameTimeout) clearTimeout(liveFrameTimeout);
    liveFrameTimeout = setTimeout(fetchLatestLiveFrame, delayMs);
  }

  function fetchLatestLiveFrame() {
    if (!liveFrameLoopActive) return;
    fetch("/api/live/frame?t=" + Date.now(), { cache: "no-store" })
      .then(function (r) {
        if (r.status === 204) return null;
        if (!r.ok) throw new Error("Live frame fetch failed");
        return r.blob();
      })
      .then(function (blob) {
        if (!liveFrameLoopActive) return;
        if (!blob) {
          scheduleLiveFramePoll(LIVE_FRAME_POLL_MS);
          return;
        }
        var nextUrl = URL.createObjectURL(blob);
        if (liveFrameUrl) URL.revokeObjectURL(liveFrameUrl);
        liveFrameUrl = nextUrl;
        liveMonitorFeed.src = nextUrl;
        scheduleLiveFramePoll(LIVE_FRAME_POLL_MS);
      })
      .catch(function () {
        if (!liveFrameLoopActive) return;
        scheduleLiveFramePoll(120);
      });
  }

  function startLiveFrameLoop() {
    liveFrameLoopActive = true;
    scheduleLiveFramePoll(0);
  }

  function stopLiveFrameLoop() {
    liveFrameLoopActive = false;
    if (liveFrameTimeout) {
      clearTimeout(liveFrameTimeout);
      liveFrameTimeout = null;
    }
    if (liveFrameUrl) {
      URL.revokeObjectURL(liveFrameUrl);
      liveFrameUrl = null;
    }
    liveMonitorFeed.src = "";
  }

  function pollJobStatus(jobId) {
    function poll() {
      fetch("/api/jobs/" + jobId)
        .then(function (r) {
          if (!r.ok) throw new Error("Job status failed");
          return r.json();
        })
        .then(function (data) {
          if (data.status === "processing") {
            setProgress(50, "Processing your video…", "Watch the live feed above. Download when done.");
            setTimeout(poll, POLL_INTERVAL_MS);
            return;
          }
          if (data.status === "done" && data.download_url) {
            setProgress(100, "Done", "Your annotated video is ready.");
            downloadBtn.href = data.download_url;
            downloadBtn.download = "annotated.mp4";
            if (liveFeed) liveFeed.src = "";
            show(progressSection);
            setTimeout(function () {
              show(resultSection);
            }, 400);
            return;
          }
          if (data.status === "failed") {
            if (liveFeed) liveFeed.src = "";
            showError(data.error || "Processing failed.");
            return;
          }
          setProgress(33, "Processing your video…", "Detection is running…");
          setTimeout(poll, POLL_INTERVAL_MS);
        })
        .catch(function (err) {
          if (liveFeed) liveFeed.src = "";
          showError(err.message || "Network error. Please try again.");
        });
    }
    poll();
  }

  function uploadFile(file) {
    if (!file || !file.type.startsWith("video/")) {
      showError("Please select a video file (MP4, AVI, MOV, MKV).");
      return;
    }
    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      showError("File is too large. Max " + MAX_FILE_SIZE_MB + " MB.");
      return;
    }

    show(progressSection);
    setProgress(10, "Uploading…", "Sending your video.");

    const formData = new FormData();
    formData.append("file", file);

    fetch("/api/upload", {
      method: "POST",
      body: formData,
    })
      .then(function (r) {
        if (!r.ok) throw new Error("Upload failed");
        return r.json();
      })
      .then(function (data) {
        setProgress(20, "Processing your video…", "Watch the live feed above; download when done.");
        if (liveFeed) liveFeed.src = "/api/jobs/" + data.job_id + "/stream";
        pollJobStatus(data.job_id);
      })
      .catch(function (err) {
        showError(err.message || "Upload failed. Please try again.");
      });
  }

  dropzone.addEventListener("click", function () {
    fileInput.click();
  });

  dropzone.addEventListener("dragover", function (e) {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", function () {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", function (e) {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    var file = e.dataTransfer && e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });

  fileInput.addEventListener("change", function () {
    var file = fileInput.files[0];
    if (file) uploadFile(file);
    fileInput.value = "";
  });

  uploadAnotherBtn.addEventListener("click", function () {
    if (liveFeed) liveFeed.src = "";
    show(uploadSection);
  });

  retryBtn.addEventListener("click", function () {
    if (liveFeed) liveFeed.src = "";
    show(uploadSection);
  });

  // Tabs
  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      var t = tab.getAttribute("data-tab");
      tabs.forEach(function (x) {
        x.classList.remove("active");
        x.setAttribute("aria-selected", "false");
      });
      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");
      if (t === "upload") {
        panelUpload.classList.remove("hidden");
        panelLive.classList.add("hidden");
      } else {
        panelUpload.classList.add("hidden");
        panelLive.classList.remove("hidden");
      }
    });
  });

  // Live monitor: Start
  liveStartBtn.addEventListener("click", function () {
    var raw = liveSourceInput.value.trim();
    var body = {};
    if (raw === "" || /^\d+$/.test(raw)) {
      body.camera_index = parseInt(raw, 10) || 0;
    } else {
      body.source = raw;
    }
    liveStartBtn.disabled = true;
    fetch("/api/live/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then(function (r) {
        if (!r.ok) return r.json().then(function (e) { throw new Error(e.detail || "Start failed"); });
        return r.json();
      })
      .then(function () {
        startLiveFrameLoop();
        liveStartBtn.classList.add("hidden");
        liveStopBtn.classList.remove("hidden");
        if (liveStatusText) liveStatusText.textContent = "Stream running";
        if (liveMetricsEl) liveMetricsEl.textContent = "FPS: —  ·  Inference: — ms";
        liveStatsInterval = setInterval(function () {
          fetch("/api/live/stats").then(function (r) { return r.json(); }).then(function (d) {
            if (liveMetricsEl && d.running) liveMetricsEl.textContent = "FPS: " + (d.fps != null ? d.fps : "—") + "  ·  Inference: " + (d.inference_ms != null ? d.inference_ms : "—") + " ms";
          });
        }, 1000);
      })
      .catch(function (err) {
        alert(err.message || "Failed to start stream.");
      })
      .finally(function () {
        liveStartBtn.disabled = false;
      });
  });

  // Live monitor: Stop
  liveStopBtn.addEventListener("click", function () {
    if (liveStatsInterval) { clearInterval(liveStatsInterval); liveStatsInterval = null; }
    fetch("/api/live/stop", { method: "POST" })
      .then(function () {
        stopLiveFrameLoop();
        liveStopBtn.classList.add("hidden");
        liveStartBtn.classList.remove("hidden");
        if (liveStatusText) liveStatusText.textContent = "Stream stopped";
        if (liveMetricsEl) liveMetricsEl.textContent = "—";
      });
  });
})();
