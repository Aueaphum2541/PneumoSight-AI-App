const http = require("http");
const fs = require("fs");
const path = require("path");
const url = require("url");
const os = require("os");
const { spawn, spawnSync } = require("child_process");

const port = Number(process.env.PORT || 3000);
const root = __dirname;
const projectRoot = path.resolve(root, "..");
const downloads = path.join(os.homedir(), "Downloads");
const uploadRoot = path.join(root, "uploads");
const dataRoot = path.join(root, "data");
const casesPath = path.join(dataRoot, "cases.json");

fs.mkdirSync(uploadRoot, { recursive: true });
fs.mkdirSync(dataRoot, { recursive: true });

const assetMap = new Map([
  ["gradcam_examples.png.png", path.join(downloads, "gradcam_examples.png.png")],
  ["evaluation_curves.png.png", path.join(downloads, "evaluation_curves.png.png")],
  ["sample_xray_images.png.png", path.join(downloads, "sample_xray_images.png.png")],
  ["threshold_tradeoff.png.png", path.join(downloads, "threshold_tradeoff.png.png")],
  ["training_curves.png.png", path.join(downloads, "training_curves.png.png")],
  ["class_distribution.png.png", path.join(downloads, "class_distribution.png.png")],
  ["model_metadata.json", path.join(downloads, "model_metadata.json")]
]);

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg"
};

function send(res, status, body, type = "text/plain; charset=utf-8") {
  res.writeHead(status, {
    "Content-Type": type,
    "Cache-Control": "no-store",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type"
  });
  res.end(body);
}

function sendJson(res, status, payload) {
  send(res, status, JSON.stringify(payload, null, 2), "application/json; charset=utf-8");
}

function sendFile(res, filePath) {
  fs.readFile(filePath, (error, data) => {
    if (error) {
      send(res, 404, "File not found");
      return;
    }
    const type = mimeTypes[path.extname(filePath).toLowerCase()] || "application/octet-stream";
    res.writeHead(200, {
      "Content-Type": type,
      "Cache-Control": "no-store",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type"
    });
    res.end(data);
  });
}

function readCases() {
  try {
    return JSON.parse(fs.readFileSync(casesPath, "utf8"));
  } catch {
    return [];
  }
}

function writeCases(cases) {
  fs.writeFileSync(casesPath, JSON.stringify(cases.slice(0, 100), null, 2));
}

function safeFilename(name) {
  const cleaned = String(name || "xray.png").replace(/[^a-zA-Z0-9._-]/g, "_");
  return cleaned || "xray.png";
}

function collectBody(req, limitBytes = 25 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let total = 0;
    req.on("data", (chunk) => {
      total += chunk.length;
      if (total > limitBytes) {
        reject(new Error("Upload is larger than 25 MB."));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function parseMultipart(buffer, contentType) {
  const match = /boundary=(?:"([^"]+)"|([^;]+))/i.exec(contentType || "");
  if (!match) throw new Error("Missing multipart boundary.");
  const boundary = Buffer.from(`--${match[1] || match[2]}`);
  const parts = [];
  let cursor = 0;

  while (cursor < buffer.length) {
    const startBoundary = buffer.indexOf(boundary, cursor);
    if (startBoundary === -1) break;
    let partStart = startBoundary + boundary.length;
    if (buffer[partStart] === 45 && buffer[partStart + 1] === 45) break;
    if (buffer[partStart] === 13 && buffer[partStart + 1] === 10) partStart += 2;

    const nextBoundary = buffer.indexOf(boundary, partStart);
    if (nextBoundary === -1) break;
    let part = buffer.slice(partStart, nextBoundary);
    if (part.length >= 2 && part[part.length - 2] === 13 && part[part.length - 1] === 10) {
      part = part.slice(0, -2);
    }

    const headerEnd = part.indexOf(Buffer.from("\r\n\r\n"));
    if (headerEnd !== -1) {
      const rawHeaders = part.slice(0, headerEnd).toString("utf8");
      const data = part.slice(headerEnd + 4);
      const disposition = /content-disposition:[^\r\n]*/i.exec(rawHeaders)?.[0] || "";
      const name = /name="([^"]+)"/i.exec(disposition)?.[1] || "";
      const filename = /filename="([^"]*)"/i.exec(disposition)?.[1] || "";
      const contentTypeMatch = /content-type:\s*([^\r\n]+)/i.exec(rawHeaders);
      parts.push({ name, filename, contentType: contentTypeMatch?.[1] || "application/octet-stream", data });
    }
    cursor = nextBoundary + boundary.length;
  }
  return parts;
}

function findPython() {
  const candidates = [
    process.env.REAL_AI_PYTHON,
    path.join(projectRoot, ".real-ai", "Scripts", "python.exe"),
    path.join(process.env.LOCALAPPDATA || "", "Programs", "Python", "Python312", "python.exe"),
    path.join(process.env.LOCALAPPDATA || "", "Programs", "Python", "Python311", "python.exe"),
    path.join(process.env.LOCALAPPDATA || "", "Programs", "Python", "Python310", "python.exe"),
    "python",
    "python3"
  ].filter(Boolean);

  for (const candidate of candidates) {
    const result = spawnSync(candidate, ["-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"], {
      windowsHide: true,
      stdio: "ignore"
    });
    if (result.status === 0) return candidate;
  }
  return null;
}

function runInference(inputPath, caseDir) {
  const python = findPython();
  if (!python) {
    return Promise.reject(new Error("No Python 3.10+ runtime found. Run setup-real-ai.cmd first."));
  }

  const script = path.join(root, "infer.py");
  const env = {
    ...process.env,
    MODEL_PATH: process.env.MODEL_PATH || path.join(downloads, "best_model.keras"),
    MODEL_METADATA_PATH: process.env.MODEL_METADATA_PATH || path.join(downloads, "model_metadata.json")
  };

  return new Promise((resolve, reject) => {
    const child = spawn(python, [script, inputPath, caseDir], { cwd: root, env, windowsHide: true });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error("Inference timed out after 180 seconds."));
    }, 180000);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      try {
        const payload = JSON.parse(stdout || "{}");
        if (code === 0 && payload.ok !== false) {
          resolve(payload);
        } else {
          reject(new Error(payload.error || stderr || `Inference failed with exit code ${code}`));
        }
      } catch {
        reject(new Error(stderr || stdout || `Inference failed with exit code ${code}`));
      }
    });
  });
}

async function handleAnalyze(req, res) {
  try {
    const body = await collectBody(req);
    const parts = parseMultipart(body, req.headers["content-type"]);
    const filePart = parts.find((part) => part.name === "image" && part.filename && part.data.length);
    if (!filePart) {
      sendJson(res, 400, { ok: false, error: "No image file was uploaded." });
      return;
    }
    const metadata = {};
    for (const part of parts) {
      if (part.name && part.name !== "image" && !part.filename) {
        metadata[part.name] = part.data.toString("utf8");
      }
    }

    const ext = path.extname(filePart.filename).toLowerCase() || ".png";
    if (![".png", ".jpg", ".jpeg"].includes(ext)) {
      sendJson(res, 400, { ok: false, error: "Please upload a PNG or JPEG chest X-ray image." });
      return;
    }

    const id = `XRLOCAL-${new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14)}-${Math.random().toString(16).slice(2, 8)}`;
    const caseDir = path.join(uploadRoot, id);
    fs.mkdirSync(caseDir, { recursive: true });
    const originalName = safeFilename(filePart.filename);
    const originalPath = path.join(caseDir, `original${ext}`);
    fs.writeFileSync(originalPath, filePart.data);

    const inference = await runInference(originalPath, caseDir);
    const overlayPath = path.join(caseDir, "gradcam.png");
    const result = {
      id,
      createdAt: new Date().toISOString(),
      filename: originalName,
      metadata,
      originalUrl: `/uploads/${id}/original${ext}`,
      overlayUrl: fs.existsSync(overlayPath) ? `/uploads/${id}/gradcam.png` : null,
      ...inference
    };

    const cases = readCases();
    cases.unshift(result);
    writeCases(cases);
    sendJson(res, 200, { ok: true, result });
  } catch (error) {
    sendJson(res, 500, {
      ok: false,
      error: error.message,
      setup: "Run setup-real-ai.cmd once, then start-real-ai-localhost.cmd."
    });
  }
}

const server = http.createServer((req, res) => {
  const parsed = url.parse(req.url || "/");
  const pathname = decodeURIComponent(parsed.pathname || "/");

  if (req.method === "OPTIONS") {
    send(res, 204, "");
    return;
  }

  if (req.method === "POST" && pathname === "/api/analyze") {
    handleAnalyze(req, res);
    return;
  }

  if (req.method === "GET" && pathname === "/api/cases") {
    sendJson(res, 200, { ok: true, cases: readCases() });
    return;
  }

  if (pathname === "/" || pathname === "/index.html") {
    sendFile(res, path.join(root, "index.html"));
    return;
  }

  if (pathname === "/health") {
    sendJson(res, 200, { status: "ok", mode: "real-upload-localhost" });
    return;
  }

  if (pathname.startsWith("/assets/")) {
    const name = pathname.replace("/assets/", "");
    const filePath = assetMap.get(name);
    if (!filePath) {
      send(res, 404, "Unknown asset");
      return;
    }
    sendFile(res, filePath);
    return;
  }

  if (pathname.startsWith("/uploads/")) {
    const relative = pathname.replace("/uploads/", "");
    const filePath = path.resolve(uploadRoot, relative);
    if (!filePath.startsWith(uploadRoot)) {
      send(res, 400, "Invalid upload path");
      return;
    }
    sendFile(res, filePath);
    return;
  }

  sendFile(res, path.join(root, "index.html"));
});

server.listen(port, "127.0.0.1", () => {
  console.log(`PneumoSight AI is running at http://localhost:${port}`);
  console.log("Upload analysis endpoint: POST /api/analyze");
  console.log("Close this window to stop the localhost server.");
});
