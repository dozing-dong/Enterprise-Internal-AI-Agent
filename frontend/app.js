const sessionInput = document.getElementById("session-id");
const questionInput = document.getElementById("question-input");
const askButton = document.getElementById("ask-button");
const historyButton = document.getElementById("history-button");
const clearButton = document.getElementById("clear-button");
const answerBox = document.getElementById("answer-box");
const sourcesBox = document.getElementById("sources-box");
const historyBox = document.getElementById("history-box");
const healthStatus = document.getElementById("health-status");
const healthDetail = document.getElementById("health-detail");


function setAnswer(text) {
    // 把回答文本显示到页面上。
    answerBox.textContent = text;
}


function renderSources(sources) {
    // 如果没有来源，就显示默认提示。
    if (!sources || sources.length === 0) {
        sourcesBox.innerHTML = '<div class="empty-state">这里会显示检索到的参考片段。</div>';
        return;
    }

    let html = "";

    // 逐个渲染来源片段。
    for (const source of sources) {
        html += `
            <div class="source-item">
                <div class="source-item__meta">排名：${source.rank} | metadata：${JSON.stringify(source.metadata)}</div>
                <div class="source-item__content">${escapeHtml(source.content)}</div>
            </div>
        `;
    }

    sourcesBox.innerHTML = html;
}


function renderHistory(messages) {
    // 如果没有历史，就显示默认提示。
    if (!messages || messages.length === 0) {
        historyBox.innerHTML = '<div class="empty-state">这里会显示会话历史。</div>';
        return;
    }

    let html = "";

    // 逐个渲染历史消息。
    for (const message of messages) {
        html += `
            <div class="history-item">
                <div class="history-item__meta">${escapeHtml(JSON.stringify(message))}</div>
                <div class="history-item__content">${escapeHtml(JSON.stringify(message, null, 2))}</div>
            </div>
        `;
    }

    historyBox.innerHTML = html;
}


function escapeHtml(text) {
    // 这里做最基础的 HTML 转义，避免把接口返回内容当作标签插入页面。
    return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}


async function loadHealth() {
    // 加载服务健康状态。
    try {
        const response = await fetch("/health");
        const data = await response.json();

        healthStatus.textContent = data.status;
        healthDetail.textContent =
            `原始文档数：${data.raw_document_count}，向量片段数：${data.vector_document_count}`;
    } catch (error) {
        healthStatus.textContent = "error";
        healthDetail.textContent = error.message;
    }
}


async function askQuestion() {
    // 发送问题到后端。
    const sessionId = sessionInput.value.trim() || "default";
    const question = questionInput.value.trim();

    if (!question) {
        setAnswer("问题不能为空。");
        return;
    }

    setAnswer("正在请求模型，请稍候...");
    renderSources([]);

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: question,
                session_id: sessionId
            })
        });

        const data = await response.json();

        if (!response.ok) {
            setAnswer(`请求失败：${data.detail || "未知错误"}`);
            return;
        }

        setAnswer(data.answer);
        renderSources(data.sources);
    } catch (error) {
        setAnswer(`请求失败：${error.message}`);
    }
}


async function loadHistory() {
    // 加载会话历史。
    const sessionId = sessionInput.value.trim() || "default";

    try {
        const response = await fetch(`/history/${encodeURIComponent(sessionId)}`);
        const data = await response.json();

        if (!response.ok) {
            historyBox.innerHTML = `<div class="empty-state">读取历史失败：${escapeHtml(data.detail || "未知错误")}</div>`;
            return;
        }

        renderHistory(data.messages);
    } catch (error) {
        historyBox.innerHTML = `<div class="empty-state">读取历史失败：${escapeHtml(error.message)}</div>`;
    }
}


async function clearHistory() {
    // 清空会话历史。
    const sessionId = sessionInput.value.trim() || "default";

    try {
        const response = await fetch(`/history/${encodeURIComponent(sessionId)}`, {
            method: "DELETE"
        });
        const data = await response.json();

        if (!response.ok) {
            historyBox.innerHTML = `<div class="empty-state">清空失败：${escapeHtml(data.detail || "未知错误")}</div>`;
            return;
        }

        historyBox.innerHTML = `<div class="empty-state">${escapeHtml(data.message)}</div>`;
    } catch (error) {
        historyBox.innerHTML = `<div class="empty-state">清空失败：${escapeHtml(error.message)}</div>`;
    }
}


// 给按钮绑定事件。
askButton.addEventListener("click", askQuestion);
historyButton.addEventListener("click", loadHistory);
clearButton.addEventListener("click", clearHistory);

// 页面第一次加载时先检查服务状态。
loadHealth();
