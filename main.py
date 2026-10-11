import os
import json
import time
import uuid
import subprocess
from typing import Dict
from fastapi import FastAPI, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import feedparser
import yt_dlp
from faster_whisper import WhisperModel
from google import genai

app = FastAPI(title="HTN Video Clipper")

# Folders setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Gemini Setup
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
ai_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

# In-Memory Database
db = {
    "tracked_channels": [],
    "tasks": {}
}

# --- Core Logic ---

def fetch_latest_youtube_video(channel_id: str):
    rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    feed = feedparser.parse(rss_url)
    if not feed.entries:
        return None
    latest = feed.entries[0]
    return {
        "title": latest.title,
        "url": latest.link,
        "video_id": latest.yt_videoid
    }

def download_video(url: str, output_path: str):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
        'overwrites': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def transcribe_audio(video_path: str):
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(video_path)
    return [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments]

def find_viral_clip_gemini(transcript: list):
    prompt = f"""
Analyze this timestamped transcript and extract the single most engaging 30 to 50-second viral hook moment for TikTok/Shorts:
{json.dumps(transcript[:120], indent=2)}

Return ONLY JSON:
{{"start_time": <number_in_seconds>, "end_time": <number_in_seconds>, "summary": "<reason>"}}
"""
    res = ai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )
    return json.loads(res.text)

def render_anti_copyright_short(input_video: str, start_time: float, duration: float, output_path: str):
    speed = 1.03
    pts = round(1.0 / speed, 4)
    filter_complex = (
        f"[0:v]setpts={pts}*PTS,eq=contrast=1.05:saturation=1.12:brightness=0.01,hflip,split=2[raw_bg][raw_fg];"
        f"[raw_bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[bg];"
        f"[raw_fg]scale=1080:-1[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2[outv];"
        f"[0:a]atempo={speed},asetrate=44100*1.02,aresample=44100[outa]"
    )
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_time), "-i", input_video, "-t", str(duration),
        "-filter_complex", filter_complex, "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-b:v", "4000k", "-c:a", "aac", "-preset", "fast", output_path
    ]
    subprocess.run(cmd, check=True)

# --- Background Task Worker ---

def process_video_pipeline(task_id: str, channel_id: str, direct_url: str):
    try:
        db["tasks"][task_id]["status"] = "processing"

        if direct_url:
            target_url = direct_url
            video_title = "Direct Video Input"
        else:
            db["tasks"][task_id]["message"] = "Channel မှ ဗီဒီယိုအသစ် စစ်ဆေးနေပါသည်..."
            latest = fetch_latest_youtube_video(channel_id)
            if not latest:
                raise Exception("Channel ID မမှန်ပါ သို့မဟုတ် ဗီဒီယိုမရှိပါ။")
            target_url = latest["url"]
            video_title = latest["title"]

        db["tasks"][task_id]["title"] = video_title
        raw_path = os.path.join(OUTPUT_DIR, f"{task_id}_raw.mp4")
        final_short = os.path.join(OUTPUT_DIR, f"{task_id}_short.mp4")

        db["tasks"][task_id]["message"] = "ဗီဒီယို ဒေါင်းလုဒ်ဆွဲနေပါသည်..."
        download_video(target_url, raw_path)

        db["tasks"][task_id]["message"] = "AI ဖြင့် စကားပြောသံများကို စာသားပြောင်းနေပါသည်..."
        transcript = transcribe_audio(raw_path)

        db["tasks"][task_id]["message"] = "Viral Hook အပိုင်းကို ရှာဖွေနေပါသည်..."
        highlight = find_viral_clip_gemini(transcript)
        start_sec = highlight["start_time"]
        duration = highlight["end_time"] - start_sec

        db["tasks"][task_id]["message"] = "Anti-Copyright 9:16 Shorts အဖြစ် ဖြတ်ညှပ်နေပါသည်..."
        render_anti_copyright_short(raw_path, start_sec, duration, final_short)

        if os.path.exists(raw_path):
            os.remove(raw_path)

        db["tasks"][task_id]["status"] = "completed"
        db["tasks"][task_id]["message"] = "အောင်မြင်စွာ ပြီးဆုံးပါပြီ!"
        db["tasks"][task_id]["result_url"] = f"/outputs/{task_id}_short.mp4"

    except Exception as e:
        db["tasks"][task_id]["status"] = "failed"
        db["tasks"][task_id]["message"] = f"Error: {str(e)}"

# --- API Routes ---

@app.post("/api/track")
def track_channel(channel_id: str = Form(...)):
    if channel_id not in db["tracked_channels"]:
        db["tracked_channels"].append(channel_id)
    return {"channels": db["tracked_channels"]}

@app.get("/api/channels")
def get_channels():
    return {"channels": db["tracked_channels"]}

@app.post("/api/create-clip")
def create_clip(
    background_tasks: BackgroundTasks,
    channel_id: str = Form(default=""),
    direct_url: str = Form(default="")
):
    task_id = str(uuid.uuid4())[:8]
    db["tasks"][task_id] = {
        "id": task_id,
        "status": "queued",
        "message": "စတင်ရန် တန်းစီနေပါသည်...",
        "title": "",
        "result_url": None,
        "created_at": time.time()
    }
    background_tasks.add_task(process_video_pipeline, task_id, channel_id, direct_url)
    return {"task_id": task_id}

@app.get("/api/task/{task_id}")
def get_task_status(task_id: str):
    task = db["tasks"].get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    return task

@app.get("/", response_class=HTMLResponse)
def index_page():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HTN Clip Studio</title>
    <link rel="manifest" href="/static/manifest.json">
    <meta name="theme-color" content="#4F46E5">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap" rel="stylesheet">
    <style>body { font-family: 'Plus Jakarta Sans', sans-serif; }</style>
</head>
<body class="bg-[#0B0F19] text-slate-100 min-h-screen">
    <!-- Navbar -->
    <nav class="border-b border-slate-800 bg-[#111827]/60 backdrop-blur-md sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center font-bold text-white shadow-lg shadow-indigo-500/30">H</div>
                <span class="font-bold text-lg tracking-tight">HTN Clip Studio</span>
            </div>
            <span class="text-xs bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 px-3 py-1 rounded-full font-medium">HTN Automation Engine</span>
        </div>
    </nav>

    <!-- Main Container -->
    <div class="max-w-6xl mx-auto px-6 py-10 grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        <!-- Left: Form Controls -->
        <div class="space-y-6">
            <div class="bg-[#111827] border border-slate-800/80 rounded-2xl p-6 shadow-xl">
                <h2 class="text-base font-bold text-white mb-4 flex items-center space-x-2">
                    <span>⚡ Create Viral Short</span>
                </h2>
                
                <form id="clipForm" class="space-y-4">
                    <div>
                        <label class="block text-xs font-semibold uppercase text-slate-400 mb-1.5">Tracked Channel ID</label>
                        <input type="text" id="channel_id" placeholder="UCX6OQ3DkcsbYNE6H8uQQuVA" 
                               class="w-full bg-[#1A2234] border border-slate-700/80 rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500">
                    </div>

                    <div class="relative flex items-center justify-center py-1">
                        <div class="border-t border-slate-800 w-full"></div>
                        <span class="bg-[#111827] px-3 text-[11px] font-bold text-slate-500 uppercase">OR</span>
                    </div>

                    <div>
                        <label class="block text-xs font-semibold uppercase text-slate-400 mb-1.5">Direct Video Link</label>
                        <input type="text" id="direct_url" placeholder="https://www.youtube.com/watch?v=..." 
                               class="w-full bg-[#1A2234] border border-slate-700/80 rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500">
                    </div>

                    <button type="submit" id="btnSubmit" 
                            class="w-full py-3 bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-500 hover:to-indigo-600 text-white font-semibold rounded-xl text-sm transition shadow-lg shadow-indigo-600/30 flex items-center justify-center space-x-2">
                        <span>Auto-Clip & Crop 9:16</span>
                    </button>
                </form>
            </div>

            <!-- Tracked Channels List -->
            <div class="bg-[#111827] border border-slate-800/80 rounded-2xl p-6 shadow-xl">
                <h2 class="text-base font-bold text-white mb-3">📡 Active Track Channels</h2>
                <div class="flex space-x-2 mb-3">
                    <input type="text" id="new_channel" placeholder="Channel ID..." 
                           class="flex-1 bg-[#1A2234] border border-slate-700/80 rounded-lg px-3 py-1.5 text-xs text-white">
                    <button onclick="addTrackChannel()" class="bg-slate-800 hover:bg-slate-700 text-xs px-3 py-1.5 rounded-lg font-medium">Add</button>
                </div>
                <div id="channelList" class="space-y-1.5 text-xs text-slate-400 max-h-36 overflow-y-auto">
                    <p class="text-slate-500">No channels tracked yet</p>
                </div>
            </div>
        </div>

        <!-- Right: Status & Video Player -->
        <div class="lg:col-span-2">
            <div class="bg-[#111827] border border-slate-800/80 rounded-2xl p-6 shadow-xl min-h-[480px] flex flex-col justify-center items-center text-center relative overflow-hidden">
                
                <!-- Idle State -->
                <div id="stateIdle" class="space-y-3">
                    <div class="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mx-auto text-2xl">🎬</div>
                    <h3 class="text-lg font-semibold text-white">HTN Studio Ready</h3>
                    <p class="text-sm text-slate-400 max-w-sm">Channel ID သို့မဟုတ် Video Link ထည့်သွင်းပြီး Auto Clip စတင်ထုတ်ယူပါ</p>
                </div>

                <!-- Processing State -->
                <div id="stateProcessing" class="hidden space-y-4">
                    <div class="w-12 h-12 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
                    <div>
                        <h4 id="procTitle" class="text-base font-bold text-white">ဗီဒီယိုကို စစ်ဆေးနေပါသည်...</h4>
                        <p id="procMsg" class="text-xs text-indigo-400 mt-1">Starting HTN engine</p>
                    </div>
                </div>

                <!-- Completed State -->
                <div id="stateCompleted" class="hidden flex flex-col items-center w-full">
                    <h4 class="text-base font-bold text-emerald-400 mb-3">✨ Generated 9:16 Anti-Copyright Short</h4>
                    <div class="w-[260px] h-[460px] rounded-2xl overflow-hidden shadow-2xl border border-slate-700 mb-4 bg-black">
                        <video id="resultVideo" controls class="w-full h-full object-cover"></video>
                    </div>
                    <a id="downloadLink" download="viral_short.mp4" 
                       class="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 rounded-xl text-xs font-bold text-white shadow-lg transition">
                        Download Short (.mp4)
                    </a>
                </div>

            </div>
        </div>

    </div>

    <script>
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/static/sw.js');
        }

        let currentTaskId = null;
        let pollTimer = null;

        document.getElementById('clipForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const channelId = document.getElementById('channel_id').value;
            const directUrl = document.getElementById('direct_url').value;

            if(!channelId && !directUrl) {
                alert("Channel ID သို့မဟုတ် Direct URL ထည့်သွင်းပေးပါ");
                return;
            }

            const formData = new FormData();
            formData.append('channel_id', channelId);
            formData.append('direct_url', directUrl);

            setUIState('processing');

            const res = await fetch('/api/create-clip', { method: 'POST', body: formData });
            const data = await res.json();
            currentTaskId = data.task_id;

            pollTimer = setInterval(pollTask, 2500);
        });

        async function pollTask() {
            if(!currentTaskId) return;
            const res = await fetch(`/api/task/${currentTaskId}`);
            const data = await res.json();

            document.getElementById('procMsg').innerText = data.message;
            if(data.title) document.getElementById('procTitle').innerText = data.title;

            if(data.status === 'completed') {
                clearInterval(pollTimer);
                document.getElementById('resultVideo').src = data.result_url;
                document.getElementById('downloadLink').href = data.result_url;
                setUIState('completed');
            } else if(data.status === 'failed') {
                clearInterval(pollTimer);
                alert("လုပ်ဆောင်မှု မအောင်မြင်ပါ: " + data.message);
                setUIState('idle');
            }
        }

        function setUIState(state) {
            document.getElementById('stateIdle').classList.toggle('hidden', state !== 'idle');
            document.getElementById('stateProcessing').classList.toggle('hidden', state !== 'processing');
            document.getElementById('stateCompleted').classList.toggle('hidden', state !== 'completed');
            document.getElementById('btnSubmit').disabled = (state === 'processing');
        }

        async function addTrackChannel() {
            const val = document.getElementById('new_channel').value.trim();
            if(!val) return;
            const fd = new FormData();
            fd.append('channel_id', val);
            const res = await fetch('/api/track', { method: 'POST', body: fd });
            const data = await res.json();
            renderChannels(data.channels);
            document.getElementById('new_channel').value = '';
        }

        function renderChannels(channels) {
            const box = document.getElementById('channelList');
            if(channels.length === 0) return;
            box.innerHTML = channels.map(c => `<div class="p-2 bg-slate-800/60 rounded-lg flex justify-between items-center"><span>${c}</span><span class="text-emerald-400">● live</span></div>`).join('');
        }
    </script>
</body>
</html>
    """
  
