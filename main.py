import os
import json
import uuid
import subprocess
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import feedparser
import yt_dlp
from faster_whisper import WhisperModel
from google import genai
import google.oauth2.credentials
import googleapiclient.discovery
from googleapiclient.http import MediaFileUpload

app = FastAPI(title="HTN Clip Studio")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
ai_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

db = {
    "tracked_channels": [],
    "tasks": {}
}

# --- YouTube Auto Post Uploader ---
def upload_scheduled_short(video_path: str, title: str, schedule_time_iso: str):
    secret_path = os.path.join(BASE_DIR, "client_secret.json")
    if not os.path.exists(secret_path):
        print("[Auto-Post] client_secret.json ရှာမတွေ့ပါ။ upload ကို ကျော်ပါမည်။")
        return None

    try:
        with open(secret_path, 'r') as f:
            secret_data = json.load(f)

        if "refresh_token" not in secret_data:
            if "installed" in secret_data:
                secret_data = secret_data["installed"]
            elif "web" in secret_data:
                secret_data = secret_data["web"]
                
        if "refresh_token" not in secret_data:
            print("[Auto-Post] client_secret.json ထဲတွင် refresh_token မပါရှိပါ။ upload ကို ကျော်ပါမည်။")
            return None

        creds = google.oauth2.credentials.Credentials(
            token=None,
            refresh_token=secret_data["refresh_token"],
            token_uri="https://oauth2.googleapis.com/token",
            client_id=secret_data.get("client_id"),
            client_secret=secret_data.get("client_secret")
        )

        youtube = googleapiclient.discovery.build("youtube", "v3", credentials=creds)

        body = {
            "snippet": {
                "title": f"{title} #Shorts #HTN"[:100],
                "description": f"{title} - HTN Clip Studio မှ အလိုအလျောက်တင်ပေးထားသည်! #Shorts #HTN",
                "tags": ["Shorts", "Viral", "HTN"],
                "categoryId": "22"
            },
            "status": {
                "privacyStatus": "private",
                "publishAt": schedule_time_iso,
                "selfDeclaredMadeForKids": False
            }
        }
        
        media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
        req = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        
        response = None
        while response is None:
            status, response = req.next_chunk()
            if status:
                print(f"[Auto-Post Upload] {int(status.progress() * 100)}% ပြီးပါပြီ")
        
        video_id = response['id']
        return f"https://youtube.com/shorts/{video_id}"
        
    except Exception as e:
        print(f"[Auto-Post Error] တင်၍မရပါ: {e}")
        return None

# --- Core Video Processing ---
def fetch_latest_youtube_video(channel_id: str):
    rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    feed = feedparser.parse(rss_url)
    if not feed.entries:
        return None
    latest = feed.entries[0]
    return {"title": latest.title, "url": latest.link}

def download_video(url: str, output_path: str):
    # Cloud Data Center IP Block & Player Response Error ကို ကျော်လွှားသည့် extractor ဖွဲ့စည်းမှု
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': output_path,
        'overwrites': True,
        'quiet': True,
        'no_warnings': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['android_embedded', 'web_creator', 'android'],
                'player_skip': ['webpage']
            }
        },
        'nocheckcertificate': True,
        'ignoreerrors': False
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def transcribe_audio(video_path: str):
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(video_path)
    return [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments]

def find_3_viral_clips(transcript: list):
    prompt = f"""
Find the top 3 DIFFERENT viral/engaging moments (30 to 50 seconds each) from this transcript for YouTube Shorts:
{json.dumps(transcript[:150], indent=2)}

Return ONLY a JSON array of 3 objects matching this schema:
[
  {{"clip_no": 1, "start_time": <number_sec>, "end_time": <number_sec>, "title": "Part 1 Title"}},
  {{"clip_no": 2, "start_time": <number_sec>, "end_time": <number_sec>, "title": "Part 2 Title"}},
  {{"clip_no": 3, "start_time": <number_sec>, "end_time": <number_sec>, "title": "Part 3 Title"}}
]
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

# --- Background Worker Pipeline ---
def process_pipeline(task_id: str, channel_id: str, direct_url: str):
    try:
        db["tasks"][task_id]["status"] = "processing"

        if direct_url:
            target_url = direct_url
            video_title = "တိုက်ရိုက်ထည့်သွင်းထားသော ဗီဒီယို"
        else:
            db["tasks"][task_id]["message"] = "ချန်နယ်မှ ဗီဒီယိုအသစ်ကို ရှာဖွေနေပါသည်..."
            latest = fetch_latest_youtube_video(channel_id)
            if not latest:
                raise Exception("Channel ID မှားယွင်းနေပါသည် သို့မဟုတ် ဗီဒီယိုမရှိပါ။")
            target_url = latest["url"]
            video_title = latest["title"]

        db["tasks"][task_id]["title"] = video_title
        raw_path = os.path.join(OUTPUT_DIR, f"{task_id}_raw.mp4")

        db["tasks"][task_id]["message"] = "ဗီဒီယိုကို စနစ်ထဲသို့ ဒေါင်းလုဒ်ဆွဲနေပါသည်..."
        download_video(target_url, raw_path)

        db["tasks"][task_id]["message"] = "AI ဖြင့် စကားပြောသံများကို စာသားပြောင်းနေပါသည်..."
        transcript = transcribe_audio(raw_path)

        db["tasks"][task_id]["message"] = "အကောင်းဆုံး Viral အပိုင်း ၃ ပုဒ်ကို ရှာဖွေနေပါသည်..."
        clips = find_3_viral_clips(transcript)

        base_time = datetime.now(timezone.utc)
        generated_clips = []
        uploaded_urls = []

        for idx, item in enumerate(clips):
            clip_name = f"{task_id}_part_{idx+1}.mp4"
            clip_path = os.path.join(OUTPUT_DIR, clip_name)
            duration = item["end_time"] - item["start_time"]
            
            db["tasks"][task_id]["message"] = f"အပိုင်း {idx+1} ကို 9:16 Anti-Copyright Shorts အဖြစ် ဖြတ်ညှပ်နေပါသည်..."
            render_anti_copyright_short(raw_path, item["start_time"], duration, clip_path)
            generated_clips.append(f"/outputs/{clip_name}")

            # ၄ နာရီခြား တစ်ပုဒ်စီ Auto Schedule တင်ခြင်း (4h, 8h, 12h)
            sched_time = (base_time + timedelta(hours=(idx + 1) * 4)).strftime("%Y-%m-%dT%H:%M:%SZ")
            
            db["tasks"][task_id]["message"] = f"အပိုင်း {idx+1} ကို YouTube ပေါ်သို့ Schedule အဖြစ် အလိုအလျောက်တင်နေပါသည်..."
            up_url = upload_scheduled_short(clip_path, item.get("title", f"Part {idx+1}"), sched_time)
            if up_url:
                uploaded_urls.append(up_url)

        if os.path.exists(raw_path):
            os.remove(raw_path)

        db["tasks"][task_id]["status"] = "completed"
        
        if uploaded_urls:
            db["tasks"][task_id]["message"] = f"ဗီဒီယို {len(generated_clips)} ပုဒ်ကို ဖြတ်တောက်ပြီး YouTube Shorts ပေါ်သို့ ၄ နာရီခြားစီ အလိုအလျောက် Schedule (Auto-Post) တင်ပေးလိုက်ပါပြီ!"
        else:
            db["tasks"][task_id]["message"] = "ဗီဒီယို ၃ ပုဒ် ဖြတ်ညှပ်ပြီးပါပြီ! (Auto-Post ကို ကျော်သွားပါသည်)"
            
        db["tasks"][task_id]["result_url"] = generated_clips[0]
        db["tasks"][task_id]["all_clips"] = generated_clips
        db["tasks"][task_id]["youtube_urls"] = uploaded_urls

    except Exception as e:
        db["tasks"][task_id]["status"] = "failed"
        db["tasks"][task_id]["message"] = f"အမှားဖြစ်ပေါ်ပါသည်: {str(e)}"

# --- API Endpoints ---
@app.post("/api/track")
def track_channel(channel_id: str = Form(...)):
    if channel_id not in db["tracked_channels"]:
        db["tracked_channels"].append(channel_id)
    return {"channels": db["tracked_channels"]}

@app.get("/api/channels")
def get_channels():
    return {"channels": db["tracked_channels"]}

@app.post("/api/create-clip")
def create_clip(background_tasks: BackgroundTasks, channel_id: str = Form(default=""), direct_url: str = Form(default="")):
    task_id = str(uuid.uuid4())[:8]
    db["tasks"][task_id] = {
        "id": task_id,
        "status": "queued",
        "message": "စတင်ရန် တန်းစီနေပါသည်...",
        "title": "",
        "result_url": None,
        "all_clips": [],
        "youtube_urls": []
    }
    background_tasks.add_task(process_pipeline, task_id, channel_id, direct_url)
    return {"task_id": task_id}

@app.get("/api/task/{task_id}")
def get_task(task_id: str):
    task = db["tasks"].get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    return task

@app.get("/", response_class=FileResponse)
def index_page():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
