import os
import re
import json
from google import genai
from dotenv import load_dotenv
from google.genai import types
from fastapi import FastAPI, HTTPException, Query, status
from typing import Optional
import yt_dlp  # For fetching transcripts

app = FastAPI()

# Initialize at startup
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
client = genai.Client(api_key=api_key)

def get_youtube_transcript(video_id: str) -> str:
    """Fetch transcript from YouTube video using yt-dlp"""
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt',
        'quiet': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            if 'subtitles' in info and info['subtitles']:
                # Get the first available English transcript
                subtitles = info['subtitles'].get('en', [{}])[0]
                if 'data' in subtitles:
                    return subtitles['data']
            raise ValueError("No English transcript available for this video")
    except Exception as e:
        raise ValueError(f"Failed to fetch transcript: {str(e)}")

def summarize_transcript_with_gemini(transcript: str) -> dict:
    """Summarize text using Gemini API"""
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"""
Summarize this YouTube transcript in JSON format:
{{
   "topic": "brief topic title",
   "summary": "concise summary",
   "key_points": ["list", "of", "key", "points"]
}}

Transcript:
\"\"\"
{transcript}
\"\"\"
""")
            ]
        )
    ]

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.3,
            response_mime_type="application/json",
        )
    )
    
    return json.loads(response.text)

def extract_youtube_id(url: str) -> str:
    """Extract video ID from various YouTube URL formats"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11})',
        r'youtu\.be\/([0-9A-Za-z_-]{11})',
        r'embed\/([0-9A-Za-z_-]{11})',
        r'shorts\/([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Invalid YouTube URL format")

@app.get("/summarize")
async def summarize_youtube(
    url: Optional[str] = Query(None, description="YouTube URL"),
    transcript: Optional[str] = Query(None, description="Direct transcript text"),
):
    """
    Summarize YouTube video either by:
    - Providing a YouTube URL (automatically fetches transcript)
    - Or directly providing transcript text
    
    Example requests:
    /summarize?url=https://www.youtube.com/watch?v=VIDEO_ID
    /summarize?transcript=Your+text+here
    """
    if not url and not transcript:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'url' or 'transcript' parameter is required"
        )
    
    try:
        if url:
            video_id = extract_youtube_id(url)
            transcript = get_youtube_transcript(video_id)
        
        return summarize_transcript_with_gemini(transcript)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)