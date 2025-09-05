import os
from functools import lru_cache
from typing import List, Optional, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
    IpBlocked,
    RequestBlocked,
    AgeRestricted,
    InvalidVideoId,
    PoTokenRequired,
    YouTubeTranscriptApiException,
    YouTubeRequestFailed,
)
from youtube_transcript_api._transcripts import FetchedTranscript
from youtube_transcript_api.proxies import WebshareProxyConfig


app = FastAPI(title="YouTube Transcript Service")

# CORS: allow all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_languages(lang: Optional[str]) -> List[str]:
    """Parse comma-separated language codes into a priority list.

    Examples:
    - None or "" -> ["en"]
    - "de,en" -> ["de", "en"]
    - "EN" -> ["en"]
    """
    if not lang:
        return ["en"]
    languages = [code.strip().lower() for code in lang.split(",")]
    return [code for code in languages if code]


def _to_srt_timestamp(seconds: float) -> str:
    milliseconds_total = int(round(seconds * 1000))
    hours = milliseconds_total // 3_600_000
    minutes = (milliseconds_total % 3_600_000) // 60_000
    secs = (milliseconds_total % 60_000) // 1000
    millis = milliseconds_total % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _to_vtt_timestamp(seconds: float) -> str:
    milliseconds_total = int(round(seconds * 1000))
    hours = milliseconds_total // 3_600_000
    minutes = (milliseconds_total % 3_600_000) // 60_000
    secs = (milliseconds_total % 60_000) // 1000
    millis = milliseconds_total % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _format_txt(transcript: FetchedTranscript) -> str:
    return "\n".join(snippet.text for snippet in transcript)


def _format_srt(transcript: FetchedTranscript) -> str:
    lines: List[str] = []
    for idx, snippet in enumerate(transcript, start=1):
        start = _to_srt_timestamp(snippet.start)
        end = _to_srt_timestamp(snippet.start + snippet.duration)
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(snippet.text)
        lines.append("")  # blank line between cues
    return "\n".join(lines).rstrip() + "\n"


def _format_vtt(transcript: FetchedTranscript) -> str:
    lines: List[str] = ["WEBVTT", ""]
    for snippet in transcript:
        start = _to_vtt_timestamp(snippet.start)
        end = _to_vtt_timestamp(snippet.start + snippet.duration)
        lines.append(f"{start} --> {end}")
        lines.append(snippet.text)
        lines.append("")  # blank line between cues
    return "\n".join(lines).rstrip() + "\n"


@lru_cache(maxsize=1)
def _get_api() -> YouTubeTranscriptApi:
    """Initialize YouTubeTranscriptApi, optionally with Webshare proxies.

    Controlled via env vars:
    - WEBSHARE_USER
    - WEBSHARE_PASS
    - WEBSHARE_COUNTRIES (comma-separated, e.g., "us,de")
    """
    username = os.getenv("WEBSHARE_USER")
    password = os.getenv("WEBSHARE_PASS")
    countries_raw = os.getenv("WEBSHARE_COUNTRIES")

    proxy_config = None
    if username and password:
        filter_ip_locations = None
        if countries_raw:
            filter_ip_locations = [c.strip() for c in countries_raw.split(",") if c.strip()]
        proxy_config = WebshareProxyConfig(
            proxy_username=username,
            proxy_password=password,
            filter_ip_locations=filter_ip_locations,
        )

    return YouTubeTranscriptApi(proxy_config=proxy_config)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/transcript")
def get_transcript(
    video_id: Optional[str] = Query(None, alias="videoId"),
    lang: Optional[str] = Query("en"),
    output_format: Literal["json", "srt", "vtt", "txt"] = Query(
        "json", alias="format"
    ),
    preserve_formatting: bool = Query(False, alias="preserveFormatting"),
):
    if not video_id:
        raise HTTPException(
            status_code=400, detail="Missing required query parameter 'videoId'."
        )

    languages = _parse_languages(lang)

    try:
        api = _get_api()
        transcript_list = api.list(video_id)
        try:
            transcript = transcript_list.find_transcript(languages)
        except NoTranscriptFound:
            transcript = transcript_list.find_generated_transcript(languages)

        fetched = transcript.fetch(preserve_formatting=preserve_formatting)

        if output_format == "json":
            # raw list of {text, start, duration}
            return JSONResponse(content=fetched.to_raw_data())
        elif output_format == "txt":
            return PlainTextResponse(content=_format_txt(fetched), media_type="text/plain")
        elif output_format == "srt":
            return PlainTextResponse(content=_format_srt(fetched), media_type="text/plain")
        elif output_format == "vtt":
            return PlainTextResponse(content=_format_vtt(fetched), media_type="text/plain")

        # Should never happen due to Literal typing, but keep a safe default
        return JSONResponse(content=fetched.to_raw_data())

    except (
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
        VideoUnplayable,
        IpBlocked,
        RequestBlocked,
        AgeRestricted,
        InvalidVideoId,
        PoTokenRequired,
        YouTubeTranscriptApiException,
    ) as e:
        # Return a clear 404-like not found/blocked/unavailable error
        raise HTTPException(status_code=404, detail=str(e)) from e
    except YouTubeRequestFailed as e:
        # Upstream YouTube error
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:  # pragma: no cover - generic safety net
        raise HTTPException(status_code=500, detail="Internal Server Error") from e


