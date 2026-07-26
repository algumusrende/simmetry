#!/usr/bin/env bash
# Download two YouTube videos as audio, trim, and join into one MP3.
#   Video 1 -> first 4:20
#   Video 2 -> first 1:00:00
#   Result  -> combined.mp3  (~1h 4m 20s)
#
# Requirements: yt-dlp, ffmpeg
#   macOS:  brew install yt-dlp ffmpeg
#   Ubuntu: sudo apt install ffmpeg && pipx install yt-dlp   (or pip install yt-dlp)
set -euo pipefail

V1="https://youtu.be/SsebrRbTDL0"
V2="https://youtu.be/M-he70PcfR4"

echo "==> Downloading audio..."
yt-dlp -x --audio-format mp3 --audio-quality 0 -o vid1.mp3 "$V1"
yt-dlp -x --audio-format mp3 --audio-quality 0 -o vid2.mp3 "$V2"

echo "==> Trimming..."
ffmpeg -y -i vid1.mp3 -t 00:04:20 -c copy vid1_trim.mp3   # first 4:20
ffmpeg -y -i vid2.mp3 -t 01:00:00 -c copy vid2_trim.mp3   # first hour

echo "==> Joining into combined.mp3..."
printf "file 'vid1_trim.mp3'\nfile 'vid2_trim.mp3'\n" > concat_list.txt
ffmpeg -y -f concat -safe 0 -i concat_list.txt -c:a libmp3lame -q:a 2 combined.mp3

echo "==> Done: combined.mp3"
