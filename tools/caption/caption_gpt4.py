import argparse
import csv
import os

import requests
import tqdm

from .utils import extract_frames, prompts, read_video_list


def get_caption(frame, prompt, api_key):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame[0]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame[1]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame[2]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame[3]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame[4]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame[5]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame[6]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame[7]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame[8]}"}},
                ],
            }
        ],
        "max_tokens": 300,
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
    print(response.json())
    caption = response.json()["choices"][0]["message"]["content"]
    caption = caption.replace("\n", " ")
    return caption


def main(args):
    # ======================================================
    # 1. read video list
    # ======================================================
    videos = read_video_list(args.video_folder, args.output_file)
    f = open(args.output_file, "a")
    writer = csv.writer(f)

    # ======================================================
    # 2. generate captions
    # ======================================================
    for video in tqdm.tqdm(videos):
        video_path = os.path.join(args.video_folder, video)
        frame, length = extract_frames(video_path, points=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), base_64=True)
        if len(frame) < 9:
            continue

        prompt = prompts[args.prompt]
        caption = get_caption(frame, prompt, args.key)

        writer.writerow((video, caption, length))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_folder", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--prompt", type=str, default="nine_frames")
    parser.add_argument("--key", type=str)
    args = parser.parse_args()

    main(args)
