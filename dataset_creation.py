from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames
from moviepy.editor import VideoFileClip
import json
import random
import cv2
import os
import numpy as np
import consts

game_list = getListGames(split="v1")
random.seed(0)
np.random.seed(0)
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=consts.FILE_PATH)
mySoccerNetDownloader.password = "s0cc3rn3t"


def extract_frames(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return frames


def delete_all_matches():
    for i in range(consts.NUMBER_OF_VIDEOS):
        try:
            os.remove(consts.FILE_PATH + game_list[i] + "/1.mkv")
            os.remove(consts.FILE_PATH + game_list[i] + "/Labels.json")
        except FileNotFoundError:
            continue


def get_labels_from_time(time):
    goal_frames = {time * consts.FPS, time * consts.FPS + 1}
    return [1 if frame in goal_frames else 0 for frame in range(consts.FRAME_AMOUNT)]


def create_clips():
    goal_time_lst = []
    for i in range(consts.NUMBER_OF_VIDEOS):
        goal_time_lst.append([])
        mySoccerNetDownloader.downloadGameIndex(i, files=["1.mkv", "Labels.json"])
        video = VideoFileClip(consts.FILE_PATH + game_list[i] + "/1.mkv")
        with open(consts.FILE_PATH + game_list[i] + "/Labels.json", 'r') as labels_file:
            data = json.loads(labels_file.read())
        soccer_ball_times = []
        for annotation in data["annotations"]:
            if annotation["label"] == "soccer-ball" and annotation["gameTime"][0] == "1":
                game_time = annotation["gameTime"].split(" - ")[1]
                soccer_ball_times.append(60 * int(game_time.split(":")[0]) + int(game_time.split(":")[1]))

        j = 0
        for goal in soccer_ball_times:
            j += 1
            output_filename = f"game_{i + 1},clip_{j}.mp4"
            sec_before_goal = random.randint(5, 25)
            goal_time_lst[-1].append(sec_before_goal)
            start_time = goal - sec_before_goal
            clip = video.subclip(start_time, start_time + consts.CLIP_DURATION)
            clip = clip.set_duration(consts.CLIP_DURATION).set_fps(consts.FPS)
            clip.write_videofile(consts.FILE_PATH + output_filename, fps=consts.FPS, audio=False)

        for not_goal in range(max(0, consts.CLIPS_PER_GAME - len(soccer_ball_times))):
            j += 1
            output_filename = f"game_{i + 1},clip_{j}.mp4"
            while True:
                start_time = random.randint(0, consts.VIDEO_LENGTH_SECS)
                for goal in soccer_ball_times:
                    if goal + 30 >= start_time >= goal - 30:
                        continue
                break
            goal_time_lst[-1].append(-1)
            clip = video.subclip(start_time, start_time + consts.CLIP_DURATION)
            clip = clip.set_duration(consts.CLIP_DURATION).set_fps(consts.FPS)
            clip.write_videofile(consts.FILE_PATH + output_filename, fps=consts.FPS, audio=False)
        video.close()

    with open(consts.FILE_PATH + 'goal_time_lst.json', 'w') as created_labels_file:
        json.dump(goal_time_lst, created_labels_file)


def get_clips_and_labels():
    if consts.REDO_CLIPS:
        create_clips()

    with open(consts.FILE_PATH + 'goal_time_lst.json', 'r') as file:
        goal_time_lst = json.load(file)

    goal_time_lst = goal_time_lst[:consts.NUMBER_OF_VIDEOS]
    clips = []
    for game in range(consts.NUMBER_OF_VIDEOS):
        for clip in range(consts.CLIPS_PER_GAME):
            clips.append(extract_frames(consts.FILE_PATH + f"game_{game + 1},clip_{clip + 1}.mp4"))
            print(f"game_{game + 1},clip_{clip + 1}.mp4")
    clips = np.array(clips)
    labels = []
    for i in range(len(goal_time_lst)):
        for j in range(consts.CLIPS_PER_GAME):
            labels.append(get_labels_from_time(goal_time_lst[i][j]))
    labels = np.array(labels)
    print("clips shape: " + str(clips.shape))
    print("labels shape: " + str(labels.shape))
    return clips, labels
