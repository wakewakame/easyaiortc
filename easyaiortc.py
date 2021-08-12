import asyncio
import nest_asyncio
nest_asyncio.apply()
import random
import numpy as np
import cv2
from av import VideoFrame
from aiortc import (
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.contrib.signaling import BYE, ApprtcSignaling
import multiprocessing as mp
import ctypes
import queue

import IPython
from IPython.display import display, HTML


class CustomVideoStreamTrack(VideoStreamTrack):
    def __init__(self, recv_queue, send_queue):
        super().__init__()
        # 映像受信キュー
        self.recv_queue = recv_queue

        # 映像送信キュー
        self.send_queue = send_queue

        # 最後に送信した画像
        self.last_image = None

    def addTrack(self, track):
        # 受信した映像を受け取るタスク
        async def recv_frame():
            while True:
                img = (await track.recv()).to_ndarray(format="rgb24")
                try:
                    # 受信した映像をキューに追加
                    self.recv_queue.put_nowait(img)
                except queue.Full:
                    pass
        asyncio.Task(recv_frame())

    async def recv(self):
        # まだ映像送信キューから一度も画像を受け取っていない場合は
        # キューに画像が追加されるまで待機する
        if self.last_image is None:
            while self.last_image is None:
                try:
                    self.last_image = self.send_queue.get_nowait()
                except queue.Empty:
                    # イベントループの他の処理を妨げないようにする
                    await asyncio.sleep(0.1)
        else:
            try:
                # 映像送信キューから画像を取得
                self.last_image = self.send_queue.get_nowait()
            except queue.Empty:
                pass

        # 映像送信
        frame = VideoFrame.from_ndarray(self.last_image, format="rgb24")
        frame.pts, frame.time_base = await self.next_timestamp()
        return frame


async def start_webrtc(pc, signaling, recv_queue, send_queue, result_pipe_child):
    # 映像の送受信を処理するクラス
    custom_video_track = CustomVideoStreamTrack(recv_queue, send_queue)

    # 受信する映像がある場合はトラックを追加
    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            custom_video_track.addTrack(track)

    # 接続の開始
    params = await signaling.connect()
    if params["is_initiator"] == "true":
        pc.addTrack(custom_video_track)
        await pc.setLocalDescription(await pc.createOffer())
        await signaling.send(pc.localDescription)
        result_pipe_child.send(params["room_link"])
        result_pipe_child.close()
    else:
        result_pipe_child.send(None)
        result_pipe_child.close()
        return

    # 終了するまでループ
    while True:
        obj = await signaling.receive()
        if isinstance(obj, RTCSessionDescription):
            await pc.setRemoteDescription(obj)
            if obj.type == "offer":
                add_tracks()
                await pc.setLocalDescription(await pc.createAnswer())
                await signaling.send(pc.localDescription)
        elif isinstance(obj, RTCIceCandidate):
            await pc.addIceCandidate(obj)
        elif obj is BYE:
            break


def multiprocess_main(recv_queue, send_queue, room_id, result_pipe_child):
    # この関数は別プロセスで実行される
    signaling = ApprtcSignaling(room_id)
    pc = RTCPeerConnection()
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(start_webrtc(pc, signaling, recv_queue, send_queue, result_pipe_child))
    finally:
        loop.run_until_complete(signaling.close())
        loop.run_until_complete(pc.close())


class EasyAppRTC:
    def __init__(self, room_id=None, preview=False, width=1280, height=720):
        # AppRTCの部屋IDが未指定の場合はここで部屋IDを決める
        if room_id == None:
            room_id = "".join([random.choice("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-") for x in range(8)])

        # 映像受信キュー
        self.__recv_queue = mp.Queue(maxsize=1)

        # 映像送信キュー
        self.__send_queue = mp.Queue(maxsize=1)

        # 作成されたAppRTCの部屋URLを受け取るPipe
        result_pipe_parent, result_pipe_child = mp.Pipe()

        # multiprocessingのProcess上でaiortcを動かすことにより、CPythonのGILによる問題を回避する。
        # これを行わない場合、処理時間を要する画像処理などを行うと映像の受信処理が追いつかなくなる。
        self.__process = mp.Process(target=multiprocess_main, args=(self.__recv_queue, self.__send_queue, room_id, result_pipe_child))
        self.__process.daemon = True
        self.__process.start()

        # 作成されたAppRTCの部屋URLを取得
        self.__room_url = result_pipe_parent.recv()
        if self.__room_url is None:
            print("failed to create room")
            self.close()
            return

        # AppRTCの画面を表示
        if preview:
            display(HTML(
                '<iframe id="inlineFrameExample" title="Inline Frame Example" width="{}" height="{}" src="{}" allow="microphone; camera"></iframe>'
                    .format(width, height, self.__room_url)
            ))
    
    def __del__(self):
        self.close()
    
    # 接続の終了
    def close(self):
        if self.__process.is_alive():
            self.__process.kill()
            self.__process.join()
    
    # 接続状態の取得
    def is_alive(self):
        return self.__process.is_alive()

    # AppRTCの部屋URLを取得
    def room_url(self):
        return self.__room_url
    
    # AppRTCから受信した映像を取得
    # 受信した映像が無ければNoneを返す
    def get(self):
        if not self.is_alive():
            return None
        img = None
        try:
            img = self.__recv_queue.get_nowait()
        except queue.Empty:
            pass
        return img
    
    # AppRTCに映像を送信する
    def put(self, img):
        if not self.is_alive():
            return
        try:
            self.__send_queue.put_nowait(img)
        except queue.Full:
            pass
