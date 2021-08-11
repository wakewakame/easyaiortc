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
        self.recv_queue = recv_queue
        self.send_queue = send_queue
        self.last_image = None

    def addTrack(self, track):
        async def recv_frame():
            while True:
                img = (await track.recv()).to_ndarray(format="rgb24")
                try:
                    self.recv_queue.put_nowait(img)
                except queue.Full:
                    pass
        asyncio.Task(recv_frame())

    async def recv(self):
        if self.last_image is None:
            while self.last_image is None:
                try:
                    self.last_image = self.send_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.1)
        else:
            try:
                self.last_image = self.send_queue.get_nowait()
            except queue.Empty:
                pass
        frame = VideoFrame.from_ndarray(self.last_image, format="rgb24")
        frame.pts, frame.time_base = await self.next_timestamp()
        return frame


async def start_webrtc(pc, signaling, recv_queue, send_queue, result_pipe_child):
    custom_video_track = CustomVideoStreamTrack(recv_queue, send_queue)
    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            custom_video_track.addTrack(track)
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
        if room_id == None:
            room_id = "".join([random.choice("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-") for x in range(8)])
        self.__recv_queue = mp.Queue(maxsize=1)
        self.__send_queue = mp.Queue(maxsize=1)
        result_pipe_parent, result_pipe_child = mp.Pipe()
        self.__process = mp.Process(target=multiprocess_main, args=(self.__recv_queue, self.__send_queue, room_id, result_pipe_child))
        self.__process.daemon = True
        self.__process.start()
        self.__room_url = result_pipe_parent.recv()
        if self.__room_url is None:
            print("failed to create room")
            self.close()
            return
        if preview:
            display(HTML(
                '<iframe id="inlineFrameExample" title="Inline Frame Example" width="{}" height="{}" src="{}" allow="microphone; camera"></iframe>'
                    .format(width, height, self.__room_url)
            ))
    
    def __del__(self):
        self.close()
    
    def close(self):
        if self.__process.is_alive():
            self.__process.kill()
            self.__process.join()
    
    def is_alive(self):
        return self.__process.is_alive()

    def room_url(self):
        return self.__room_url
    
    def get(self):
        if not self.is_alive():
            return None
        img = None
        try:
            img = self.__recv_queue.get_nowait()
        except queue.Empty:
            pass
        return img
    
    def put(self, img):
        if not self.is_alive():
            return
        try:
            self.__send_queue.put_nowait(img)
        except queue.Full:
            pass


if __name__ == "__main__":
    rtc = EasyAppRTC()
    print("room url is {}".format(rtc.room_url()))
    deg = 0
    try:
        while rtc.is_alive():
            img = rtc.get()
            if img is None:
                continue
            n_h, n_w, n_ch = img.shape
            M = cv2.getRotationMatrix2D((n_w / 2, n_h / 2), deg, 1)
            img = cv2.warpAffine(img, M, (n_w, n_h))
            deg += 1
            rtc.put(img)
    except KeyboardInterrupt:
        pass
