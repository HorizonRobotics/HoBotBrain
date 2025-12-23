import asyncio
import websockets
import time
import json
import pyaudio
import os
import base64
from loguru import logger
import numpy as np
from queue import Queue
import threading
import traceback


class DRobotC:
    def __init__(
        self, 
        host:str = "180.76.187.170", 
        port:int = 10071,
        device_name:str = "ReSpeaker",  # 使用支持全双工的音频设备
        token:str = "",
        ):
        logger.add("drobotc.log", level="INFO")             # 设置logger

        logger.info(f"Init DRobotC ......")
        self.host = host
        self.port = port
        self.device_name = device_name
        self.token = token
        
        # websocket设置, 连接
        self.audio_url = f"ws://{self.host}:{self.port}"    # 设置ws地址
        self.audio_ws: websockets.ClientConnection          # 设置ws连接
        logger.info(f"Websocket url: {self.audio_url}")
        self.heartbeat_interval = 10                        # 设置心跳间隔, 单位ms

        # 获取录音和播放设备
        self.p = pyaudio.PyAudio()
        self.audio_device_id = self._get_device_by_name(self.device_name, self.p)   # 获取设备id
        assert self.audio_device_id != -1, f"未找到设备: {self.device_name}"
        self.audio_device_rate = int(self.p.get_device_info_by_index(self.audio_device_id)['defaultSampleRate'])   # 获取设备采样率
        logger.info(f"Audio device info: {self.p.get_device_info_by_index(self.audio_device_id)}")

        # 录音设置, 发送
        self.send_queue = Queue(maxsize=100000)   # 发送队列
        self.channels = 1           # 声道数
        self.record_rate = 16000    # 采样率, 录音时目标采样率
        self.record_chunk = 512     # 缓冲区大小
        self.record_device_chunk = int(self.record_chunk * self.audio_device_rate / self.record_rate)
        logger.info(f"Record thread record chunk size:{self.record_device_chunk}")
        self.send_audio_interval = 0.025    # 设置发送音频间隔, 单位秒, 一个chunk默认是512, 16000采样率, 所以一个chunk是0.032秒, 所以发送音频间隔是0.032秒

        # 播放设置, 接收
        self.recv_queue = Queue(maxsize=100000)   # 接收队列
        self.recv_rate = 24000      # 采样率, 播放时目标采样率
        self.recv_chunk = 1024      # 缓冲区大小
        self.recv_device_chunk = int(self.recv_chunk * self.audio_device_rate / self.recv_rate)
        logger.info(f"Play thread recv chunk size:{self.recv_device_chunk}")
        self.play_chat_id = 0

        # 位置接收, QA接收, 信号接收
        self.text_queue = Queue(maxsize=100000)   # 位置接收队列
        # 接收文本信号, 例如ROS发过来的信号
        self.control_queue = Queue(maxsize=10000)
        self.is_introduce = False   # 用于判断是否在介绍展厅

        # 定义和启动线程
        self.record_stream = None   # 录音流, 线程中会用, 先定义下
        self.send_thread = threading.Thread(target=self._record_audio)
        self.send_thread.start()
        logger.info(f"Record thread started")
        self.play_stream = None     # 播放流, 线程中会用, 先定义下
        self.play_thread = threading.Thread(target=self._play_audio)
        self.play_thread.start()
        logger.info(f"Play thread started")

        logger.info(f"Init DRobotC done")

    def get_text_queue(self):
        return self.text_queue
    
    def get_control_queue(self):
        return self.control_queue

    def _get_device_by_name(self, name:str = "MCP", p = None) -> int:
        """查找全双工音频设备
        
        Args:
            name: 设备名称
            p: PyAudio实例
            
        Returns:
            设备ID, 如果未找到返回-1
        """
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if name in dev_info['name']:
                logger.info(f"Find Device, Input:{dev_info['maxInputChannels']}, Output:{dev_info['maxOutputChannels']}")
                # 检查设备是否同时支持输入和输出
                if dev_info['maxInputChannels'] > 0 and dev_info['maxOutputChannels'] > 0:
                    return i
        return -1
    
    def _resample_audio(self, audio_data, src_rate, dst_rate):
        """重采样音频数据
        
        Args:
            audio_data: 原始音频数据（字节格式）
            src_rate: 原始采样率
            dst_rate: 目标采样率
        
        Returns:
            重采样后的音频数据（字节格式）
        """
        # 将字节数据转换为numpy数组
        samples = np.frombuffer(audio_data, dtype=np.float32)
        
        # 计算重采样后的长度
        new_length = int(len(samples) * dst_rate / src_rate)
        
        # 创建时间数组
        time_old = np.linspace(0, len(samples), len(samples))
        time_new = np.linspace(0, len(samples), new_length)
        
        # 使用线性插值进行重采样
        resampled = np.interp(time_new, time_old, samples)
        
        # 转换回int16类型并返回字节格式
        return resampled.astype(np.float32)

    def _record_audio(self):
        try:
            self.record_stream = self.p.open(
                format=pyaudio.paInt16, 
                channels=self.channels, 
                rate=self.audio_device_rate, 
                input=True, 
                frames_per_buffer=self.record_device_chunk, 
                input_device_index=self.audio_device_id
            )
            while True:
                audio_data = self.record_stream.read(self.record_device_chunk)
                # 重采样
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                resampled = np.interp(
                    np.linspace(0, len(audio_array), self.record_chunk),   # time new
                    np.arange(len(audio_array)),                    # time old
                    audio_array                                     # samples
                    ).astype(np.int16)
                audio_data = resampled.tobytes()
                self.send_queue.put(audio_data)
                time.sleep(self.send_audio_interval / 2)
        except Exception as e:
            logger.error(f"Record thread error: {e}")
            self.record_stream.stop_stream()
            self.record_stream.close()
            self.record_stream = None
            self.p.terminate()
            self.send_thread.join()
            self.send_thread = None
    
    def _play_audio(self):
        try:
            self.play_stream = self.p.open(
                format=pyaudio.paFloat32, 
                channels=self.channels, 
                rate=self.audio_device_rate, 
                output=True, 
                frames_per_buffer=1024,
                output_device_index=self.audio_device_id
            )
            
            while True:
                if not self.recv_queue.empty():
                    audio_data, chat_id = self.recv_queue.get_nowait()
                    if chat_id in [-100, -101, -102]:
                        self.is_introduce = True
                    if chat_id != self.play_chat_id and chat_id not in  [-100, -101, -102]:
                        logger.warning(f"drop audio, play chat id: {self.play_chat_id} -> {chat_id}")
                        continue
                    if self.is_introduce and chat_id not in [-100, -101, -102]:
                        self.is_introduce = False

                        # 介绍完毕了
                        text_data_str = f"signal::introduce_end::{self.is_introduce}"
                        # 放入队列
                        self.text_queue.put(text_data_str)
                        logger.info(f"recv signal: `{text_data_str}`")
                    self.play_stream.write(audio_data)
                    time.sleep(0.01)   # 线程休息一下, chunk / rate = 1024 / 24000 = 0.04, sleep 10ms即可
                else:
                    if self.is_introduce:
                        self.is_introduce = False

                        # 介绍完毕了
                        text_data_str = f"signal::introduce_end::{self.is_introduce}"
                        # 放入队列
                        self.text_queue.put(text_data_str)
                        logger.info(f"recv signal: `{text_data_str}`")
                    # 不知道为什么? 每次队列为空再有数据时都要重启线程, 否则播放失败
                    self.play_stream.stop_stream()
                    self.play_stream.close()
                    self.play_stream = self.p.open(
                        format=pyaudio.paFloat32, 
                        channels=self.channels, 
                        rate=self.audio_device_rate, 
                        output=True, 
                        frames_per_buffer=1024,
                        output_device_index=self.audio_device_id
                    )
                    time.sleep(0.01)
        except Exception as e:
            logger.error(f"Play thread error: {e}")
            self.play_stream.stop_stream()
            self.play_stream.close()
            self.play_stream = None
            self.p.terminate()
            self.play_thread.join()
            self.play_thread = None
    
    async def connect(self):
        backoff = 1
        while True:
            try:
                logger.info(f"Connecting {self.audio_url}")
                async with websockets.connect(self.audio_url, max_size=10 * 1024 * 1024) as ws:
                    self.audio_ws = ws
                    logger.info("Connected successfully")
                    await asyncio.gather(
                        self.send_audio_loop(),
                        self.receive_audio_loop(),
                        self.heartbeat_loop(),
                        self.send_control_loop()
                    )
            except Exception as e:
                logger.error(f"Connect error: {e}")
            logger.info(f"{backoff} second retry ...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

    async def heartbeat_loop(self):
        try:
            while True:
                if self.audio_ws:
                    await self.audio_ws.send(json.dumps({
                        "token": self.token,
                        "type": "heartbeat",
                        "data": "ping",
                        "timestamp": time.time()
                    }))
                await asyncio.sleep(self.heartbeat_interval)
        except Exception as e:
            logger.error(f"heartbeat error: {e}")
    
    async def send_audio_loop(self):
        while True:
            try:
                if not self.send_queue.empty():
                    audio_data = self.send_queue.get()
                    audio_message = {
                        "token": self.token,
                        "type": "audio",
                        "data": base64.b64encode(audio_data).decode('ascii'),
                        "timestamp": time.time()
                    }
                    await self.audio_ws.send(json.dumps(audio_message))
                await asyncio.sleep(self.send_audio_interval)
            except Exception as e:
                logger.error(f"send audio chunk error: {e}")
                self.send_thread.join()
                self.send_thread = None
                self.p.terminate()

    async def receive_audio_loop(self):
        async for message in self.audio_ws:
            try:
                msg = json.loads(message)
                msg_type = msg.get("type", "unknown")
                msg_data = msg.get("data", "")
                msg_timestamp = msg.get("timestamp", 0)
                msg_chat_id = msg.get("chat_id", None)
                
                if msg_type == "heartbeat":
                    logger.debug(f"Recv server heatbeat message, time cost {time.time() - msg_timestamp} seconds")
                elif msg_type == "audio":
                    assert isinstance(msg_chat_id, int), "chat_id is not int"
                    self.play_chat_id = msg_chat_id # 更新播放的chat_id
                    # 处理音频数据
                    audio_data = base64.b64decode(msg_data)
                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                    audio_array = self._resample_audio(audio_array, 24000, self.audio_device_rate)
                    # 放入队列
                    self.recv_queue.put((audio_array.tobytes(), msg_chat_id))
                elif msg_type == "loc":
                    # 解码, loc_data是字符串, 是大模型输出的, 可以json.loads()
                    loc_data = base64.b64decode(msg_data).decode()
                    # 处理loc数据
                    loc_data = dict(json.loads(loc_data))
                    floor = "unknown" if loc_data['floor'] == "" else loc_data['floor']
                    room = "unknown" if loc_data['room'] == "" else loc_data['room']
                    object2find = "unknown" if loc_data['object'] == "" else loc_data['object']
                    loc_data_str = f"loc::{floor},{room},{object2find}::{msg_chat_id}"
                    # 放入队列
                    self.text_queue.put(loc_data_str)
                    logger.debug(f"recv loc: `{loc_data_str}`")
                elif msg_type == "signal":
                    # 解码
                    text_data = base64.b64decode(msg_data).decode()
                    # 处理信号数据
                    text_data_str = f"signal::{text_data}::{msg_chat_id}"
                    # 放入队列
                    self.text_queue.put(text_data_str)
                    logger.debug(f"recv signal: `{text_data_str}`")
                elif msg_type == "qa":
                    # 解码
                    qa_data = base64.b64decode(msg_data).decode()
                    # 处理qa数据
                    qa_data = dict(json.loads(qa_data))
                    qa_text = qa_data['text']
                    qa_type = qa_data['type']
                    if qa_type == "user":
                        qa_data_str = f"qa::User:{qa_text}::{msg_chat_id}"
                    elif qa_type == "assistant":
                        qa_data_str = f"qa::Assistant:{qa_text}::{msg_chat_id}"
                    else:
                        raise ValueError(f"recv wrong qa type: {qa_type}")
                    # 放入队列
                    self.text_queue.put(qa_data_str)
                    logger.debug(f"recv qa: `{qa_data_str}`")
                else:
                    raise ValueError(f"recv wrong message: {msg_type}")
                
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"msg:{message}")
                logger.error(traceback.format_exc())
                # self.play_thread.join()
                # self.play_thread = None
                # self.p.terminate()

    async def send_control_loop(self):
        while True:
            if not self.control_queue.empty():
                control_data = self.control_queue.get()
                await self.audio_ws.send(json.dumps({"type": "control", "data": base64.b64encode(control_data.encode()).decode('ascii')}))
                await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0.05)

def main():
    client = DRobotC()
    try:
        asyncio.run(client.connect())
    except KeyboardInterrupt:
        logger.info("[exit] Ctrl+C triggered")
    finally:
        client.p.terminate()

if __name__ == "__main__":
    main()
