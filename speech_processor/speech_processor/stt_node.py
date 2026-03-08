#!/usr/bin/env python3
import io
import time
import threading
import tempfile
import wave
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import pyaudio
from faster_whisper import WhisperModel

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5


class STTNode(Node):
    def __init__(self):
        super().__init__("stt_node")

        # Parameters
        self.declare_parameter("model_size", "tiny")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("record_seconds", 5)
        self.declare_parameter("publish_topic", "/stt")

        model_size = self.get_parameter("model_size").get_parameter_value().string_value
        device = self.get_parameter("device").get_parameter_value().string_value
        self.record_seconds = self.get_parameter("record_seconds").get_parameter_value().integer_value
        publish_topic = self.get_parameter("publish_topic").get_parameter_value().string_value

        # Publisher — publishes transcribed text
        self.pub = self.create_publisher(String, publish_topic, 10)

        # Load Whisper model
        self.get_logger().info(f" Loading Whisper model ({model_size})...")
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        self.get_logger().info(" STT Node ready — listening...")

        # Start listening in background thread
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def _listen_loop(self):
        pa = pyaudio.PyAudio()

        while self._running and rclpy.ok():
            try:
                self.get_logger().info("🎙 Recording...")
                stream = pa.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK
                )

                frames = []
                for _ in range(int(RATE / CHUNK * self.record_seconds)):
                    frames.append(stream.read(CHUNK, exception_on_overflow=False))

                stream.stop_stream()
                stream.close()

                # Save to temp WAV
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                wf = wave.open(tmp.name, "wb")
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pa.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b"".join(frames))
                wf.close()

                # Transcribe
                segments, _ = self.model.transcribe(tmp.name)
                text = " ".join(s.text for s in segments).strip()
                os.unlink(tmp.name)

                if text:
                    self.get_logger().info(f"📝 Heard: {text}")
                    msg = String()
                    msg.data = text
                    self.pub.publish(msg)
                else:
                    self.get_logger().info("(silence)")

            except Exception as e:
                self.get_logger().error(f"❌ STT error: {str(e)}")
                time.sleep(1.0)

        pa.terminate()

    def destroy_node(self):
        self._running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    try:
        node = STTNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ STT Node error: {e}")
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()