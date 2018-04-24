使用操作：setup以后，第一次先创建三个新文件夹，确认父目录video_server里放有视频，然后运行get_video_sizes.py，生成video_size_?文件。训练网络运行multi_agent.py。

当前目录中，video_size_?文件里储存着6个码率的视频片段大小，由env.py读取。env.py是模拟环境，a3c.py负责生成神经网络，load_trace.py从traces文件夹里读取网络状况。
