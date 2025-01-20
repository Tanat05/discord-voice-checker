# discord-voice-checker
Record Discord voice channel and display volume

# 라이브러리
```
discord.py, sounddevice, NumPy, SciPy, Matplotlib
```

# 주요기능
/녹음: 현재 접속 중인 음성 채널에서 10초간 음성을 녹음합니다.
/결과: 이전에 녹음된 음성 파일을 분석합니다.
자동 분석: 녹음이 완료되면, 자동으로 음성 파일을 분석하여 결과를 제공합니다.
시각화: 분석 결과는 파형(Waveform) 그래프와 **주파수 히스토그램(Frequency Histogram)**으로 알기 쉽게 시각화됩니다.
상세 정보: 평균 RMS, 평균 dB, 평균 dBA, 최대 피크 dB 등 다양한 음성 분석 정보를 제공합니다.
