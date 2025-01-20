import discord
import asyncio
from discord.ext import commands
from discord import app_commands
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


SAMPLE_RATE = 48000
CHANNELS = 2
DURATION = 10

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix="!", intents=intents)

user_recording = {}

def format_negative_decibels(x, pos):
    return f"{x:.0f}"


def analyze_audio(filename):
    """
    음성 파일의 크기를 분석합니다.
    """
    rate, data = wav.read(filename)

    if len(data.shape) > 1:
        data = data.mean(axis=1)

    rms = np.sqrt(np.mean(data**2))

    decibels = 20 * np.log10(rms)

    frequencies = np.fft.fftfreq(data.size, 1 / rate)
    a_weighting = 12194**2 * frequencies**4 / (
        (frequencies**2 + 20.6**2)
        * np.sqrt((frequencies**2 + 107.7**2) * (frequencies**2 + 737.9**2))
        * (frequencies**2 + 12194**2)
    )
    a_weighted_data = np.fft.ifft(np.fft.fft(data) * a_weighting).real
    decibels_a = 20 * np.log10(np.sqrt(np.mean(a_weighted_data**2)))

    peak = np.max(np.abs(data))

    decibels_peak = 20 * np.log10(peak)

    crest_factor = peak / rms

    zero_crossings = np.where(np.diff(np.sign(data)))[0]
    zcr = len(zero_crossings) / (len(data) / rate)

    epsilon = 1e-10
    data_db = 20 * np.log10(np.abs(data) + epsilon)
    data_db_positive = data_db[data_db >= 0]
    hist, bin_edges = np.histogram(
        data_db_positive, bins=50
    )

    time = np.linspace(0, len(data) / rate, num=len(data))

    plt.figure(figsize=(10, 4))
    plt.plot(time, data)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    waveform_filename = filename.replace(".wav", "_waveform.png")
    plt.savefig(waveform_filename)
    plt.close()

    return {
        "rms": rms,
        "decibels": decibels,
        "decibels_a": decibels_a,
        "peak": peak,
        "decibels_peak": decibels_peak,
        "crest_factor": crest_factor,
        "zcr": zcr,
        "histogram": (hist, bin_edges),
        "waveform": waveform_filename,
    }


@bot.event
async def on_ready():
    print(f"{bot.user} (ID: {bot.user.id}) 봇 시작")
    try:
        synced = await bot.tree.sync()
        print(f"{len(synced)}개의 슬래쉬 커맨드가 동기화되었습니다.")
    except Exception as e:
        print(f"슬래쉬 커맨드 동기화 중 오류 발생: {e}")


@bot.tree.command(name="녹음", description=f"{DURATION}초 동안 음성을 녹음 후 분석합니다.")
async def record(interaction: discord.Interaction):
    """사용자가 접속한 음성 채널을 10초 동안 녹음합니다."""

    try:
        if interaction.user.voice is None or interaction.user.voice.channel is None:
            await interaction.response.send_message(
                "음성 채널에 접속한 후에 명령어를 사용해주세요.", ephemeral=True
            )
            return

        voice_channel = interaction.user.voice.channel
        user_id = interaction.user.id

        if user_id in user_recording and user_recording[user_id]:
            await interaction.response.send_message(
                "이미 녹음 중입니다. 잠시 후 다시 시도해주세요.", ephemeral=True
            )
            return

        filename = f"recording_{user_id}.wav"

        if os.path.exists(filename):
            os.remove(filename)

        user_recording[user_id] = True

        voice_client = await voice_channel.connect()

        await interaction.response.send_message(
            f"{voice_channel.name} 채널에서 {DURATION}초 동안 녹음됩니다."
        )

        frames = []

        async def record_audio():
            with sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16"
            ) as stream:
                for _ in range(int(DURATION * SAMPLE_RATE / 1024)):
                    frames.append(stream.read(1024)[0])
                remaining_frames = DURATION * SAMPLE_RATE % 1024
                if remaining_frames > 0:
                    frames.append(stream.read(remaining_frames)[0])

        await asyncio.gather(record_audio())

        audio_data = np.concatenate(frames, axis=0)

        wav.write(filename, SAMPLE_RATE, audio_data)

        await voice_client.disconnect()

        await analyze_and_send_results(interaction, filename)

    except Exception as e:
        await interaction.followup.send(f"녹음 중 오류가 발생했습니다: {e}")

    finally:
        user_recording[user_id] = False


async def analyze_and_send_results(interaction: discord.Interaction, filename: str):
    """
    음성 파일을 분석하고 결과를 생성하여 사용자에게 전송합니다.
    """
    try:
        analysis_result = analyze_audio(filename)

        hist, bin_edges = analysis_result["histogram"]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(bin_edges[:-1], hist, width=(bin_edges[1] - bin_edges[0]), align='edge')
        ax.set_xlabel("Decibel (dB)")
        ax.set_ylabel("Frequency")
        ax.set_title("Decibel Histogram")

        formatter = FuncFormatter(format_negative_decibels)
        ax.xaxis.set_major_formatter(formatter)

        histogram_filename = filename.replace(".wav", "_histogram.png")
        plt.savefig(histogram_filename)
        plt.close(fig)

        result_message = (
            f"🔊 **녹음 분석 결과:**\n\n"
            f"```"
            f"• 평균 RMS: {analysis_result['rms']:.4f}\n"
            f"• 평균 dB: {analysis_result['decibels']:.2f} dB\n"
            f"• 평균 dBA: {analysis_result['decibels_a']:.2f} dBA\n"
            f"• 최대 피크 dB: {analysis_result['decibels_peak']:.2f} dB\n"
            f"```\n"
        )

        waveform_file = discord.File(analysis_result["waveform"])

        histogram_file = discord.File(histogram_filename)

        await interaction.followup.send(
            content=result_message,
            files=[waveform_file, histogram_file]
        )

    except Exception as e:
        await interaction.followup.send(f"분석 중 오류가 발생했습니다: {e}")
        

@bot.tree.command(name="결과", description="저장되어있는 음성 파일을 분석합니다.")
async def analyze(interaction: discord.Interaction):
    """저장되어있는 음성 파일을 분석합니다."""

    user_id = interaction.user.id
    filename = f"recording_{user_id}.wav"

    if not os.path.exists(filename):
        await interaction.response.send_message(
            "녹음된 파일이 없습니다. 녹음 후 다시 시도해주세요.", ephemeral=True
        )
        return
    
    await interaction.response.defer()

    await analyze_and_send_results(interaction, filename)


bot.run("토큰을 입력해 주세요")
