import pandas as pd
import os

if __name__ == '__main__':
    os.chdir('C:/Users/Riccardo/Documents/TTF_M+3_Trades_Tick')
    frames = []
    for file in os.listdir():
      if file.endswith('json'):
        frame = pd.read_json(file)
        frames.append(frame)

    data = pd.concat(frames)

    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data['dealDate'] = pd.to_datetime(data['dealDate'], unit='ms').dt.date

    data.to_csv('TTF_M+3_Trades_Tick.csv')