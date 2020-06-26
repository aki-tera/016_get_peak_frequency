# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# データのパラメータ
N = 512          # サンプル数
dt = 0.00005          # サンプリング間隔
fc = 200  # カットオフ周波数
t = np.arange(0, N*dt, dt) # 時間軸
freq = np.linspace(0, 1.0/dt, N) # 周波数軸

x_range = 1000 #プロット幅の指定
display_range = 15 #周波数の表示数指定


#ファイル読み込み
#delimiterは、区切り文字の指定
#skiprowsは、スキップする行数を指定
#usecolsは、使用する列を指定
#特定の行のみ使用する場合、[使用する先頭行+1:使用する最終行]
a = np.loadtxt("0012.csv", delimiter=",", skiprows=17, usecols = (1, 2))[4000:4000+N]

#特定の列のみで計算
#1列目のみ指定する場合、[:, 0]
f = a[:, 0]-a[:, 1]

#フーリエ変換
F = np.fft.fft(f)



# 正規化 + 交流成分2倍
F = F/(N/2)
F[0] = F[0]/2

#周波数の値を表示させる
for i in range(display_range):
    print("周波数:{0:.2f}  振幅：{1:.2f}".format(freq[i], np.abs(F[i])))



# 配列Fをコピー
F2 = F.copy()

# ローパスフィル処理（カットオフ周波数を超える帯域の周波数信号を0にする）
F2[(freq > fc)] = 0

#周波数の値を表示させる
for i in range(display_range):
    print("（フィルタあり）周波数:{0:.2f}  振幅：{1:.2f}".format(freq[i], np.abs(F2[i])))

# 高速逆フーリエ変換（時間信号に戻す）
f2 = np.fft.ifft(F2)


# 振幅を元のスケールに戻す
f2 = np.real(f2*N)

# グラフ表示
fig = plt.figure(figsize=(10.0, 8.0))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 時間信号（元）
plt.subplot(221)
plt.plot(t, f, label='f(n)')
plt.xlabel("Time", fontsize=12)
plt.ylabel("Signal", fontsize=12)
plt.grid()
leg = plt.legend(loc=1, fontsize=15)
leg.get_frame().set_alpha(1)

# 周波数信号(元)
plt.subplot(222)
plt.plot(freq, np.abs(F), label='|F(k)|')
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
#X軸の最大値を固定する
plt.xlim([0,x_range])
plt.grid()
leg = plt.legend(loc=1, fontsize=15)
leg.get_frame().set_alpha(1)

# 時間信号(処理後)
plt.subplot(223)
plt.plot(t, f2, label='f2(n)')
plt.xlabel("Time", fontsize=12)
plt.ylabel("Signal", fontsize=12)
plt.grid()
leg = plt.legend(loc=1, fontsize=15)
leg.get_frame().set_alpha(1)

# 周波数信号(処理後)
plt.subplot(224)
plt.plot(freq, np.abs(F2), label='|F2(k)|')
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
#X軸の最大値を固定する
plt.xlim([0,x_range])
plt.grid()
leg = plt.legend(loc=1, fontsize=15)
leg.get_frame().set_alpha(1)

plt.show()
