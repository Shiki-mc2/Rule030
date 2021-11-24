# -*- coding: utf-8 -*-

import time
import cv2
import numpy as np
from numba import jit

@jit
def plot(row, col, ppu, cmg):
    img = np.full((row*ppu, col*ppu, 3), 0, dtype=np.uint8)
    blank = int(ppu/7)
    for i in range(row):
        for j in range(col):
            if cmg[i,j] == 1:
                img[i*ppu:(i+1)*ppu-blank, j*ppu:(j+1)*ppu-blank, 0] = 210
                img[i*ppu:(i+1)*ppu-blank, j*ppu:(j+1)*ppu-blank, 1] = 210
                img[i*ppu:(i+1)*ppu-blank, j*ppu:(j+1)*ppu-blank, 2] = 150
    return img

@jit
def run(row, col, cmg, i):
    if [cmg[i,-1], cmg[i,0],cmg[i,1]] in [[0, 0, 1],[0,1,0],[0,1,1],[1,0,0]]:
        cmg[i+1, 0] = 1
    if [cmg[i,col-2], cmg[i,-1],cmg[i,0]] in [[0, 0, 1],[0,1,0],[0,1,1],[1,0,0]]:
        cmg[i+1, col-1] = 1
    
    for j in range(1, col-1):
        state = list(cmg[i, j-1:j+2])
        if state in [[0, 0, 1],[0,1,0],[0,1,1],[1,0,0]]:
            cmg[i+1, j] = 1
    
    return cmg
            
def main():
    fps  = 30
    ppu  = 10
    row_cal = int(960/ppu)
    col_cal = int(int(1920/ppu))
    n       = row_cal
    row_img = row_cal*ppu
    col_img = col_cal*ppu
      
    wname = "Rule030"
    fourcc  = cv2.VideoWriter_fourcc(*"h264")
    video  = cv2.VideoWriter(f"{wname}.mp4", fourcc, fps, (col_img, row_img))
    
    cmg = np.zeros((row_cal, col_cal),dtype=np.uint8)
    cmg[0,col_cal//2] = 1
    
    sw1 = time.perf_counter()

    for i in range(n):
        img = plot(row_cal, col_cal, ppu, cmg)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        video.write(img)
         
        sw2 = time.perf_counter()
        print(f"{i:4d}", f"{(sw2-sw1):.4f}")
        sw1 = sw2
        if i < n - 1:
            cmg = run(row_cal, col_cal, cmg, i)
            
    video.release()
    cv2.imwrite(f"{wname}.png", img)
if __name__ == "__main__":
    main()